"""Numerical orbit propagation.

Two integrators are provided:
  * `rk4`    - a hand-written fixed-step 4th-order Runge-Kutta, so the
               numerical method is visible and its error can be studied.
  * `dop853` - SciPy's adaptive 8th-order Dormand-Prince, used by default
               for accuracy-critical runs and event detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from .constants import MU_EARTH, R_EARTH
from .elements import OrbitalElements, rv_to_coe

Perturbation = Callable[[float, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class Trajectory:
    """Time history of an orbit: t [s], r [km] (N x 3), v [km/s] (N x 3)."""
    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    mu: float = MU_EARTH
    label: str = ""
    event_times: list = field(default_factory=list)

    @property
    def final_state(self):
        return self.r[-1].copy(), self.v[-1].copy()

    @property
    def radius(self):
        return np.linalg.norm(self.r, axis=1)

    @property
    def altitude(self):
        return self.radius - R_EARTH

    @property
    def speed(self):
        return np.linalg.norm(self.v, axis=1)

    def energy(self):
        """Specific mechanical energy at every sample [km^2/s^2]."""
        return 0.5 * self.speed ** 2 - self.mu / self.radius

    def angular_momentum(self):
        return np.linalg.norm(np.cross(self.r, self.v), axis=1)

    def elements(self, every: int = 1) -> list[OrbitalElements]:
        return [rv_to_coe(self.r[k], self.v[k], self.mu) for k in range(0, len(self.t), every)]

    def element_history(self, every: int = 1) -> dict[str, np.ndarray]:
        """Orbital elements over time as arrays (angles in degrees, unwrapped)."""
        els = self.elements(every)
        out = {
            "t": self.t[::every],
            "a": np.array([e.a for e in els]),
            "e": np.array([e.e for e in els]),
            "i": np.degrees([e.i for e in els]),
            "raan": np.degrees(np.unwrap([e.raan for e in els])),
            "argp": np.degrees(np.unwrap([e.argp for e in els])),
            "nu": np.degrees(np.unwrap([e.nu for e in els])),
        }
        return out

    @staticmethod
    def concatenate(parts: Sequence["Trajectory"], label: str = "") -> "Trajectory":
        t = np.concatenate([p.t for p in parts])
        r = np.vstack([p.r for p in parts])
        v = np.vstack([p.v for p in parts])
        return Trajectory(t, r, v, parts[0].mu, label)


def equations_of_motion(mu: float, perturbations: Sequence[Perturbation] = ()):
    """Return f(t, y) for y = [r, v] including any perturbing accelerations."""
    def f(t, y):
        r, v = y[:3], y[3:]
        a = -mu * r / np.linalg.norm(r) ** 3
        for p in perturbations:
            a = a + p(t, r, v)
        return np.concatenate((v, a))
    return f


def rk4(f, t0: float, y0: np.ndarray, t_end: float, dt: float):
    """Classic fixed-step 4th-order Runge-Kutta. Returns (t, y) arrays."""
    n = int(math.ceil((t_end - t0) / dt))
    t = np.empty(n + 1)
    y = np.empty((n + 1, len(y0)))
    t[0], y[0] = t0, y0
    for k in range(n):
        h = min(dt, t_end - t[k])
        tk, yk = t[k], y[k]
        k1 = f(tk, yk)
        k2 = f(tk + h / 2, yk + h / 2 * k1)
        k3 = f(tk + h / 2, yk + h / 2 * k2)
        k4 = f(tk + h, yk + h * k3)
        y[k + 1] = yk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t[k + 1] = tk + h
    return t, y


def _reentry_event(t, y):
    return np.linalg.norm(y[:3]) - (R_EARTH + 100.0)   # Karman line
_reentry_event.terminal = True
_reentry_event.direction = -1


def propagate(r0, v0, duration: float, *, t0: float = 0.0, dt: float = 10.0,
              perturbations: Sequence[Perturbation] = (), method: str = "dop853",
              mu: float = MU_EARTH, rtol: float = 1e-11, atol: float = 1e-11,
              events=None, stop_at_reentry: bool = True, label: str = "") -> Trajectory:
    """Propagate an orbit for `duration` seconds.

    Output is sampled every `dt` seconds. With method="rk4", `dt` is also the
    integration step. `events` (DOP853 only) are SciPy-style event functions;
    a terminal event ends the propagation early.
    """
    y0 = np.concatenate((np.asarray(r0, float), np.asarray(v0, float)))
    f = equations_of_motion(mu, perturbations)
    t_end = t0 + duration

    if method == "rk4":
        t, y = rk4(f, t0, y0, t_end, dt)
        return Trajectory(t, y[:, :3], y[:, 3:], mu, label)

    if method != "dop853":
        raise ValueError(f"Unknown method {method!r}; use 'dop853' or 'rk4'.")

    ev = list(events or [])
    if stop_at_reentry:
        ev.append(_reentry_event)
    n = max(2, int(math.ceil(duration / dt)) + 1)
    t_eval = np.linspace(t0, t_end, n)
    sol = solve_ivp(f, (t0, t_end), y0, method="DOP853", t_eval=t_eval,
                    rtol=rtol, atol=atol, events=ev or None, dense_output=bool(ev))
    if not sol.success:
        raise RuntimeError(sol.message)
    t, y = sol.t, sol.y.T
    event_times = [float(te) for times in (sol.t_events or []) for te in times]
    if sol.status == 1:                      # stopped by a terminal event
        te = sol.t_events and max((tt[-1] for tt in sol.t_events if len(tt)), default=None)
        if te is not None and (len(t) == 0 or t[-1] < te):
            t = np.append(t, te)
            y = np.vstack((y, sol.sol(te)))
    return Trajectory(t, y[:, :3], y[:, 3:], mu, label, event_times)


def argument_of_latitude_event(u_target_rad: float):
    """Event that fires when the argument of latitude passes `u_target_rad`."""
    cu, su = math.cos(u_target_rad), math.sin(u_target_rad)

    def g(t, y):
        r, v = y[:3], y[3:]
        h = np.cross(r, v)
        h /= np.linalg.norm(h)
        n = np.cross([0.0, 0.0, 1.0], h)
        nn = np.linalg.norm(n)
        n = n / nn if nn > 1e-12 else np.array([1.0, 0.0, 0.0])
        rh = r / np.linalg.norm(r)
        cos_u = np.dot(n, rh)
        sin_u = np.dot(np.cross(h, n), rh)
        return sin_u * cu - cos_u * su       # = sin(u - u_target)
    g.terminal = True
    g.direction = 1
    return g
