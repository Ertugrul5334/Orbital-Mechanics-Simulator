"""Multi-manoeuvre mission planning with full numerical propagation.

A mission is a list of steps (coasts and impulsive burns). The planner
propagates the spacecraft through every step, optionally with J2 and drag,
and keeps a delta-v budget, propellant budget (Tsiolkovsky) and timeline.

Example
-------
    mission = Mission(OrbitalElements.circular(300, i_deg=28.5), mass_kg=1000, isp_s=320)
    mission.add(CoastToNode()).add(Hohmann(r_geo, final_inclination_deg=0.0))
    result = mission.run()
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .constants import MU_EARTH
from .elements import OrbitalElements, coe_to_rv, rv_to_coe
from .maneuvers import (apply_burn_rtn, apply_plane_change, apply_prograde_burn,
                        circular_speed, hohmann, tsiolkovsky_propellant, vis_viva)
from .propagator import Trajectory, argument_of_latitude_event, propagate


@dataclass
class BurnRecord:
    name: str
    time: float
    dv: float
    propellant_kg: float
    mass_after_kg: float


@dataclass
class MissionResult:
    trajectory: Trajectory
    segments: list[Trajectory]
    burns: list[BurnRecord]
    final_elements: OrbitalElements
    initial_mass_kg: float

    @property
    def total_dv(self) -> float:
        return sum(b.dv for b in self.burns)

    @property
    def total_propellant(self) -> float:
        return sum(b.propellant_kg for b in self.burns)

    @property
    def duration(self) -> float:
        return float(self.trajectory.t[-1] - self.trajectory.t[0])

    def summary(self) -> str:
        lines = [f"{'Burn':<38}{'t [h]':>9}{'dv [m/s]':>11}{'prop [kg]':>11}{'mass [kg]':>11}"]
        for b in self.burns:
            lines.append(f"{b.name:<38}{b.time / 3600:>9.3f}{b.dv * 1000:>11.1f}"
                         f"{b.propellant_kg:>11.1f}{b.mass_after_kg:>11.1f}")
        lines.append(f"{'TOTAL':<38}{self.duration / 3600:>9.3f}{self.total_dv * 1000:>11.1f}"
                     f"{self.total_propellant:>11.1f}")
        lines.append(f"Final orbit: {self.final_elements}")
        return "\n".join(lines)


class _Context:
    def __init__(self, mission: "Mission"):
        self.m = mission
        self.t = 0.0
        self.r, self.v = coe_to_rv(mission.initial)
        self.mass = mission.mass_kg
        self.segments: list[Trajectory] = []
        self.burns: list[BurnRecord] = []

    def coast(self, duration, label="coast", events=None):
        traj = propagate(self.r, self.v, duration, t0=self.t, dt=self.m.output_dt,
                         perturbations=self.m.perturbations, mu=self.m.mu,
                         events=events, label=label)
        self.segments.append(traj)
        self.t = float(traj.t[-1])
        self.r, self.v = traj.final_state
        return traj

    def burn(self, name, v_new):
        dv = float(np.linalg.norm(v_new - self.v))
        prop = tsiolkovsky_propellant(dv, self.mass, self.m.isp_s) if self.m.isp_s else 0.0
        self.mass -= prop
        self.v = v_new
        self.burns.append(BurnRecord(name, self.t, dv, prop, self.mass))


# --- Mission steps ------------------------------------------------------------
@dataclass
class Coast:
    duration_s: float

    def execute(self, ctx: _Context):
        ctx.coast(self.duration_s, f"coast {self.duration_s / 60:.0f} min")


@dataclass
class CoastToArgLat:
    """Coast until the argument of latitude reaches `u_deg` (0 = ascending node)."""
    u_deg: float = 0.0

    def execute(self, ctx: _Context):
        el = rv_to_coe(ctx.r, ctx.v, ctx.m.mu)
        ev = argument_of_latitude_event(math.radians(self.u_deg))
        if abs(ev(0.0, np.concatenate((ctx.r, ctx.v)))) < 1e-9:
            return                                          # already there
        ctx.coast(1.5 * el.period, f"coast to u={self.u_deg:g} deg", events=[ev])


def CoastToNode(descending: bool = False) -> CoastToArgLat:
    return CoastToArgLat(180.0 if descending else 0.0)


@dataclass
class Hohmann:
    """Hohmann transfer from the current (near-circular) orbit to radius r_target.

    Burn 1 is targeted with a shooting method so the numerically propagated
    far apsis hits r_target even with perturbations; burn 2 fires when the
    spacecraft actually reaches that apsis and circularises the orbit.

    If `final_inclination_deg` is given, the inclination change is combined with
    the circularisation burn at the far apsis (cheapest place to do it, where
    the spacecraft is slowest). Start at a node so that apsis is also a node.
    """
    r_target: float
    final_inclination_deg: float | None = None
    transfer_inclination_deg: float | None = None   # optional plane change in burn 1

    target_numerically: bool = True                  # shooting method on burn 1
    tolerance_km: float = 0.01

    def _velocity_after_burn1(self, ctx, speed):
        if self.transfer_inclination_deg is None:
            _, v_new, _ = apply_prograde_burn(ctx.r, ctx.v, speed - np.linalg.norm(ctx.v))
            return v_new
        _, v_new, _ = apply_plane_change(ctx.r, _strip_radial(ctx.r, ctx.v),
                                         math.radians(self.transfer_inclination_deg), speed)
        return v_new

    def _far_apsis(self, ctx, v_new, raising, tof):
        """Propagate burn-1 state to the far apsis; return (r_apsis, trajectory)."""
        ev = _apsis_event(raising)
        traj = propagate(ctx.r, v_new, 1.5 * tof, t0=ctx.t, dt=ctx.m.output_dt,
                         perturbations=ctx.m.perturbations, mu=ctx.m.mu, events=[ev],
                         label="Hohmann transfer arc")
        return float(np.linalg.norm(traj.r[-1])), traj

    def execute(self, ctx: _Context):
        r1 = float(np.linalg.norm(ctx.r))
        plan = hohmann(r1, self.r_target, ctx.m.mu)
        raising = self.r_target > r1
        # Burn 1: start from the two-body (vis-viva) speed ...
        speed = vis_viva(r1, plan.a_transfer, ctx.m.mu)
        r_apsis, arc = self._far_apsis(ctx, self._velocity_after_burn1(ctx, speed), raising, plan.tof)
        # ... then correct it with a secant iteration so that the *numerically
        # propagated* far apsis (including J2/drag) lands on the target radius.
        if self.target_numerically and abs(r_apsis - self.r_target) > self.tolerance_km:
            s0, f0 = speed, r_apsis - self.r_target
            s1 = speed + (1e-4 if raising else -1e-4)
            for _ in range(20):
                r_apsis, arc = self._far_apsis(ctx, self._velocity_after_burn1(ctx, s1), raising, plan.tof)
                f1 = r_apsis - self.r_target
                if abs(f1) < self.tolerance_km or f1 == f0:
                    break
                s0, s1, f0 = s1, s1 - f1 * (s1 - s0) / (f1 - f0), f1
            speed = s1
        name = (f"Hohmann burn 1 (r={r1:.0f} km)" if self.transfer_inclination_deg is None
                else f"Burn 1 + plane change to {self.transfer_inclination_deg:g} deg")
        ctx.burn(name, self._velocity_after_burn1(ctx, speed))
        ctx.segments.append(arc)
        ctx.t = float(arc.t[-1])
        ctx.r, ctx.v = arc.final_state
        # Burn 2: circularise (and optionally change plane) at the far apsis.
        r2 = float(np.linalg.norm(ctx.r))
        vc = circular_speed(r2, ctx.m.mu)
        v_h = _strip_radial(ctx.r, ctx.v)
        if self.final_inclination_deg is None:
            ctx.burn(f"Hohmann burn 2 (r={r2:.0f} km)", vc * v_h / np.linalg.norm(v_h))
        else:
            _, v_new, _ = apply_plane_change(ctx.r, v_h, math.radians(self.final_inclination_deg), vc)
            ctx.burn(f"Burn 2 + plane change to {self.final_inclination_deg:g} deg", v_new)


def _apsis_event(apoapsis: bool):
    """Terminal event at apoapsis (r.v goes + to -) or periapsis (- to +)."""
    def g(t, y):
        return float(np.dot(y[:3], y[3:]))
    g.terminal = True
    g.direction = -1 if apoapsis else 1
    return g


def _strip_radial(r, v):
    """Velocity with its radial component removed (circularisation target)."""
    r_hat = r / np.linalg.norm(r)
    return v - np.dot(v, r_hat) * r_hat


@dataclass
class PlaneChange:
    """Pure inclination change, performed at the next node crossing."""
    new_inclination_deg: float
    name: str | None = None

    def execute(self, ctx: _Context):
        CoastToNode().execute(ctx)
        _, v_new, _ = apply_plane_change(ctx.r, ctx.v, math.radians(self.new_inclination_deg))
        ctx.burn(self.name or f"Plane change to {self.new_inclination_deg:g} deg", v_new)


@dataclass
class Burn:
    """Arbitrary impulsive burn in RTN components [km/s]."""
    dv_rtn: tuple
    name: str = "RTN burn"

    def execute(self, ctx: _Context):
        _, v_new, _ = apply_burn_rtn(ctx.r, ctx.v, np.asarray(self.dv_rtn, float))
        ctx.burn(self.name, v_new)


@dataclass
class Mission:
    initial: OrbitalElements
    mass_kg: float = 1000.0
    isp_s: float = 320.0
    perturbations: list = field(default_factory=list)
    output_dt: float = 30.0
    mu: float = MU_EARTH
    steps: list = field(default_factory=list)

    def add(self, step) -> "Mission":
        self.steps.append(step)
        return self

    def run(self) -> MissionResult:
        ctx = _Context(self)
        for step in self.steps:
            step.execute(ctx)
        traj = Trajectory.concatenate(ctx.segments, "mission") if ctx.segments else \
            Trajectory(np.array([0.0]), ctx.r[None], ctx.v[None], self.mu)
        return MissionResult(traj, ctx.segments, ctx.burns,
                             rv_to_coe(ctx.r, ctx.v, self.mu), self.mass_kg)
