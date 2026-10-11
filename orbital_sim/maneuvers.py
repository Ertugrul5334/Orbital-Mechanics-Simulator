"""Impulsive manoeuvres: Hohmann and bi-elliptic transfers, plane changes.

Closed-form results (textbook equations) live next to helpers that apply a
burn to a state vector, so the analytic delta-v can be checked against a
full numerical simulation of the same manoeuvre.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .constants import MU_EARTH


def circular_speed(r: float, mu: float = MU_EARTH) -> float:
    return math.sqrt(mu / r)


def vis_viva(r: float, a: float, mu: float = MU_EARTH) -> float:
    """Orbital speed at radius r on an orbit of semi-major axis a."""
    return math.sqrt(mu * (2.0 / r - 1.0 / a))


@dataclass(frozen=True)
class HohmannTransfer:
    r1: float
    r2: float
    a_transfer: float
    e_transfer: float
    dv1: float          # signed: + = prograde (raising), - = retrograde (lowering)
    dv2: float
    tof: float          # time of flight = half the transfer-orbit period [s]

    @property
    def dv_total(self) -> float:
        return abs(self.dv1) + abs(self.dv2)

    def __str__(self):
        return (f"Hohmann {self.r1:.1f} km -> {self.r2:.1f} km: dv1={self.dv1:+.4f} km/s, "
                f"dv2={self.dv2:+.4f} km/s, total={self.dv_total:.4f} km/s, "
                f"TOF={self.tof / 3600:.3f} h")


def hohmann(r1: float, r2: float, mu: float = MU_EARTH) -> HohmannTransfer:
    """Two-burn Hohmann transfer between coplanar circular orbits of radii r1, r2."""
    a_t = 0.5 * (r1 + r2)
    e_t = abs(r2 - r1) / (r1 + r2)
    dv1 = vis_viva(r1, a_t, mu) - circular_speed(r1, mu)
    dv2 = circular_speed(r2, mu) - vis_viva(r2, a_t, mu)
    tof = math.pi * math.sqrt(a_t ** 3 / mu)
    return HohmannTransfer(r1, r2, a_t, e_t, dv1, dv2, tof)


@dataclass(frozen=True)
class BiEllipticTransfer:
    r1: float
    r2: float
    rb: float
    dv1: float
    dv2: float
    dv3: float
    tof: float

    @property
    def dv_total(self) -> float:
        return abs(self.dv1) + abs(self.dv2) + abs(self.dv3)


def bi_elliptic(r1: float, r2: float, rb: float, mu: float = MU_EARTH) -> BiEllipticTransfer:
    """Three-burn bi-elliptic transfer via intermediate apoapsis rb (>= r2)."""
    a1, a2 = 0.5 * (r1 + rb), 0.5 * (r2 + rb)
    dv1 = vis_viva(r1, a1, mu) - circular_speed(r1, mu)
    dv2 = vis_viva(rb, a2, mu) - vis_viva(rb, a1, mu)
    dv3 = circular_speed(r2, mu) - vis_viva(r2, a2, mu)
    tof = math.pi * (math.sqrt(a1 ** 3 / mu) + math.sqrt(a2 ** 3 / mu))
    return BiEllipticTransfer(r1, r2, rb, dv1, dv2, dv3, tof)


def plane_change_dv(v: float, delta_i: float) -> float:
    """Single-burn pure plane change at speed v by angle delta_i [rad]."""
    return 2.0 * v * math.sin(abs(delta_i) / 2.0)


def combined_plane_change_dv(v1: float, v2: float, delta_i: float) -> float:
    """One burn that changes speed v1 -> v2 and inclination by delta_i [rad]."""
    return math.sqrt(v1 ** 2 + v2 ** 2 - 2.0 * v1 * v2 * math.cos(delta_i))


# --- Applying burns to state vectors -----------------------------------------
def rtn_frame(r, v):
    """Unit vectors (radial, transverse/along-track, normal) for a state."""
    r_hat = r / np.linalg.norm(r)
    n_hat = np.cross(r, v)
    n_hat /= np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, r_hat)
    return r_hat, t_hat, n_hat


def apply_burn_rtn(r, v, dv_rtn):
    """Apply an impulsive delta-v given in RTN components [km/s]."""
    R, T, N = rtn_frame(r, v)
    dv = dv_rtn[0] * R + dv_rtn[1] * T + dv_rtn[2] * N
    return r, v + dv, dv


def apply_prograde_burn(r, v, dv: float):
    """Burn of magnitude dv along the velocity vector (negative = retrograde)."""
    dv_vec = dv * v / np.linalg.norm(v)
    return r, v + dv_vec, dv_vec


def _east_north(r):
    r_hat = r / np.linalg.norm(r)
    east = np.cross([0.0, 0.0, 1.0], r_hat)
    if np.linalg.norm(east) < 1e-12:
        raise ValueError("Heading undefined over the poles.")
    east /= np.linalg.norm(east)
    north = np.cross(r_hat, east)
    return r_hat, east, north


def velocity_for_inclination(r, v, new_i: float, new_speed: float | None = None):
    """Velocity at a node that gives inclination `new_i` [rad].

    Must be called at (or very near) a node crossing, where the heading
    angle measured from local east equals the inclination. The radial
    velocity component is kept; the horizontal speed is preserved unless
    `new_speed` (a horizontal speed, e.g. circular speed) is given.
    """
    r_hat, east, north = _east_north(r)
    vr = np.dot(v, r_hat)
    ve, vn = np.dot(v, east), np.dot(v, north)
    v_h = new_speed if new_speed is not None else math.hypot(ve, vn)
    sign = 1.0 if vn >= 0 else -1.0          # ascending (+) / descending (-) node
    return vr * r_hat + v_h * (math.cos(new_i) * east + sign * math.sin(new_i) * north)


def apply_plane_change(r, v, new_i: float, new_speed: float | None = None):
    """Rotate the velocity at a node so the orbit has inclination new_i."""
    v_new = velocity_for_inclination(r, v, new_i, new_speed)
    return r, v_new, v_new - v


def tsiolkovsky_propellant(dv: float, m0: float, isp: float, g0: float = 9.80665e-3) -> float:
    """Propellant mass [same unit as m0] needed for delta-v dv [km/s]."""
    return m0 * (1.0 - math.exp(-dv / (isp * g0)))
