"""Perturbing accelerations and analytic perturbation theory.

Each perturbation is a callable `f(t, r, v) -> a` returning an extra
acceleration [km/s^2] that the propagator adds to two-body gravity.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import J2_EARTH, MU_EARTH, OMEGA_EARTH, R_EARTH, SUN_SYNC_RAAN_RATE


class J2Perturbation:
    """Earth-oblateness (J2) acceleration - the dominant LEO perturbation."""

    def __init__(self, j2: float = J2_EARTH, mu: float = MU_EARTH, radius: float = R_EARTH):
        self.j2, self.mu, self.radius = j2, mu, radius

    def __call__(self, t, r, v):
        x, y, z = r
        rn2 = x * x + y * y + z * z
        rn = math.sqrt(rn2)
        k = 1.5 * self.j2 * self.mu * self.radius ** 2 / rn ** 5
        z2 = 5.0 * z * z / rn2
        return np.array([k * x * (z2 - 1.0), k * y * (z2 - 1.0), k * z * (z2 - 3.0)])

    def __repr__(self):
        return "J2"


# Piecewise-exponential atmosphere (Vallado, Table 8-4):
# base altitude [km], base density [kg/m^3], scale height [km]
_ATMOSPHERE = np.array([
    (0, 1.225, 7.249), (25, 3.899e-2, 6.349), (30, 1.774e-2, 6.682),
    (40, 3.972e-3, 7.554), (50, 1.057e-3, 8.382), (60, 3.206e-4, 7.714),
    (70, 8.770e-5, 6.549), (80, 1.905e-5, 5.799), (90, 3.396e-6, 5.382),
    (100, 5.297e-7, 5.877), (110, 9.661e-8, 7.263), (120, 2.438e-8, 9.473),
    (130, 8.484e-9, 12.636), (140, 3.845e-9, 16.149), (150, 2.070e-9, 22.523),
    (180, 5.464e-10, 29.740), (200, 2.789e-10, 37.105), (250, 7.248e-11, 45.546),
    (300, 2.418e-11, 53.628), (350, 9.518e-12, 53.298), (400, 3.725e-12, 58.515),
    (450, 1.585e-12, 60.828), (500, 6.967e-13, 63.822), (600, 1.454e-13, 71.835),
    (700, 3.614e-14, 88.667), (800, 1.170e-14, 124.64), (900, 5.245e-15, 181.05),
    (1000, 3.019e-15, 268.00),
])


def atmospheric_density(altitude_km: float) -> float:
    """Atmospheric density [kg/m^3] from the exponential model."""
    if altitude_km < 0:
        altitude_km = 0.0
    idx = np.searchsorted(_ATMOSPHERE[:, 0], altitude_km, side="right") - 1
    h0, rho0, H = _ATMOSPHERE[idx]
    return rho0 * math.exp(-(altitude_km - h0) / H)


class AtmosphericDrag:
    """Aerodynamic drag with a co-rotating exponential atmosphere.

    Parameters
    ----------
    cd : drag coefficient (2.2 is typical for a satellite)
    area_m2 : cross-sectional area [m^2]
    mass_kg : spacecraft mass [kg]
    """

    def __init__(self, cd: float = 2.2, area_m2: float = 0.01, mass_kg: float = 1.33):
        self.cd, self.area, self.mass = cd, area_m2, mass_kg
        self.ballistic = cd * area_m2 / mass_kg          # [m^2/kg]
        self._omega = np.array([0.0, 0.0, OMEGA_EARTH])

    def __call__(self, t, r, v):
        alt = np.linalg.norm(r) - R_EARTH
        rho = atmospheric_density(alt)
        v_rel = v - np.cross(self._omega, r)              # [km/s]
        # a [m/s^2] = -0.5 rho B |v| v with v in m/s -> convert to km/s^2.
        return -0.5 * rho * self.ballistic * 1000.0 * np.linalg.norm(v_rel) * v_rel

    def __repr__(self):
        return f"Drag(Cd={self.cd}, A={self.area} m^2, m={self.mass} kg)"


# --- Analytic secular J2 rates ------------------------------------------------
def j2_secular_rates(a, e, i, mu=MU_EARTH, j2=J2_EARTH, radius=R_EARTH):
    """Secular drift rates caused by J2 [rad/s].

    Returns (raan_dot, argp_dot). These are the first-order averaged rates
    used for mission design (e.g. sun-synchronous orbits).
    """
    n = math.sqrt(mu / a ** 3)
    p = a * (1 - e ** 2)
    k = n * j2 * (radius / p) ** 2
    raan_dot = -1.5 * k * math.cos(i)
    argp_dot = 0.75 * k * (5 * math.cos(i) ** 2 - 1)
    return raan_dot, argp_dot


def sun_synchronous_inclination(a, e=0.0, mu=MU_EARTH, j2=J2_EARTH, radius=R_EARTH):
    """Inclination [rad] giving a sun-synchronous RAAN rate (~0.9856 deg/day)."""
    n = math.sqrt(mu / a ** 3)
    p = a * (1 - e ** 2)
    cos_i = -SUN_SYNC_RAAN_RATE / (1.5 * n * j2 * (radius / p) ** 2)
    if abs(cos_i) > 1:
        raise ValueError("No sun-synchronous inclination exists for this orbit.")
    return math.acos(cos_i)
