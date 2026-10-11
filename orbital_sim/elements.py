"""Classical orbital elements and conversions to/from Cartesian state vectors.

The six classical (Keplerian) elements are:
    a     semi-major axis                        [km]
    e     eccentricity                           [-]
    i     inclination                            [rad]
    raan  right ascension of the ascending node  [rad]
    argp  argument of periapsis                  [rad]
    nu    true anomaly                           [rad]

Circular and/or equatorial orbits make some of these undefined. Following
Vallado (Algorithm 9), rv_to_coe sets the undefined angles to zero and puts
the position angle into `nu`, so coe_to_rv(rv_to_coe(r, v)) always
reproduces the original state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .constants import MU_EARTH, R_EARTH

_TOL = 1e-10


@dataclass(frozen=True)
class OrbitalElements:
    a: float
    e: float
    i: float
    raan: float
    argp: float
    nu: float
    mu: float = MU_EARTH

    @classmethod
    def from_degrees(cls, a, e, i_deg, raan_deg, argp_deg, nu_deg, mu=MU_EARTH):
        return cls(a, e, math.radians(i_deg), math.radians(raan_deg),
                   math.radians(argp_deg), math.radians(nu_deg), mu)

    @classmethod
    def circular(cls, altitude_km, i_deg=0.0, raan_deg=0.0, u_deg=0.0, mu=MU_EARTH):
        """Circular orbit at a given altitude. `u_deg` is the argument of latitude."""
        return cls.from_degrees(R_EARTH + altitude_km, 0.0, i_deg, raan_deg, 0.0, u_deg, mu)

    # --- derived quantities -------------------------------------------------
    @property
    def p(self) -> float:
        """Semi-latus rectum [km]."""
        return self.a * (1.0 - self.e ** 2)

    @property
    def period(self) -> float:
        """Orbital period [s]."""
        return 2.0 * math.pi * math.sqrt(self.a ** 3 / self.mu)

    @property
    def mean_motion(self) -> float:
        """Mean motion [rad/s]."""
        return math.sqrt(self.mu / self.a ** 3)

    @property
    def r_periapsis(self) -> float:
        return self.a * (1.0 - self.e)

    @property
    def r_apoapsis(self) -> float:
        return self.a * (1.0 + self.e)

    @property
    def energy(self) -> float:
        """Specific orbital energy [km^2/s^2]."""
        return -self.mu / (2.0 * self.a)

    def to_rv(self):
        return coe_to_rv(self)

    def with_(self, **changes) -> "OrbitalElements":
        return replace(self, **changes)

    def as_degrees(self) -> dict:
        return {
            "a_km": self.a,
            "e": self.e,
            "i_deg": math.degrees(self.i),
            "raan_deg": math.degrees(self.raan),
            "argp_deg": math.degrees(self.argp),
            "nu_deg": math.degrees(self.nu),
        }

    def __str__(self) -> str:
        d = self.as_degrees()
        return (f"a={d['a_km']:.3f} km, e={d['e']:.6f}, i={d['i_deg']:.4f} deg, "
                f"RAAN={d['raan_deg']:.4f} deg, argp={d['argp_deg']:.4f} deg, "
                f"nu={d['nu_deg']:.4f} deg")


def _rot1(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def _rot3(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def perifocal_to_eci(raan, i, argp):
    """Rotation matrix from the perifocal (PQW) frame to ECI."""
    return _rot3(-raan) @ _rot1(-i) @ _rot3(-argp)


def coe_to_rv(el: OrbitalElements):
    """Classical orbital elements -> ECI position [km] and velocity [km/s]."""
    if el.e >= 1.0:
        raise ValueError("Only closed (elliptical) orbits are supported, e < 1.")
    p = el.p
    cnu, snu = math.cos(el.nu), math.sin(el.nu)
    r_pqw = np.array([cnu, snu, 0.0]) * p / (1.0 + el.e * cnu)
    v_pqw = np.array([-snu, el.e + cnu, 0.0]) * math.sqrt(el.mu / p)
    Q = perifocal_to_eci(el.raan, el.i, el.argp)
    return Q @ r_pqw, Q @ v_pqw


def _angle(u, v):
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return math.acos(max(-1.0, min(1.0, c)))


def rv_to_coe(r, v, mu: float = MU_EARTH) -> OrbitalElements:
    """ECI position [km] and velocity [km/s] -> classical orbital elements."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)

    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    n = np.cross([0.0, 0.0, 1.0], h)          # node vector
    nn = np.linalg.norm(n)
    e_vec = ((vn ** 2 - mu / rn) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)

    energy = vn ** 2 / 2.0 - mu / rn
    if abs(energy) < 1e-14:
        raise ValueError("Parabolic orbit: semi-major axis undefined.")
    a = -mu / (2.0 * energy)
    i = math.acos(max(-1.0, min(1.0, h[2] / hn)))

    circular = e < _TOL
    equatorial = nn < _TOL * hn

    raan = 0.0
    if not equatorial:
        raan = math.acos(max(-1.0, min(1.0, n[0] / nn)))
        if n[1] < 0:
            raan = 2 * math.pi - raan

    if not circular and not equatorial:
        argp = _angle(n, e_vec)
        if e_vec[2] < 0:
            argp = 2 * math.pi - argp
        nu = _angle(e_vec, r)
        if np.dot(r, v) < 0:
            nu = 2 * math.pi - nu
    elif circular and not equatorial:
        # Argument of latitude stored in nu.
        argp = 0.0
        nu = _angle(n, r)
        if r[2] < 0:
            nu = 2 * math.pi - nu
    elif not circular and equatorial:
        # Longitude of periapsis stored in argp.
        argp = math.atan2(e_vec[1], e_vec[0]) % (2 * math.pi)
        if h[2] < 0:
            argp = (2 * math.pi - argp) % (2 * math.pi)
        nu = _angle(e_vec, r)
        if np.dot(r, v) < 0:
            nu = 2 * math.pi - nu
    else:
        # True longitude stored in nu.
        argp = 0.0
        nu = math.atan2(r[1], r[0]) % (2 * math.pi)
        if h[2] < 0:
            nu = (2 * math.pi - nu) % (2 * math.pi)

    return OrbitalElements(a, e, i, raan, argp, nu, mu)


# --- anomaly conversions ----------------------------------------------------
def true_to_eccentric(nu, e):
    return 2.0 * math.atan2(math.sqrt(1 - e) * math.sin(nu / 2), math.sqrt(1 + e) * math.cos(nu / 2))


def eccentric_to_true(E, e):
    return 2.0 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))


def true_to_mean(nu, e):
    E = true_to_eccentric(nu, e)
    return E - e * math.sin(E)


def solve_kepler(M, e, tol=1e-14, max_iter=50):
    """Solve Kepler's equation M = E - e sin E for E (Newton-Raphson)."""
    M = math.remainder(M, 2 * math.pi)
    E = M if e < 0.8 else math.pi * math.copysign(1.0, M)
    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        E -= f / (1 - e * math.cos(E))
        if abs(f) < tol:
            break
    return E


def mean_to_true(M, e):
    return eccentric_to_true(solve_kepler(M, e), e) % (2 * math.pi)


def kepler_propagate(el: OrbitalElements, dt: float) -> OrbitalElements:
    """Analytic two-body propagation: advance the true anomaly by dt seconds.

    Used as the exact reference solution when checking the numerical integrators.
    """
    M = true_to_mean(el.nu, el.e) + el.mean_motion * dt
    return el.with_(nu=mean_to_true(M, el.e))
