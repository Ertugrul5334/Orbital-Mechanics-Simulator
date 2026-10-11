"""Ground tracks: rotate ECI positions into the Earth-fixed frame."""

from __future__ import annotations

import numpy as np

from .constants import OMEGA_EARTH


def eci_to_ecef(r_eci: np.ndarray, t: np.ndarray, gmst0_rad: float = 0.0) -> np.ndarray:
    """Rotate ECI positions (N x 3) into ECEF using a uniformly rotating Earth."""
    theta = gmst0_rad + OMEGA_EARTH * np.asarray(t)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = r_eci[:, 0], r_eci[:, 1], r_eci[:, 2]
    return np.column_stack((c * x + s * y, -s * x + c * y, z))


def ground_track(r_eci: np.ndarray, t: np.ndarray, gmst0_rad: float = 0.0):
    """Geocentric latitude and longitude [deg] of the sub-satellite point."""
    r_ecef = eci_to_ecef(r_eci, t, gmst0_rad)
    lat = np.degrees(np.arcsin(r_ecef[:, 2] / np.linalg.norm(r_ecef, axis=1)))
    lon = np.degrees(np.arctan2(r_ecef[:, 1], r_ecef[:, 0]))
    return lat, lon, r_ecef


def ground_track_shift_deg(period_s: float, raan_dot: float = 0.0) -> float:
    """Westward longitude shift of the ground track per orbit [deg]."""
    return np.degrees((OMEGA_EARTH - raan_dot) * period_s)
