"""orbital_sim - a two-body orbital mechanics and attitude control simulator."""

from .constants import J2_EARTH, MU_EARTH, OMEGA_EARTH, R_EARTH
from .elements import OrbitalElements, coe_to_rv, kepler_propagate, rv_to_coe
from .maneuvers import (bi_elliptic, combined_plane_change_dv, hohmann,
                        plane_change_dv, tsiolkovsky_propellant)
from .perturbations import (AtmosphericDrag, J2Perturbation, j2_secular_rates,
                            sun_synchronous_inclination)
from .propagator import Trajectory, propagate
from .attitude import PIDGains, Spacecraft, quat_from_euler_deg, simulate_attitude
from .mission import (Burn, Coast, CoastToArgLat, CoastToNode, Hohmann, Mission,
                      PlaneChange)

__version__ = "1.0.0"
