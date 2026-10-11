"""Physical constants used throughout the simulator.

Units: kilometres, seconds, radians (unless a name says otherwise).
Values follow WGS-84 / EGM-96 conventions as used in Vallado,
"Fundamentals of Astrodynamics and Applications".
"""

import math

MU_EARTH = 398600.4418          # Earth gravitational parameter [km^3/s^2]
R_EARTH = 6378.137              # Earth equatorial radius [km]
J2_EARTH = 1.08262668e-3        # Earth oblateness coefficient [-]
OMEGA_EARTH = 7.2921159e-5      # Earth rotation rate [rad/s]
G0 = 9.80665e-3                 # Standard gravity [km/s^2]

DAY = 86400.0                   # Solar day [s]
TROPICAL_YEAR_DAYS = 365.2422   # Tropical year [days]

# Nodal precession rate a sun-synchronous orbit needs: one full turn per year.
SUN_SYNC_RAAN_RATE = 2.0 * math.pi / (TROPICAL_YEAR_DAYS * DAY)   # [rad/s]
