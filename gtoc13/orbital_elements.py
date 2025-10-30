"""
Orbital elements representation for celestial bodies.
"""
from typing import NamedTuple


class OrbitalElements(NamedTuple):
    """
    Keplerian orbital elements for a celestial body.

    These elements define the orbit of a body around a central mass (Altaira).
    All angular quantities are in radians.

    Attributes:
        a: Semi-major axis (km)
        e: Eccentricity (dimensionless, 0 ≤ e < 1 for elliptical orbits)
        i: Inclination relative to reference plane (radians)
        Omega: Longitude of the ascending node (radians)
        omega: Argument of periapsis (radians)
        M0: Mean anomaly at epoch t=0 (radians)

    Note:
        - For elliptical orbits: 0 ≤ e < 1
        - For parabolic orbits: e = 1
        - For hyperbolic orbits: e > 1
    """
    a: float  # semi-major axis (km)
    e: float  # eccentricity
    i: float  # inclination (rad)
    Omega: float  # longitude of ascending node (rad)
    omega: float  # argument of periapsis (rad)
    M0: float  # mean anomaly at epoch (rad)
