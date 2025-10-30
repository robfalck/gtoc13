"""
Physical and mission constants for GTOC13.

This module contains all constants used throughout the GTOC13 trajectory optimization framework.
"""

import jax.numpy as jnp

# Basic astronomical and time constants
KMPAU = 149597870.691  # km per AU
MU_ALTAIRA = 139348062043.343  # km^3/s^2 (gravitational parameter of Altaira)
DAY = 86400.0  # seconds per day
YEAR = 365.25 * DAY  # seconds per year

# Derived distance and time units
KMPDU = KMPAU  # 1 AU in km (Distance Unit)
SPTU = jnp.sqrt(KMPDU**3 / MU_ALTAIRA)  # Time unit in seconds
YPTU = SPTU / YEAR  # Years per time unit

# Solar sail parameters
C_FLUX = 5.4026e-6  # N/m^2 at 1 AU (solar radiation pressure)
R0 = 1.0 * KMPAU  # Reference distance (1 AU in km)
SAIL_AREA = 15000.0  # m^2 (solar sail area)
SPACECRAFT_MASS = 500.0  # kg (spacecraft mass)
