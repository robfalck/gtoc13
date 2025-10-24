from .orbital_elements import OrbitalElements
from .spacecraft_state import SpacecraftState

from .astrodynamics import (
    # Constants
    KMPAU,
    MU_ALTAIRA,
    DAY,
    YEAR,
    C_FLUX,
    R0,
    SAIL_AREA,
    SPACECRAFT_MASS,

    # Functions
    solve_kepler,
    elements_to_cartesian,
    solar_sail_acceleration,
    keplerian_derivatives,
    solar_sail_derivatives,
    compute_v_infinity,
    patched_conic_flyby,
    seasonal_penalty,
    flyby_velocity_penalty,
    time_bonus,
    compute_score,
)

from .solution import (
    # Solution models
    StatePoint,
    FlybyArc,
    ConicArc,
    PropagatedArc,
    GTOC13Solution,

    # Convenience functions
    create_flyby,
    create_conic,
    create_propagated,
)

from .lambert import (
    # Lambert solver
    lambert,
    lambert_universal_variables,
    lambert_delta_v,
    get_body_state,
)

from .bodies import (
    # Body class
    Body,
    load_bodies_data,
    bodies_data
)

# Alias for compatibility with existing code
AU = KMPAU

__all__ = [
    # Constants
    "AU",
    "KMPAU",
    "MU_ALTAIRA",
    "DAY",
    "YEAR",
    "C_FLUX",
    "R0",
    "SAIL_AREA",
    "SPACECRAFT_MASS",

    # Named tuples
    "OrbitalElements",
    "SpacecraftState",

    # Functions
    "solve_kepler",
    "elements_to_cartesian",
    "solar_sail_acceleration",
    "keplerian_derivatives",
    "solar_sail_derivatives",
    "compute_v_infinity",
    "patched_conic_flyby",
    "seasonal_penalty",
    "flyby_velocity_penalty",
    "time_bonus",
    "compute_score",

    # Solution models
    "StatePoint",
    "FlybyArc",
    "ConicArc",
    "PropagatedArc",
    "GTOC13Solution",

    # Convenience functions
    "create_flyby",
    "create_conic",
    "create_propagated",

    # Lambert solver
    "lambert",
    "lambert_universal_variables",
    "lambert_delta_v",
    "get_body_state",

    # Bodies
    "Body",
    "load_bodies_data",
    "bodies_data"
]
