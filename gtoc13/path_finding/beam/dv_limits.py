"""
Helpers for computing dynamic Δv limits during Lambert pruning.

Currently hosts the solar-sail inspired bound used when the beam search runs
with ``dv_mode="dynamic"``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = ["max_transfer_dv_solar_sail"]


def max_transfer_dv_solar_sail(
    r_depart_km: Iterable[float],
    r_arrive_km: Iterable[float],
    tof_days: float,
    *,
    factor: float = 0.25,
    a1au_mps2: float = 3.24156e-4,
) -> float:
    """
    Approximate per-leg Δv cap (km/s) for a solar sail using an average heliocentric radius.

    The bound assumes continuous thrust with magnitude scaled by 1/r^2:

        Δv ≈ a(r_avg) * TOF * factor

    where ``a(r) = a1au_mps2 / r^2`` with ``r`` measured in astronomical units.

    Args:
        r_depart_km: Heliocentric departure position [km].
        r_arrive_km: Heliocentric arrival position [km].
        tof_days: Time of flight for the leg [days].
        factor: Efficiency scale (0, 1]; default 0.25 from GTOC13 heuristics.
        a1au_mps2: Sail acceleration at 1 AU [m/s²].

    Returns:
        float: Δv cap in km/s (0.0 if inputs are invalid).
    """

    if tof_days <= 0.0 or factor <= 0.0 or a1au_mps2 <= 0.0:
        return 0.0

    AU_KM = 149_597_870.7
    DAY_SECONDS = 86_400.0
    MIN_RADIUS_AU = 0.05

    r1 = float(np.linalg.norm(np.asarray(tuple(r_depart_km), dtype=float)))
    r2 = float(np.linalg.norm(np.asarray(tuple(r_arrive_km), dtype=float)))
    if not np.isfinite(r1) or not np.isfinite(r2) or r1 <= 0.0 or r2 <= 0.0:
        return 0.0

    r1_au = max(r1 / AU_KM, MIN_RADIUS_AU)
    r2_au = max(r2 / AU_KM, MIN_RADIUS_AU)
    r_avg_au = 0.5 * (r1_au + r2_au)

    accel_mps2 = a1au_mps2 / (r_avg_au**2)
    dv_mps = accel_mps2 * (tof_days * DAY_SECONDS) * factor
    dv_km_s = dv_mps * 1e-3
    return float(max(0.0, dv_km_s))
