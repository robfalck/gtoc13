from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from gtoc13.bodies import bodies_data

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SUBMISSION_TIME_DAYS = 0.0
DEFAULT_DV_MAX = 1.0  # km/s threshold for admissible powered flybys
DEFAULT_VINF_MAX = 100.0  # km/s limit for hyperbolic excess
DEFAULT_TOF_MAX_DAYS = 200.0 * 365.25  # ~200 years, expressed in days
DEFAULT_TOF_SAMPLE_COUNT = 200
DEFAULT_SCORE_MODE = "medium"
DEFAULT_DV_MODE = "fixed"
DEFAULT_DV_FACTOR = 0.25


@dataclass(frozen=True, slots=True)
class LambertConfig:
    dv_max: Optional[float]
    vinf_max: Optional[float]
    tof_max_days: Optional[float]
    dv_mode: str = DEFAULT_DV_MODE
    dv_factor: Optional[float] = DEFAULT_DV_FACTOR
    submission_time_days: float = DEFAULT_SUBMISSION_TIME_DAYS


def _infer_body_type(body_id: int) -> str:
    entry = bodies_data.get(body_id)
    if entry is None:
        return "unknown"
    name = getattr(entry, "name", "") or ""
    mu = float(getattr(entry, "mu", 0.0))
    if mu > 0:
        return "planet"
    if name.startswith("Comet_"):
        return "comet"
    if name.startswith("Asteroid_"):
        return "asteroid"
    return "small"


BODY_TYPES = {bid: _infer_body_type(bid) for bid in bodies_data.keys()}
BASE_BODY_IDS = tuple(sorted(bodies_data.keys()))
BASE_SEMI_MAJOR_AXES = {
    bid: float(getattr(getattr(bodies_data[bid], "elements", None), "a", getattr(bodies_data[bid], "a", 0.0)))
    for bid in BASE_BODY_IDS
}
BASE_BODY_WEIGHTS = {bid: float(getattr(bodies_data[bid], "weight", 0.0)) for bid in BASE_BODY_IDS}


@dataclass(frozen=True, slots=True)
class BodyRegistry:
    body_ids: Tuple[int, ...]
    semi_major_axes: Dict[int, float]
    weights: Dict[int, float]
    tof_sample_count: int = DEFAULT_TOF_SAMPLE_COUNT


def build_body_registry(body_types: Iterable[str], tof_sample_count: int = DEFAULT_TOF_SAMPLE_COUNT) -> BodyRegistry:
    allowed = {t.lower() for t in body_types if t}
    if not allowed:
        allowed = {"planet", "asteroid", "comet", "small"}
    body_ids = tuple(bid for bid in BASE_BODY_IDS if BODY_TYPES.get(bid) in allowed)
    if not body_ids:
        raise SystemExit("No bodies remain after applying --body-types.")
    semi_major_axes = {bid: BASE_SEMI_MAJOR_AXES[bid] for bid in body_ids}
    weights = {bid: BASE_BODY_WEIGHTS[bid] for bid in body_ids}
    return BodyRegistry(body_ids=body_ids, semi_major_axes=semi_major_axes, weights=weights, tof_sample_count=tof_sample_count)


def make_lambert_config(
    dv_max: Optional[float],
    vinf_max: Optional[float],
    tof_max_days: Optional[float],
    *,
    dv_mode: str = DEFAULT_DV_MODE,
    dv_factor: Optional[float] = DEFAULT_DV_FACTOR,
    submission_time_days: float = DEFAULT_SUBMISSION_TIME_DAYS,
) -> LambertConfig:
    """Normalize CLI-style inputs into a LambertConfig."""
    dv = None if dv_max is not None and dv_max < 0 else dv_max
    vinf = None if vinf_max is not None and vinf_max < 0 else vinf_max
    tof = None if tof_max_days is not None and tof_max_days < 0 else tof_max_days
    mode = dv_mode.lower() if dv_mode else DEFAULT_DV_MODE
    factor = dv_factor
    if factor is not None and factor <= 0.0:
        factor = 0.0
    return LambertConfig(
        dv_max=dv,
        vinf_max=vinf,
        tof_max_days=tof,
        dv_mode=mode,
        dv_factor=factor,
        submission_time_days=submission_time_days,
    )


def parse_body_type_string(body_types: str) -> Iterable[str]:
    if not body_types:
        return []
    return [token.strip().lower() for token in body_types.split(',') if token.strip()]
