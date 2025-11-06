from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import math
import numpy as np

from gtoc13.astrodynamics import compute_score, flyby_velocity_penalty, seasonal_penalty
from gtoc13.bodies import bodies_data
from gtoc13.constants import DAY, MU_ALTAIRA

from .config import BodyRegistry, LambertConfig, BASE_SEMI_MAJOR_AXES
from .lambert import Encounter, LambertLegMeta

Vec3 = Tuple[float, float, float]

# ---------------------------------------------------------------------------
# Heuristic constants
# ---------------------------------------------------------------------------

DEPTH_BONUS_SCALE = 0.16  # medium, depth
NOVELTY_BONUS_SCALE = 0.1  # medium, depth, simple
CONTINUATION_SLACK_WEIGHT = 0.7  # medium
CONTINUATION_GENTLE_WEIGHT = 0.3  # medium
DEPTH_MODE_ALPHA = 0.35  # depth
DEPTH_MODE_REPEAT_FACTOR = 0.7  # depth
DEPTH_MODE_QUICK_RATIO = 0.6  # depth
DEPTH_MODE_QUICK_BONUS_SCALE = 0.1  # depth


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _flybys_from_path(path: Sequence[Encounter], registry: BodyRegistry) -> list[dict]:
    flybys: list[dict] = []
    for encounter in path:
        if encounter.vinf_in is None:
            continue
        r_vec = np.asarray(encounter.r, dtype=float)
        r_norm = np.linalg.norm(r_vec)
        if r_norm <= 0.0:
            continue
        r_hat = r_vec / r_norm
        vinf_mag = encounter.vinf_in
        if vinf_mag is None and encounter.vinf_in_vec is not None:
            vinf_mag = float(np.linalg.norm(np.asarray(encounter.vinf_in_vec, dtype=float)))
        if vinf_mag is None:
            continue
        flybys.append(
            {
                "body_id": encounter.body,
                "r_hat": r_hat,
                "v_infinity": float(vinf_mag),
                "is_scientific": registry.weights.get(encounter.body, 0.0) > 0.0,
                "vinf_in_vec": encounter.vinf_in_vec,
                "vinf_out_vec": encounter.vinf_out_vec,
                "flyby_valid": encounter.flyby_valid,
                "flyby_altitude": encounter.flyby_altitude,
                "dv_periapsis": encounter.dv_periapsis,
                "dv_periapsis_vec": encounter.dv_periapsis_vec,
            }
        )
    return flybys


def hohmann_tof_bounds(a1_km: float, a2_km: float, mu: float) -> Tuple[float, float]:
    if a1_km <= 0.0 or a2_km <= 0.0:
        raise ValueError("Semi-major axes must be positive for Hohmann TOF bounds.")
    a_t = 0.5 * (a1_km + a2_km)
    T_H = math.pi * math.sqrt(a_t**3 / mu)
    P1 = 2.0 * math.pi * math.sqrt(a1_km**3 / mu)
    P2 = 2.0 * math.pi * math.sqrt(a2_km**3 / mu)
    tmin = max(0.01 * T_H, 5.0 * DAY)
    tmax = min(3.0 * T_H, 2.0 * max(P1, P2))
    return tmin / DAY, tmax / DAY


def hohmann_bounds_for_bodies(body_a: int, body_b: int, registry: BodyRegistry) -> Tuple[float, float]:
    a1 = registry.semi_major_axes.get(body_a)
    if a1 is None:
        a1 = BASE_SEMI_MAJOR_AXES.get(body_a)
    a2 = registry.semi_major_axes.get(body_b)
    if a2 is None:
        a2 = BASE_SEMI_MAJOR_AXES.get(body_b)
    if a1 is None or a1 <= 0.0 or a2 is None or a2 <= 0.0:
        raise ValueError(f"Missing or invalid semi-major axis for bodies {body_a} or {body_b}.")
    return hohmann_tof_bounds(a1, a2, MU_ALTAIRA)


def _seasonal_factor(r_hat: np.ndarray, prev_dirs: list[np.ndarray]) -> float:
    if prev_dirs:
        prev = np.stack(prev_dirs, axis=0)
    else:
        prev = np.zeros((0, 3))
    return float(np.asarray(seasonal_penalty(r_hat, prev)))


def _vinf_bonus(vinf_mag: float) -> float:
    return float(np.asarray(flyby_velocity_penalty(vinf_mag)))


def _turn_slack(
    vinf_in_vec: Optional[Vec3],
    vinf_out_vec: Optional[Vec3],
    mu_body: float,
    radius: float,
    hp_min: float,
    hp_max: float,
) -> Optional[float]:
    if vinf_in_vec is None or vinf_out_vec is None:
        return 1.0
    if mu_body <= 0.0 or radius <= 0.0:
        return 1.0
    vinf_in = np.asarray(vinf_in_vec, dtype=float)
    vinf_out = np.asarray(vinf_out_vec, dtype=float)
    norm_in = np.linalg.norm(vinf_in)
    norm_out = np.linalg.norm(vinf_out)
    if norm_in < 1e-9 or norm_out < 1e-9:
        return 1.0
    cos_delta = np.dot(vinf_in, vinf_out) / (norm_in * norm_out)
    cos_delta = np.clip(cos_delta, -1.0, 1.0)
    delta_req = math.acos(cos_delta)

    def max_turn(hp: float) -> float:
        rp = radius + max(hp, 0.0)
        s = mu_body / (rp * norm_in**2 + mu_body)
        s = np.clip(s, 0.0, 1.0)
        return 2.0 * math.asin(s)

    delta_max_lo = max_turn(hp_min)
    if delta_req > delta_max_lo + 1e-12:
        return None
    delta_max_hi = max_turn(hp_max)
    rng = max(1e-6, delta_max_lo - delta_max_hi)
    slack = (delta_max_lo - delta_req) / rng
    return float(np.clip(slack, 0.0, 1.0))


def _continuation_bonus(turn_slack: float, vinf_mag: float, vinf_scale: float = 12.0) -> float:
    slack = np.clip(turn_slack if turn_slack is not None else 0.0, 0.0, 1.0)
    gentle = 1.0 / (1.0 + math.exp((vinf_mag - vinf_scale) / 1.5))
    mix = CONTINUATION_SLACK_WEIGHT * slack + CONTINUATION_GENTLE_WEIGHT * gentle
    return 0.6 + 0.4 * mix


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def mission_score(config: LambertConfig, registry: BodyRegistry, path: Sequence[Encounter]) -> float:
    flybys = _flybys_from_path(path, registry)
    if not flybys:
        return 0.0
    score = compute_score(
        flybys=flybys,
        body_weights=registry.weights,
        grand_tour_achieved=False,
        submission_time_days=config.submission_time_days,
    )
    return float(np.asarray(score))


def score_leg_mission(
    config: LambertConfig,
    registry: BodyRegistry,
    prefix: Sequence[Encounter],
    parent: Encounter,
    child: Encounter,
    meta: LambertLegMeta,
) -> Tuple[float, Encounter]:
    candidate_path = tuple(prefix) + (child,)
    total_score = mission_score(config, registry, candidate_path)
    if config.tof_max_days:
        tof_days = float(getattr(meta.proposal, "tof", 0.0))
        remaining_budget = float(config.tof_max_days - child.t)
        effective_remaining = max(remaining_budget, 1e-6)
        scale = max(tof_days / effective_remaining, 1e-6)
        total_score /= scale
    child_contrib = total_score - parent.J_total
    updated_child = replace(child, J_total=total_score)
    return child_contrib, updated_child


def score_leg_mission_raw(
    config: LambertConfig,
    registry: BodyRegistry,
    prefix: Sequence[Encounter],
    parent: Encounter,
    child: Encounter,
    meta: LambertLegMeta,
) -> Tuple[float, Encounter]:
    candidate_path = tuple(prefix) + (child,)
    total_score = mission_score(config, registry, candidate_path)
    child_contrib = total_score - parent.J_total
    updated_child = replace(child, J_total=total_score)
    return child_contrib, updated_child


def score_leg_medium(
    config: LambertConfig,
    registry: BodyRegistry,
    prefix: Sequence[Encounter],
    parent: Encounter,
    child: Encounter,
    meta: LambertLegMeta,
) -> Tuple[float, Encounter]:
    tof_days = float(getattr(meta.proposal, "tof", 0.0))
    if tof_days <= 0.0:
        return 0.0, child

    weight = registry.weights.get(child.body)
    if weight is None:
        body = bodies_data.get(child.body)
        weight = float(getattr(body, "weight", 0.0)) if body is not None else 0.0

    r_vec = np.asarray(child.r, dtype=float)
    r_norm = np.linalg.norm(r_vec)
    r_hat = r_vec / r_norm if r_norm > 0.0 else None

    prev_dirs: list[np.ndarray] = []
    if r_hat is not None:
        for enc in prefix:
            if enc.body != child.body:
                continue
            r_prev = np.asarray(enc.r, dtype=float)
            norm_prev = np.linalg.norm(r_prev)
            if norm_prev > 0.0:
                prev_dirs.append(r_prev / norm_prev)

    seasonal = _seasonal_factor(r_hat, prev_dirs) if r_hat is not None else 1.0

    vinf_mag = child.vinf_in or 0.0
    if vinf_mag <= 0.0 and meta.vinf_in_vec is not None:
        vinf_mag = float(np.linalg.norm(np.asarray(meta.vinf_in_vec, dtype=float)))
    vinf_mag = max(vinf_mag, 1e-6)
    vinf_bonus = _vinf_bonus(vinf_mag)

    parent_body = bodies_data.get(parent.body)
    mu_parent = float(getattr(parent_body, "mu", 0.0)) if parent_body is not None else 0.0
    radius_parent = float(getattr(parent_body, "radius", 0.0)) if parent_body is not None else 0.0
    hp_min = 0.1 * radius_parent if radius_parent > 0.0 else 0.0
    hp_max = 100.0 * radius_parent if radius_parent > 0.0 else 0.0
    slack = _turn_slack(parent.vinf_in_vec, meta.vinf_out_vec, mu_parent, radius_parent, hp_min, hp_max)
    if slack is None:
        return 0.0, child

    base = weight * seasonal * vinf_bonus * (0.5 + 0.5 * slack)
    if base <= 0.0:
        return 0.0, child

    if config.tof_max_days:
        remaining_budget = float(config.tof_max_days - child.t)
        effective_remaining = max(remaining_budget, 1e-6)
        scale = max(tof_days / effective_remaining, 1e-6)
        base /= scale

    continuation_factor = _continuation_bonus(slack, vinf_mag)
    depth_index = len(prefix)
    depth_factor = 1.0 + DEPTH_BONUS_SCALE * math.log1p(max(0, depth_index))
    visited_bodies = {enc.body for enc in prefix}
    novelty_factor = (
        1.0 + NOVELTY_BONUS_SCALE * weight if child.body not in visited_bodies else 1.0
    )

    increment = base * continuation_factor * depth_factor * novelty_factor
    updated_child = replace(child, J_total=parent.J_total + increment)
    return increment, updated_child


def score_leg_simple(
    config: LambertConfig,
    registry: BodyRegistry,
    prefix: Sequence[Encounter],
    parent: Encounter,
    child: Encounter,
    meta: LambertLegMeta,
) -> Tuple[float, Encounter]:
    tof_days = float(getattr(meta.proposal, "tof", 0.0))
    if tof_days <= 0.0:
        return 0.0, child
    weight = registry.weights.get(child.body, 0.0)
    vinf_mag = child.vinf_in or 0.0
    if vinf_mag <= 0.0 and meta.vinf_in_vec is not None:
        vinf_mag = float(np.linalg.norm(np.asarray(meta.vinf_in_vec, dtype=float)))
    vinf_mag = max(vinf_mag, 1e-6)
    base = weight / (tof_days * vinf_mag) if weight > 0.0 else 0.0
    depth_index = len(prefix)
    depth_bonus = DEPTH_BONUS_SCALE * max(1.0, float(depth_index))
    visited_bodies = {enc.body for enc in prefix}
    novelty_bonus = NOVELTY_BONUS_SCALE * weight if child.body not in visited_bodies else 0.0
    increment = base + depth_bonus + novelty_bonus
    updated_child = replace(child, J_total=parent.J_total + increment)
    return increment, updated_child


def score_leg_depth(
    config: LambertConfig,
    registry: BodyRegistry,
    prefix: Sequence[Encounter],
    parent: Encounter,
    child: Encounter,
    meta: LambertLegMeta,
) -> Tuple[float, Encounter]:
    tof_days = float(getattr(meta.proposal, "tof", 0.0))
    if tof_days <= 0.0:
        return 0.0, child
    weight = registry.weights.get(child.body, 0.0)
    vinf_mag = child.vinf_in or 0.0
    if vinf_mag <= 0.0 and meta.vinf_in_vec is not None:
        vinf_mag = float(np.linalg.norm(np.asarray(meta.vinf_in_vec, dtype=float)))
    vinf_mag = max(vinf_mag, 1e-6)
    base = weight / (tof_days * vinf_mag) if weight > 0.0 else 0.0

    depth_index = len(prefix)
    depth_multiplier = 1.0 + DEPTH_MODE_ALPHA * math.log1p(max(0, depth_index))

    quick_bonus = 0.0
    try:
        tmin, tmax = hohmann_bounds_for_bodies(parent.body, child.body, registry)
    except ValueError:
        tmin = tmax = None
    if tmin is not None and tmax is not None and tmax > tmin:
        midpoint = 0.5 * (tmin + tmax)
        quick_threshold = DEPTH_MODE_QUICK_RATIO * midpoint
        if tof_days <= quick_threshold:
            quick_bonus = DEPTH_MODE_QUICK_BONUS_SCALE * weight

    visited_bodies = {enc.body for enc in prefix}
    novelty_bonus = NOVELTY_BONUS_SCALE * weight if child.body not in visited_bodies else 0.0
    repeat_factor = 1.0 if child.body not in visited_bodies else DEPTH_MODE_REPEAT_FACTOR

    incremental = base * depth_multiplier * repeat_factor + quick_bonus + novelty_bonus
    updated_child = replace(child, J_total=parent.J_total + incremental)
    return incremental, updated_child


SCORING_FUNCTIONS: Dict[
    str,
    Callable[[LambertConfig, BodyRegistry, Sequence[Encounter], Encounter, Encounter, LambertLegMeta], Tuple[float, Encounter]],
] = {
    "mission": score_leg_mission,
    "mission-raw": score_leg_mission_raw,
    "medium": score_leg_medium,
    "simple": score_leg_simple,
    "depth": score_leg_depth,
}


__all__ = [
    "SCORING_FUNCTIONS",
    "DEPTH_BONUS_SCALE",
    "NOVELTY_BONUS_SCALE",
    "CONTINUATION_SLACK_WEIGHT",
    "CONTINUATION_GENTLE_WEIGHT",
    "DEPTH_MODE_ALPHA",
    "DEPTH_MODE_REPEAT_FACTOR",
    "DEPTH_MODE_QUICK_RATIO",
    "DEPTH_MODE_QUICK_BONUS_SCALE",
    "hohmann_tof_bounds",
    "hohmann_bounds_for_bodies",
    "mission_score",
    "score_leg_depth",
    "score_leg_medium",
    "score_leg_mission",
    "score_leg_mission_raw",
    "score_leg_simple",
]
