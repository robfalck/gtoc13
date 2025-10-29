from dataclasses import dataclass, replace
from typing import Iterable, Optional, Tuple, Hashable
import math
import argparse
import functools
import numpy as np
from gtoc13.bodies import bodies_data
from gtoc13.lambert import lambert
from gtoc13.astrodynamics import (
    DAY,
    MU_ALTAIRA,
    compute_score,
    patched_conic_flyby,
    seasonal_penalty,
    flyby_velocity_penalty,
)
from gtoc13.path_finding.beam_search import BeamSearch

# --------------------- Data shapes ---------------------
"""
script to use the beam search with lambert for the dynamics
"""

Vec3 = Tuple[float, float, float]


def _get_body_elements(body_id: int):
    entry = bodies_data.get(body_id)
    if entry is None:
        raise ValueError(f"Unknown body id {body_id}")
    if hasattr(entry, "elements"):
        return entry.elements
    return entry


def _infer_body_type(body_id: int) -> str:
    body = bodies_data.get(body_id)
    if body is None:
        return "unknown"
    name = getattr(body, "name", "") or ""
    mu = float(getattr(body, "mu", 0.0))
    if mu > 0.0:
        return "planet"
    if name.startswith("Comet_"):
        return "comet"
    if name.startswith("Asteroid_"):
        return "asteroid"
    return "small"


BODY_TYPES = {bid: _infer_body_type(bid) for bid in bodies_data.keys()}
BASE_BODY_IDS = tuple(sorted(bodies_data.keys()))
BASE_SEMI_MAJOR_AXES = {
    bid: float(getattr(_get_body_elements(bid), "a", 0.0))
    for bid in BASE_BODY_IDS
}
BASE_BODY_WEIGHTS = {
    bid: float(getattr(bodies_data[bid], "weight", 0.0))
    for bid in BASE_BODY_IDS
}

ACTIVE_BODY_IDS: tuple[int, ...] = BASE_BODY_IDS
ACTIVE_SEMI_MAJOR_AXES = BASE_SEMI_MAJOR_AXES.copy()
ACTIVE_BODY_WEIGHTS = BASE_BODY_WEIGHTS.copy()

SUBMISSION_TIME_DAYS = 0
DV_PERIAPSIS_MAX = 1.0  # km/s threshold for admissible powered flybys
VINF_MAX = 300.0  # km/s limit for hyperbolic excess (sanity check)
TOF_MISSION_MAX_DAYS = 200.0  # global mission duration cap (days)
TOF_SAMPLE_COUNT = 200  # number of sampled TOFs between bounds
DEFAULT_SCORE_MODE = "mission"


def _activate_body_subset(types: set[str]) -> None:
    global ACTIVE_BODY_IDS, ACTIVE_SEMI_MAJOR_AXES, ACTIVE_BODY_WEIGHTS

    normalized = {t.lower() for t in types if t}
    if not normalized:
        normalized = {"planet", "asteroid", "comet", "small"}

    allowed_ids = [bid for bid in BASE_BODY_IDS if BODY_TYPES.get(bid) in normalized]

    if not allowed_ids:
        raise SystemExit("No bodies remain after applying --body-types filter.")

    ACTIVE_BODY_IDS = tuple(allowed_ids)
    ACTIVE_SEMI_MAJOR_AXES = {bid: BASE_SEMI_MAJOR_AXES[bid] for bid in allowed_ids}
    ACTIVE_BODY_WEIGHTS = {bid: BASE_BODY_WEIGHTS[bid] for bid in allowed_ids}


def _flybys_from_path(path: "State") -> list[dict]:
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
                "is_scientific": ACTIVE_BODY_WEIGHTS.get(encounter.body, 0.0) > 0.0,
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
    tmin = max(0.1 * T_H, 10.0 * DAY)
    tmax = min(5.0 * T_H, 2.0 * max(P1, P2))
    return tmin / DAY, tmax / DAY  # convert to days


def hohmann_bounds_for_bodies(body_a: int, body_b: int) -> Tuple[float, float]:
    a1 = ACTIVE_SEMI_MAJOR_AXES.get(body_a)
    a2 = ACTIVE_SEMI_MAJOR_AXES.get(body_b)
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


def _resonance_bonus(tof_seconds: float, orbital_period: Optional[float], ks=(1, 2, 3), halfwidth_frac: float = 0.08) -> float:
    if orbital_period is None or orbital_period <= 0.0:
        return 1.0
    val = 0.0
    for k in ks:
        center = k * orbital_period
        hw = max(1e-6, halfwidth_frac * center)
        val += math.exp(-((tof_seconds - center) ** 2) / (2.0 * hw * hw))
    return 0.5 + 0.5 * math.tanh(0.6 * val)


def _continuation_bonus(turn_slack: float, vinf_mag: float, vinf_scale: float = 12.0) -> float:
    slack = np.clip(turn_slack if turn_slack is not None else 0.0, 0.0, 1.0)
    gentle = 1.0 / (1.0 + math.exp((vinf_mag - vinf_scale) / 2.0))
    return 0.5 + 0.5 * (0.6 * slack + 0.4 * gentle)


def mission_score(path: "State") -> float:
    flybys = _flybys_from_path(path)
    if not flybys:
        return 0.0
    score = compute_score(
        flybys=flybys,
        body_weights=ACTIVE_BODY_WEIGHTS,
        grand_tour_achieved=False,
        submission_time_days=SUBMISSION_TIME_DAYS,
    )
    return float(np.asarray(score))


def _unit_vector(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        raise ValueError("Zero vector cannot be normalized.")
    return vec / norm


def _rotate_vector(vec: np.ndarray, axis: np.ndarray, theta: float, eps: float = 1e-12) -> np.ndarray:
    """Rotate `vec` about `axis` by angle `theta` (Rodrigues' formula)."""
    axis_norm = np.linalg.norm(axis)
    if axis_norm < eps:
        # Fallback axis: pick one orthogonal to vec
        fallback = np.array([1.0, 0.0, 0.0]) if abs(vec[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis_unit = _unit_vector(fallback)
    else:
        axis_unit = axis / axis_norm
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return (
        vec * cos_t
        + np.cross(axis_unit, vec) * sin_t
        + axis_unit * np.dot(axis_unit, vec) * (1.0 - cos_t)
    )


def powered_flyby_dv(vinf_in: np.ndarray, vinf_out: np.ndarray, mu_body: float, rp: float) -> tuple[np.ndarray, float]:
    """
    Compute the periapsis impulse (vector and magnitude) needed to connect incoming/outgoing
    asymptotic velocity vectors via a powered flyby at radius `rp`.
    """
    v_in = np.asarray(vinf_in, dtype=float)
    v_out = np.asarray(vinf_out, dtype=float)

    s_in = _unit_vector(v_in)
    s_out = _unit_vector(v_out)
    v1 = np.linalg.norm(v_in)
    v2 = np.linalg.norm(v_out)

    # Hyperbolic eccentricities implied by rp
    e1 = 1.0 + rp * v1**2 / mu_body
    e2 = 1.0 + rp * v2**2 / mu_body

    # Turn angles for the ballistic portions
    value1 = max(-1.0, min(1.0, 1.0 / e1))
    value2 = max(-1.0, min(1.0, 1.0 / e2))
    delta1 = 2.0 * math.asin(value1)
    delta2 = 2.0 * math.asin(value2)

    n_axis = np.cross(s_in, s_out)
    up_in = _unit_vector(_rotate_vector(-s_in, n_axis, +0.5 * delta1))
    up_out = _unit_vector(_rotate_vector(+s_out, n_axis, -0.5 * delta2))

    vp_in = math.sqrt(v1**2 + 2.0 * mu_body / rp)
    vp_out = math.sqrt(v2**2 + 2.0 * mu_body / rp)

    dv_vec = vp_out * up_out - vp_in * up_in
    dv_mag = float(np.linalg.norm(dv_vec))
    return dv_vec, dv_mag


def evaluate_flyby(body_id: int, vinf_in_vec: Optional[Vec3], vinf_out_vec: Vec3):
    """
    Assess flyby feasibility; return (is_valid, altitude_km, dv_mag, dv_vec).
    dv_mag/dv_vec correspond to the periapsis impulse required to achieve the desired turn.
    """
    if vinf_in_vec is None:
        return None, None, None, None

    body = bodies_data.get(body_id)
    if body is None:
        return None, None, None, None

    mu_body = float(getattr(body, "mu", 0.0))
    radius = float(getattr(body, "radius", 0.0))

    v_in = np.asarray(vinf_in_vec, dtype=float)
    v_out = np.asarray(vinf_out_vec, dtype=float)

    if np.linalg.norm(v_in) < 1e-8 or np.linalg.norm(v_out) < 1e-8:
        dv_vec = v_out - v_in
        dv_mag = float(np.linalg.norm(dv_vec))
        return False, None, dv_mag, tuple(float(x) for x in dv_vec)

    if mu_body <= 0.0 or radius <= 0.0:
        dv_vec = v_out - v_in
        dv_mag = float(np.linalg.norm(dv_vec))
        return False, None, dv_mag, tuple(float(x) for x in dv_vec)

    h_p, is_valid = patched_conic_flyby(v_in, v_out, mu_body, radius)
    altitude = float(np.asarray(h_p))
    valid_bool = bool(np.asarray(is_valid))

    rp_nominal = radius + altitude
    if not math.isfinite(rp_nominal) or rp_nominal <= 0.0:
        rp_nominal = radius * 1.1

    rp_min = radius * 1.1
    rp_max = radius * 101.0
    rp_clamped = min(max(rp_nominal, rp_min), rp_max)
    if rp_clamped <= 0.0:
        rp_clamped = max(radius, 1.0)

    if valid_bool:
        return True, altitude, 0.0, (0.0, 0.0, 0.0)

    altitude_adjusted = rp_clamped - radius
    try:
        dv_vec, dv_mag = powered_flyby_dv(v_in, v_out, mu_body, rp_clamped)
    except ValueError:
        dv_vec = v_out - v_in
        dv_mag = float(np.linalg.norm(dv_vec))
    return False, altitude_adjusted, dv_mag, tuple(float(x) for x in dv_vec)


def medium_leg_score(
    prefix: "State",
    parent: "Encounter",
    child: "Encounter",
    vinf_out_vec: Vec3,
    vinf_in_vec: Vec3,
    tof_days: float,
) -> Tuple[float, "Encounter"]:
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
    if vinf_mag <= 0.0 and vinf_in_vec is not None:
        vinf_mag = float(np.linalg.norm(np.asarray(vinf_in_vec, dtype=float)))
    vinf_mag = max(vinf_mag, 1e-6)
    vinf_bonus = _vinf_bonus(vinf_mag)

    parent_body = bodies_data.get(parent.body)
    mu_parent = float(getattr(parent_body, "mu", 0.0)) if parent_body is not None else 0.0
    radius_parent = float(getattr(parent_body, "radius", 0.0)) if parent_body is not None else 0.0
    hp_min = 0.1 * radius_parent if radius_parent > 0.0 else 0.0
    hp_max = 100.0 * radius_parent if radius_parent > 0.0 else 0.0
    slack = _turn_slack(parent.vinf_in_vec, vinf_out_vec, mu_parent, radius_parent, hp_min, hp_max)
    if slack is None:
        return 0.0, child

    base = weight * seasonal * vinf_bonus * (0.5 + 0.5 * slack)
    if base <= 0.0:
        return 0.0, child

    a_child = ACTIVE_SEMI_MAJOR_AXES.get(child.body, 0.0)
    orbital_period = (
        2.0 * math.pi * math.sqrt(a_child**3 / MU_ALTAIRA) if a_child > 0.0 else None
    )
    resonance = _resonance_bonus(tof_days * DAY, orbital_period)
    continuation = _continuation_bonus(slack, vinf_mag)

    increment = base * resonance * continuation
    updated_child = replace(child, J_total=parent.J_total + increment)
    return increment, updated_child


@dataclass(frozen=True, slots=True)
class Encounter:
    """State element at an encounter epoch with hyperbolic excess bookkeeping."""
    body: int
    t: float  # encounter epoch (days)
    r: Vec3  # position at epoch
    vinf_in: Optional[float] = None
    vinf_in_vec: Optional[Vec3] = None
    vinf_out: Optional[float] = None
    vinf_out_vec: Optional[Vec3] = None
    flyby_valid: Optional[bool] = None
    flyby_altitude: Optional[float] = None  # periapsis altitude above surface (km)
    dv_periapsis: Optional[float] = None  # required periapsis impulse magnitude (km/s)
    dv_periapsis_vec: Optional[Vec3] = None  # periapsis impulse vector (km/s)
    J_total: float = 0.0  # cumulative score up to/including this encounter


@dataclass(frozen=True, slots=True)
class Proposal:
    """Cheap next-step proposal from expand_fn."""
    body: int  # candidate next target
    tof: float  # proposed time-of-flight (days)


State = tuple[Encounter, ...]  # full path (immutable)


# --------------------- Cheap expansion ---------------------

def expand_fn(path: State) -> Iterable[Proposal]:
    """
    Enumerate cheap proposals only (no heavy physics).
    Replace (targets, TOFs) with your sampler.
    """
    last_body = path[-1].body if path else 2  # fallback if root missing
    mission_start = path[0].t if path else 0.0
    current_time = path[-1].t if path else mission_start
    for tgt in ACTIVE_BODY_IDS:
        if tgt == last_body:
            continue
        try:
            tmin, tmax = hohmann_bounds_for_bodies(last_body, tgt)
        except ValueError:
            continue
        if not math.isfinite(tmin) or not math.isfinite(tmax) or tmax <= tmin:
            continue
        tof_grid = np.linspace(tmin, tmax, TOF_SAMPLE_COUNT)
        for tof_days in tof_grid:
            tof_days = float(tof_days)
            if tof_days <= 0.0:
                continue
            arrival_time = current_time + tof_days
            total_duration = arrival_time - mission_start
            if TOF_MISSION_MAX_DAYS is not None and total_duration > TOF_MISSION_MAX_DAYS:
                continue
            yield Proposal(body=tgt, tof=tof_days)


# --------------------- Heavy scoring/resolution ---------------------

def score_fn_dispatch(mode: str, path: State, prop: Proposal):
    """
    Heavy step: run Lambert/feasibility, retro-resolve parent.vinf_out,
    build child Encounter (with t, r, vinf_in), and return (delta, new_path).
    """
    # Parent encounter (or synthetic launch if empty)
    if not path:
        t0 = 0.0
        r0 = ephemeris_position(2, t0)  # Earth at t0 (example)
        parent = Encounter(body=2, t=t0, r=r0, J_total=0.0)
        parent_path = (parent,)
    else:
        parent = path[-1]
        parent_path = path

    # Child epoch and Lambert solve
    t1 = parent.t + prop.tof
    if TOF_MISSION_MAX_DAYS is not None:
        mission_start = parent_path[0].t if parent_path else 0.0
        if (t1 - mission_start) > TOF_MISSION_MAX_DAYS:
            return float("-inf")
    try:
        child_contrib, new_path = resolve_lambert_leg(parent_path, prop, t1, mode)
    except InfeasibleLeg:
        return float("-inf")  # prune this candidate

    return child_contrib, new_path


# --------------------- Dedup key (coarse) ---------------------

def key_fn(state: State) -> Hashable:
    """
    Coarsely bucket by latest encounter to collapse near-duplicates.
    Tune bin widths to control pruning strength.
    """
    if not state:
        return ("root",)
    last = state[-1]
    return last.body


# --------------------- Ephemeris / Lambert helpers ---------------------


class InfeasibleLeg(Exception):
    """Raised when a proposed Lambert leg is infeasible or invalid."""


def body_state(body: int, t_days: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return heliocentric position and velocity of `body` at epoch `t_days`.
    Epoch is specified in days; internal ephemeris expects seconds.
    """
    body_obj = bodies_data.get(body)
    if body_obj is None:
        raise InfeasibleLeg(f"Unknown body id {body}")

    epoch_sec = t_days * DAY
    state = body_obj.get_state(epoch_sec)

    r_vec = np.asarray(state.r, dtype=float)
    v_vec = np.asarray(state.v, dtype=float)
    return r_vec, v_vec


def ephemeris_position(body: int, t_days: float) -> Vec3:
    """Return a tuple position for encounter bookkeeping."""
    r_vec, _ = body_state(body, t_days)
    return tuple(float(x) for x in r_vec)


def resolve_lambert_leg(parent_path: State, prop: Proposal, t_arrival: float, mode: str):
    """
    Use the real Lambert solver + ephemeris to build the resolved child path.
    Raises InfeasibleLeg if any feasibility condition fails.
    """
    if not parent_path:
        raise InfeasibleLeg("Parent path must contain at least one encounter.")

    parent = parent_path[-1]
    t_depart = parent.t

    if t_arrival <= t_depart:
        raise InfeasibleLeg("Arrival time must be after departure.")

    t_depart_sec = t_depart * DAY
    t_arrival_sec = t_arrival * DAY

    _, v_body_depart = body_state(parent.body, t_depart)
    _, v_body_arrival = body_state(prop.body, t_arrival)

    _, v_depart, r_arrival, v_arrive, converged = lambert(
        parent.body,
        prop.body,
        t_depart_sec,
        t_arrival_sec,
        bodies_data=bodies_data,
    )
    if not converged:
        raise InfeasibleLeg("Lambert solver did not converge.")

    vinf_out_vec = np.asarray(v_depart, dtype=float) - v_body_depart
    vinf_in_vec = np.asarray(v_arrive, dtype=float) - v_body_arrival

    vinf_out = float(np.linalg.norm(vinf_out_vec))
    vinf_in = float(np.linalg.norm(vinf_in_vec))
    vinf_out_vec_tuple = tuple(float(x) for x in vinf_out_vec)
    vinf_in_vec_tuple = tuple(float(x) for x in vinf_in_vec)

    if (
        not math.isfinite(vinf_out)
        or not math.isfinite(vinf_in)
        or (VINF_MAX is not None and (vinf_out > VINF_MAX or vinf_in > VINF_MAX))
    ):
        raise InfeasibleLeg(
            f"Hyperbolic excess exceeds limit: vinf_out={vinf_out:.3f} km/s, vinf_in={vinf_in:.3f} km/s"
        )

    flyby_valid, flyby_altitude, dv_mag, dv_vec = evaluate_flyby(
        parent.body, parent.vinf_in_vec, vinf_out_vec_tuple
    )
    if dv_mag is not None and DV_PERIAPSIS_MAX is not None and dv_mag > DV_PERIAPSIS_MAX:
        raise InfeasibleLeg(
            f"Required periapsis burn {dv_mag:.3f} km/s exceeds threshold {DV_PERIAPSIS_MAX:.3f} km/s."
        )

    parent_resolved = replace(
        parent,
        vinf_out=vinf_out,
        vinf_out_vec=vinf_out_vec_tuple,
        flyby_valid=flyby_valid,
        flyby_altitude=flyby_altitude,
        dv_periapsis=dv_mag,
        dv_periapsis_vec=dv_vec,
    )
    prefix = parent_path[:-1] + (parent_resolved,)

    provisional_child = Encounter(
        body=prop.body,
        t=t_arrival,
        r=tuple(float(x) for x in np.asarray(r_arrival, dtype=float)),
        vinf_in=vinf_in,
        vinf_in_vec=vinf_in_vec_tuple,
        vinf_out=None,
        vinf_out_vec=None,
        flyby_valid=None,
        flyby_altitude=None,
        dv_periapsis=None,
        dv_periapsis_vec=None,
        J_total=parent_resolved.J_total,  # placeholder
    )

    if mode == "mission":
        candidate_path = prefix + (provisional_child,)
        total_score = mission_score(candidate_path)
        child_contrib = total_score - parent_resolved.J_total
        child = replace(provisional_child, J_total=total_score)
    elif mode == "medium":
        child_contrib, child = medium_leg_score(
            prefix,
            parent_resolved,
            provisional_child,
            vinf_out_vec_tuple,
            vinf_in_vec_tuple,
            prop.tof,
        )
    else:
        tof_days = prop.tof
        if tof_days <= 0.0:
            raise InfeasibleLeg("Time of flight must be positive for scoring.")
        weight = ACTIVE_BODY_WEIGHTS.get(provisional_child.body, 0.0)
        vinf_mag = provisional_child.vinf_in or 1e-6
        child_contrib = weight / (tof_days * vinf_mag)
        child = replace(provisional_child, J_total=parent_resolved.J_total + child_contrib)

    return child_contrib, prefix + (child,)


def _format_encounter(enc: Encounter, idx: int) -> str:
    vinf_in = "—" if enc.vinf_in is None else f"{enc.vinf_in:.3f}"
    vinf_out = "—" if enc.vinf_out is None else f"{enc.vinf_out:.3f}"
    dv = "—" if enc.dv_periapsis is None else f"{enc.dv_periapsis:.3f}"
    flyby = "?" if enc.flyby_valid is None else ("✓" if enc.flyby_valid else "×")
    return (
        f"    [{idx}] body={enc.body:3d} t={enc.t:10.2f} d  "
        f"score={enc.J_total:10.4f}  v∞in={vinf_in:>7}  v∞out={vinf_out:>7}  "
        f"flyby={flyby}  dvp={dv}"
    )


def run_cli() -> None:
    global DV_PERIAPSIS_MAX, VINF_MAX, TOF_MISSION_MAX_DAYS
    parser = argparse.ArgumentParser(
        description="Beam search using Lambert arcs and patched-conic flybys."
    )
    parser.add_argument("--beam-width", type=int, default=5, help="Beam width (survivors per depth).")
    parser.add_argument("--max-depth", type=int, default=4, help="Maximum search depth (number of legs).")
    parser.add_argument(
        "--start-body",
        type=int,
        default=10,
        help="Starting body ID (default: 10 = PlanetX).",
    )
    parser.add_argument(
        "--start-epoch",
        type=float,
        default=0.0,
        help="Starting epoch in days.",
    )
    parser.add_argument(
        "--start-vinf",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        default=None,
        help="Optional inbound hyperbolic excess velocity vector (km/s) for the starting body.",
    )
    parser.add_argument(
        "--dv-max",
        type=float,
        default=DV_PERIAPSIS_MAX,
        help="Maximum allowed periapsis Δv (km/s). Use a negative value to disable pruning.",
    )
    parser.add_argument(
        "--vinf-max",
        type=float,
        default=VINF_MAX,
        help="Maximum allowable |v∞| (km/s) before pruning (negative disables).",
    )
    parser.add_argument(
        "--tof-max",
        type=float,
        default=TOF_MISSION_MAX_DAYS,
        help="Maximum total mission duration (days). Use a negative value to disable.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final beam nodes to display.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum workers for parallel backend.",
    )
    parser.add_argument(
        "--score-chunksize",
        type=int,
        default=64,
        help="Number of proposals per scoring task batch.",
    )
    parser.add_argument(
        "--score-mode",
        choices=("mission", "medium", "simple"),
        default=DEFAULT_SCORE_MODE,
        help="Scoring model: 'mission' uses compute_score; 'simple' uses weight/TOF per leg.",
    )
    parser.add_argument(
        "--body-types",
        default="planets,asteroids,comets",
        help="Comma-separated list of body categories to include (planets, asteroids, comets).",
    )
    parser.add_argument(
        "--tof-samples",
        type=int,
        default=TOF_SAMPLE_COUNT,
        help="Number of sampled TOFs per candidate leg.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress per-depth progress logging.",
    )
    args = parser.parse_args()
    DV_PERIAPSIS_MAX = None if args.dv_max is not None and args.dv_max < 0 else args.dv_max
    VINF_MAX = None if args.vinf_max is not None and args.vinf_max < 0 else args.vinf_max
    TOF_MISSION_MAX_DAYS = None if args.tof_max is not None and args.tof_max < 0 else args.tof_max
    global TOF_SAMPLE_COUNT
    TOF_SAMPLE_COUNT = max(1, int(args.tof_samples))

    type_aliases = {
        "planet": "planet",
        "planets": "planet",
        "asteroid": "asteroid",
        "asteroids": "asteroid",
        "comet": "comet",
        "comets": "comet",
        "small": "small",
    }
    requested_types: set[str] = set()
    for token in (args.body_types or "").split(","):
        token = token.strip().lower()
        if not token:
            continue
        mapped = type_aliases.get(token)
        if mapped is None:
            raise SystemExit(f"Unknown body type '{token}' in --body-types.")
        requested_types.add(mapped)
    if not requested_types:
        requested_types = {"planet", "asteroid", "comet"}

    _activate_body_subset(requested_types)

    if args.start_body not in bodies_data:
        available = ", ".join(str(k) for k in sorted(bodies_data.keys()))
        raise SystemExit(f"Unknown start body {args.start_body}. Available IDs: {available}")

    if args.start_body not in ACTIVE_BODY_IDS:
        body_type = BODY_TYPES.get(args.start_body, "unknown")
        allowed_list = ", ".join(sorted(requested_types))
        raise SystemExit(
            f"Start body {args.start_body} is type '{body_type}' which is excluded by --body-types ({allowed_list})."
        )

    vinf_vec_tuple: Optional[Vec3] = None
    vinf_mag: Optional[float] = None

    if args.start_vinf is not None:
        vinf_vec = np.asarray(args.start_vinf, dtype=float)
        vinf_mag = float(np.linalg.norm(vinf_vec))
        if vinf_mag < 1e-8:
            raise SystemExit("Provided start v-infinity vector has near-zero magnitude.")
        vinf_vec_tuple = tuple(float(x) for x in vinf_vec)

    start_r = ephemeris_position(args.start_body, args.start_epoch)
    root_state: State = (
        Encounter(
            body=args.start_body,
            t=float(args.start_epoch),
            r=start_r,
            vinf_in=vinf_mag,
            vinf_in_vec=vinf_vec_tuple,
            vinf_out=None,
            vinf_out_vec=None,
            flyby_valid=None,
            flyby_altitude=None,
            dv_periapsis=None,
            dv_periapsis_vec=None,
            J_total=0.0,
        ),
    )

    def progress_logger(
        depth: int,
        survivors: int,
        expansions: int,
        depth_time: float,
        total_elapsed: float,
    ) -> None:
        if args.no_progress:
            return
        if depth == 0:
            summary = (
                f"Beam search start: beam_width={args.beam_width}, max_depth={args.max_depth}, "
                f"start_body={args.start_body}, score_mode={args.score_mode}, "
                f"dv_max={args.dv_max}, vinf_max={args.vinf_max}, tof_max={args.tof_max}, "
                f"tof_samples={TOF_SAMPLE_COUNT}"
                f", body_types={args.body_types}"
            )
            print(summary, flush=True)
            return
        msg = (
            f"[depth {depth}] parents={survivors} expansions={expansions} "
            f"depth_time={depth_time:.3f}s total_elapsed={total_elapsed:.3f}s"
        )
        print(msg, flush=True)

    selected_score_fn = functools.partial(score_fn_dispatch, args.score_mode)

    beam = BeamSearch(
        expand_fn=expand_fn,
        score_fn=selected_score_fn,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        key_fn=key_fn,
        max_workers=args.max_workers,
        score_chunksize=args.score_chunksize,
        progress_fn=progress_logger,
    )

    final_nodes = beam.run(root_state)
    if not final_nodes:
        print("Beam search terminated without feasible nodes.")
        return

    final_nodes.sort(key=lambda n: n.cum_score, reverse=True)
    top = final_nodes[: args.top_k]

    print(f"Beam search complete. Displaying top {len(top)} nodes (beam width={args.beam_width}).")
    for rank, node in enumerate(top, start=1):
        path_state: State = node.state
        path_depth = len(path_state)
        print(f"\n#{rank}: score={node.cum_score:.4f} depth={path_depth} node_id={node.id}")
        if not path_state:
            print("    <empty path>")
            continue
        for idx, enc in enumerate(path_state):
            print(_format_encounter(enc, idx))

        # Aggregate periapsis Δv
        dv_sum = sum(
            enc.dv_periapsis or 0.0
            for enc in path_state
            if enc.dv_periapsis is not None
        )
        print(f"    Total periapsis Δv along path: {dv_sum:.3f} km/s")


if __name__ == '__main__':
    run_cli()
