from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Sequence, Tuple

import math
import numpy as np
import pykep

from gtoc13.bodies import bodies_data, INTERSTELLAR_BODY_ID
from gtoc13.constants import DAY, MU_ALTAIRA
from gtoc13.astrodynamics import patched_conic_flyby

from .config import BodyRegistry, LambertConfig, DEFAULT_DV_FACTOR
from .dv_limits import max_transfer_dv_solar_sail

MIN_DYNAMIC_DV_CAP_KM_S = 0.25

# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Encounter:
    """State element at an encounter epoch with hyperbolic excess bookkeeping."""

    body: int
    t: float  # encounter epoch (days)
    r: Vec3  # heliocentric position at epoch (km)
    vinf_in: Optional[float] = None
    vinf_in_vec: Optional[Vec3] = None
    vinf_out: Optional[float] = None
    vinf_out_vec: Optional[Vec3] = None
    flyby_valid: Optional[bool] = None
    flyby_altitude: Optional[float] = None  # periapsis altitude above surface (km)
    dv_periapsis: Optional[float] = None  # required periapsis impulse magnitude (km/s)
    dv_periapsis_vec: Optional[Vec3] = None  # periapsis impulse vector (km/s)
    dv_limit: Optional[float] = None  # pruning cap applied for this leg (km/s)
    J_total: float = 0.0  # cumulative score up to/including this encounter
    J_total_raw: float = 0.0  # cumulative mission-raw score (always tracked)
    seed_offset: Optional[Tuple[float, float]] = None  # (dy, dz) in AU for Interstellar fan-out


State = Tuple[Encounter, ...]


class InfeasibleLeg(Exception):
    """Raised when a proposed Lambert leg is infeasible or invalid."""


@dataclass(frozen=True, slots=True)
class LambertLegMeta:
    """
    Auxiliary metadata passed to the scorer for each Lambert branch under consideration.
    """

    proposal: Any
    lambert_solution: dict[str, Any]
    vinf_out_vec: Vec3
    vinf_in_vec: Vec3
    vinf_out: float
    vinf_in: float


# ScoreFn returns the incremental score and the fully-updated child encounter.
ScoreFn = Callable[
    [LambertConfig, BodyRegistry, Sequence[Encounter], Encounter, Encounter, LambertLegMeta],
    Optional[Tuple[float, Encounter]],
]


# ---------------------------------------------------------------------------
# Ephemeris helpers
# ---------------------------------------------------------------------------


def body_state(body: int, t_days: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return heliocentric position and velocity of ``body`` at epoch ``t_days``.

    The ephemeris expects seconds; we therefore convert days to seconds before
    querying :mod:`gtoc13.bodies`.
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
    """Return a plain tuple with the heliocentric position."""

    r_vec, _ = body_state(body, t_days)
    return tuple(float(x) for x in r_vec)


# ---------------------------------------------------------------------------
# Lambert and flyby primitives
# ---------------------------------------------------------------------------


def enumerate_lambert_solutions(
    body_depart: int,
    body_arrive: int,
    t_depart: float,
    t_arrive: float,
    max_revs: int = 2,
) -> list[dict[str, Any]]:
    """
    Enumerate PyKEP Lambert solutions for the leg body_depart -> body_arrive.

    PyKEP returns a single zero-revolution solution per call; the ``cw`` flag
    selects the short (False) or long (True) branch. For higher revolutions the
    solver yields multiple solutions at once. We evaluate both ``cw`` settings
    and filter duplicates to obtain a comprehensive solution set.
    """

    tof = t_arrive - t_depart
    if tof <= 0.0:
        return []

    state_depart = bodies_data[body_depart].get_state(t_depart)
    state_arrive = bodies_data[body_arrive].get_state(t_arrive)
    r1 = np.asarray(state_depart.r, dtype=float)
    r2 = np.asarray(state_arrive.r, dtype=float)

    solutions: list[dict[str, Any]] = []

    def add_solution(entry: dict[str, Any]) -> None:
        v1 = entry["v1"]
        v2 = entry["v2"]
        rev = entry["rev"]
        for existing in solutions:
            if existing["rev"] != rev:
                continue
            if np.allclose(existing["v1"], v1, atol=1e-9, rtol=0.0) and np.allclose(
                existing["v2"], v2, atol=1e-9, rtol=0.0
            ):
                return
        solutions.append(entry)

    args = (r1.tolist(), r2.tolist(), tof, MU_ALTAIRA)
    # Evaluate both short/long-way (cw False/True) branches to avoid missing solutions.
    for cw_flag in (False, True):
        try:
            lp = pykep.lambert_problem(*args, cw=cw_flag, max_revs=max_revs)
        except Exception:
            continue

        v1_list = lp.get_v1()
        v2_list = lp.get_v2()
        nrev_list: list[int] = []
        if hasattr(lp, "get_nrev"):
            try:
                nrev_list = list(lp.get_nrev())
            except Exception:
                nrev_list = []

        branch_counter: dict[tuple[bool, int], int] = {}
        for idx, (v1, v2) in enumerate(zip(v1_list, v2_list)):
            if nrev_list and idx < len(nrev_list):
                rev = int(nrev_list[idx])
            else:
                rev = 0 if idx == 0 else min(1 + (idx - 1) // 2, max_revs)

            if rev > max_revs:
                continue

            branch_idx_key = (cw_flag, rev)
            branch_idx = branch_counter.get(branch_idx_key, 0)
            branch_counter[branch_idx_key] = branch_idx + 1

            if rev == 0:
                branch = "short" if not cw_flag else "long"
            else:
                label_base = "cw" if cw_flag else "ccw"
                branch = f"{label_base}_{branch_idx}"

            add_solution(
                {
                    "rev": rev,
                    "branch": branch,
                    "cw": cw_flag,
                    "v1": np.asarray(v1, dtype=float),
                    "v2": np.asarray(v2, dtype=float),
                    "r1": r1,
                    "r2": np.asarray(r2, dtype=float),
                }
            )

    return solutions


def _requires_low_perihelion(
    r_depart: np.ndarray,
    v_depart: np.ndarray,
    r_arrive: np.ndarray,
    revolutions: int,
    rp_min_km: float,
) -> bool:
    """Return True if the Lambert arc must pass below the configured perihelion floor."""

    threshold = float(rp_min_km)
    r1 = np.asarray(r_depart, dtype=float)
    r2 = np.asarray(r_arrive, dtype=float)
    v1 = np.asarray(v_depart, dtype=float)
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    if r1_norm <= 0.0 or r2_norm <= 0.0:
        return False
    if r1_norm < threshold or r2_norm < threshold:
        return True

    h_vec = np.cross(r1, v1)
    h_norm = np.linalg.norm(h_vec)
    if h_norm <= 1e-9:
        return False

    e_vec = np.cross(v1, h_vec) / MU_ALTAIRA - r1 / r1_norm
    e = np.linalg.norm(e_vec)
    if e < 1e-8:
        # Nearly circular; radius stays close to |r|.
        return False

    peri_radius = (h_norm**2) / (MU_ALTAIRA * (1.0 + e))
    tol = max(1e-6 * threshold, 1e-3)
    if peri_radius >= threshold - tol:
        return False

    if revolutions is not None and revolutions > 0:
        return True

    def _true_anomaly(r_vec: np.ndarray) -> float:
        cos_nu = np.dot(e_vec, r_vec) / (e * np.linalg.norm(r_vec))
        cos_nu = float(np.clip(cos_nu, -1.0, 1.0))
        nu = math.acos(cos_nu)
        direction = np.dot(np.cross(e_vec, r_vec), h_vec)
        if direction < 0.0:
            nu = 2.0 * math.pi - nu
        return nu

    nu1 = _true_anomaly(r1)
    nu2 = _true_anomaly(r2)
    dnu = (nu2 - nu1) % (2.0 * math.pi)
    if math.isclose(dnu, 0.0, abs_tol=1e-9):
        return False

    wraps_peri = dnu > (2.0 * math.pi - nu1)
    return wraps_peri


def _unit_vector(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        raise ValueError("Zero vector cannot be normalized.")
    return vec / norm


def _rotate_vector(vec: np.ndarray, axis: np.ndarray, theta: float, eps: float = 1e-12) -> np.ndarray:
    """Rotate ``vec`` around ``axis`` by ``theta`` using Rodrigues' formula."""

    axis_norm = np.linalg.norm(axis)
    if axis_norm < eps:
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
    Compute the periapsis impulse (vector and magnitude) that connects
    ``vinf_in`` to ``vinf_out`` via a powered flyby at periapsis radius ``rp``.
    """

    v_in = np.asarray(vinf_in, dtype=float)
    v_out = np.asarray(vinf_out, dtype=float)

    s_in = _unit_vector(v_in)
    s_out = _unit_vector(v_out)
    v1 = np.linalg.norm(v_in)
    v2 = np.linalg.norm(v_out)

    e1 = 1.0 + rp * v1**2 / mu_body
    e2 = 1.0 + rp * v2**2 / mu_body

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
    Assess flyby feasibility; return ``(is_valid, altitude_km, dv_mag, dv_vec)``.

    ``dv_mag``/``dv_vec`` correspond to the periapsis impulse required to achieve
    the desired turn. ``None`` values indicate missing data or bodies with no
    meaningful flyby definition.
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

    altitude = float(np.asarray(h_p)) if h_p is not None else None
    valid_bool = bool(np.asarray(is_valid))

    rp_nominal = radius + (altitude if altitude is not None else 0.0)
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


# ---------------------------------------------------------------------------
# High-level resolution
# ---------------------------------------------------------------------------


def resolve_lambert_leg(
    config: LambertConfig,
    registry: BodyRegistry,
    parent_path: Sequence[Encounter],
    proposal: Any,
    score_fn: ScoreFn,
    *,
    max_revs: int = 1,
) -> Tuple[float, State]:
    """
    Solve a Lambert leg for the given proposal, evaluate each branch, and return
    the best-scoring child path.
    """

    if not parent_path:
        raise InfeasibleLeg("Parent path must contain at least one encounter.")

    parent = parent_path[-1]
    t_depart = parent.t
    t_arrival = t_depart + getattr(proposal, "tof")

    if t_arrival <= t_depart:
        raise InfeasibleLeg("Arrival time must be after departure.")

    t_depart_sec = t_depart * DAY
    t_arrival_sec = t_arrival * DAY

    _, v_body_depart = body_state(parent.body, t_depart)
    _, v_body_arrival = body_state(getattr(proposal, "body"), t_arrival)

    local_max_revs = max_revs
    if parent.body == INTERSTELLAR_BODY_ID and parent.seed_offset is not None:
        local_max_revs = 1

    solutions = enumerate_lambert_solutions(
        parent.body,
        getattr(proposal, "body"),
        t_depart_sec,
        t_arrival_sec,
        max_revs=local_max_revs,
    )
    if not solutions:
        raise InfeasibleLeg("Lambert solver did not converge.")

    best: Optional[Tuple[float, State]] = None

    for sol in solutions:
        vinf_out_vec = np.asarray(sol["v1"], dtype=float) - v_body_depart
        vinf_in_vec = np.asarray(sol["v2"], dtype=float) - v_body_arrival

        vinf_out = float(np.linalg.norm(vinf_out_vec))
        vinf_in = float(np.linalg.norm(vinf_in_vec))
        if vinf_out < 0.1:
            continue
        if parent.body == INTERSTELLAR_BODY_ID and parent.seed_offset is not None:
            # Limit lateral (y/z) components of the outbound v∞ when departing from seeded Interstellar starts.
            transverse_vinf = float(np.linalg.norm(vinf_out_vec[1:]))
            if transverse_vinf >= 15.0:
                continue
        vinf_cap = config.vinf_max
        if (
            not math.isfinite(vinf_out)
            or not math.isfinite(vinf_in)
            or (vinf_cap is not None and (vinf_out > vinf_cap or vinf_in > vinf_cap))
        ):
            continue

        vinf_out_vec_tuple = tuple(float(x) for x in vinf_out_vec)
        vinf_in_vec_tuple = tuple(float(x) for x in vinf_in_vec)

        if config.rp_min_km is not None:
            r_start = sol.get("r1")
            v_start = sol.get("v1")
            r_end = sol.get("r2")
            rev_count = int(sol.get("rev", 0)) if sol.get("rev") is not None else 0
            if (
                r_start is not None
                and v_start is not None
                and r_end is not None
                and _requires_low_perihelion(
                    r_start,
                    v_start,
                    r_end,
                    rev_count,
                    config.rp_min_km,
                )
            ):
                continue

        flyby_valid, flyby_altitude, dv_mag, dv_vec = evaluate_flyby(
            parent.body, parent.vinf_in_vec, vinf_out_vec_tuple
        )

        dv_cap: Optional[float]
        if config.dv_mode == "dynamic":
            factor = config.dv_factor if config.dv_factor is not None else DEFAULT_DV_FACTOR
            r_start = sol.get("r1")
            r_end = sol.get("r2")
            tof_days = float(getattr(proposal, "tof", 0.0))
            if r_start is None or r_end is None or tof_days <= 0.0:
                dv_cap = 0.0
            else:
                dv_cap = max_transfer_dv_solar_sail(
                    r_start,
                    r_end,
                    tof_days,
                    factor=float(factor),
                )
            if dv_cap is not None:
                dv_cap = max(dv_cap, MIN_DYNAMIC_DV_CAP_KM_S)
            if dv_cap is not None and config.dv_max is not None:
                dv_cap = min(dv_cap, config.dv_max)
        else:
            dv_cap = config.dv_max

        if dv_mag is not None and dv_cap is not None and dv_mag > dv_cap:
            continue

        parent_resolved = replace(
            parent,
            vinf_out=vinf_out,
            vinf_out_vec=vinf_out_vec_tuple,
            flyby_valid=flyby_valid,
            flyby_altitude=flyby_altitude,
            dv_periapsis=dv_mag,
            dv_periapsis_vec=dv_vec,
            dv_limit=dv_cap,
        )
        prefix = tuple(parent_path[:-1]) + (parent_resolved,)

        provisional_child = Encounter(
            body=getattr(proposal, "body"),
            t=t_arrival,
            r=tuple(float(x) for x in sol["r2"]),
            vinf_in=vinf_in,
            vinf_in_vec=vinf_in_vec_tuple,
            vinf_out=None,
            vinf_out_vec=None,
            flyby_valid=None,
            flyby_altitude=None,
            dv_periapsis=None,
            dv_periapsis_vec=None,
            J_total=parent_resolved.J_total,
        )

        meta = LambertLegMeta(
            proposal=proposal,
            lambert_solution=sol,
            vinf_out_vec=vinf_out_vec_tuple,
            vinf_in_vec=vinf_in_vec_tuple,
            vinf_out=vinf_out,
            vinf_in=vinf_in,
        )

        scored = score_fn(config, registry, prefix, parent_resolved, provisional_child, meta)
        if scored is None:
            continue
        child_contrib, scored_child = scored
        candidate_state = prefix + (scored_child,)

        if best is None or child_contrib > best[0]:
            best = (child_contrib, candidate_state)

    if best is None:
        raise InfeasibleLeg("Lambert solver did not converge.")

    return best
