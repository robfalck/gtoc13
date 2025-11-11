"""Shared beam-search plumbing: expansion, scoring, and dedup helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Hashable, Iterable, Optional, Sequence, Tuple

import math
import numpy as np

from gtoc13.constants import DAY, KMPAU, MU_ALTAIRA, YEAR
from gtoc13.bodies import INTERSTELLAR_BODY_ID

from .config import BASE_SEMI_MAJOR_AXES, BodyRegistry, LambertConfig
from .lambert import Encounter, InfeasibleLeg, State, resolve_lambert_leg
from .scoring import SCORING_FUNCTIONS, hohmann_bounds_for_bodies

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Proposal:
    """Cheap next-step proposal from expand_fn."""

    body: int  # candidate next target
    tof: float  # proposed time-of-flight (days)
    seed_offset: Optional[Tuple[float, float]] = None  # (dy, dz) offsets in AU for Interstellar
    is_seed: bool = False


def _orbit_period_days(body_id: int) -> float:
    a = BASE_SEMI_MAJOR_AXES.get(body_id)
    if a is None or a <= 0.0:
        raise ValueError(f"Semi-major axis unavailable for body {body_id}")
    period_seconds = 2.0 * math.pi * math.sqrt((a**3) / MU_ALTAIRA)
    return period_seconds / 86400.0


def make_expand_fn(
    config: LambertConfig,
    registry: BodyRegistry,
    *,
    allow_repeat: bool = True,
    same_body_samples: Optional[int] = None,
    seed_count: Optional[int] = None,
) -> Callable[[State], Iterable[Proposal]]:
    """Return a cheap proposal generator bound to the given configuration."""

    def expand(path: State) -> Iterable[Proposal]:
        if not path:
            return []
        last_body = path[-1].body
        current_time = path[-1].t
        samples_same = same_body_samples if same_body_samples is not None else registry.tof_sample_count

        if (
            seed_count is not None
            and seed_count > 0
            and len(path) == 1
            and last_body == INTERSTELLAR_BODY_ID
            and getattr(path[-1], "seed_offset", None) is None
        ):
            desired = max(1, int(seed_count))
            axis = 1
            while axis * axis < desired:
                axis += 1
            if axis == 1:
                grid_vals = np.array([0.0], dtype=float)
            else:
                grid_vals = np.linspace(-50.0, 50.0, axis, dtype=float)
            offsets = [(float(y), float(z)) for y in grid_vals for z in grid_vals]
            offsets.sort(key=lambda pair: (abs(pair[0]) + abs(pair[1]), pair[0], pair[1]))
            for offset in offsets[:desired]:
                yield Proposal(
                    body=INTERSTELLAR_BODY_ID,
                    tof=0.0,
                    seed_offset=offset,
                    is_seed=True,
                )
            return

        # Build a TOF grid that respects both the global sample budget and a per-body
        # angular resolution (≈1 deg of mean motion) so we avoid oversampling slow movers.
        def _build_tof_grid(target_body: int, tmin: float, tmax: float, max_samples: int) -> Optional[np.ndarray]:
            if (
                not math.isfinite(tmin)
                or not math.isfinite(tmax)
                or tmax <= tmin
            ):
                return None
            span = tmax - tmin
            count = max(2, int(max_samples))
            min_step = None
            try:
                period_days = _orbit_period_days(target_body)
            except ValueError:
                period_days = None
            if period_days is not None and period_days > 0.0:
                # Minimum step of 0.5 deg of mean motion (clipped to avoid zero step).
                min_step = max(period_days / 1440.0, 1e-6)
                degree_count = int(math.floor(span / min_step)) + 1
                if degree_count >= 2:
                    # Do not exceed the configured grid size, but shrink it when the degree
                    # sampling would skip fewer points (fast movers keep dense coverage).
                    count = min(count, max(2, degree_count))
                else:
                    # Span smaller than the angular step: just evaluate endpoints.
                    count = 2
            return np.linspace(tmin, tmax, count)

        for tgt in registry.body_ids:
            if tgt == last_body:
                if not allow_repeat:
                    continue
                try:
                    period_days = _orbit_period_days(tgt)
                except ValueError:
                    continue
                tmin = max(1e-6, 0.4 * period_days)
                tmax = 4.0 * period_days
                grid = _build_tof_grid(tgt, tmin, tmax, samples_same)
                if grid is None:
                    continue
            else:
                if last_body == INTERSTELLAR_BODY_ID:
                    tmin = 5.0 * YEAR / DAY
                    tmax = 100.0 * YEAR / DAY
                else:
                    try:
                        tmin, tmax = hohmann_bounds_for_bodies(last_body, tgt, registry)
                    except ValueError:
                        continue
                grid = _build_tof_grid(tgt, tmin, tmax, registry.tof_sample_count)
                if grid is None:
                    continue
            for tof_days in grid:
                tof_days = float(tof_days)
                if tof_days <= 0.0:
                    continue
                arrival_time = current_time + tof_days
                if config.tof_max_days is not None and arrival_time > config.tof_max_days:
                    continue
                yield Proposal(body=tgt, tof=tof_days)

    return expand


@dataclass(frozen=True)
class BoundScoreFunction:
    """Picklable callable that wraps Lambert scoring for multiprocessing."""

    config: LambertConfig
    registry: BodyRegistry
    mode: str
    scorer: Callable = field(init=False, repr=False)

    def __post_init__(self) -> None:
        scorer = SCORING_FUNCTIONS.get(self.mode)
        if scorer is None:
            valid = ",".join(sorted(SCORING_FUNCTIONS.keys()))
            raise ValueError(f"Unknown score mode '{self.mode}'. Available: {valid}")
        object.__setattr__(self, "scorer", scorer)

    def __call__(self, path: State, prop: Proposal) -> Tuple[float, State]:
        if not path:
            return float("-inf"), path
        parent = path[-1]
        if prop.is_seed:
            offset = prop.seed_offset if prop.seed_offset is not None else (0.0, 0.0)
            if parent.body != INTERSTELLAR_BODY_ID:
                return float("-inf"), path
            y_km = float(offset[0]) * KMPAU
            z_km = float(offset[1]) * KMPAU
            new_r = (parent.r[0], y_km, z_km)
            updated_parent = replace(parent, r=new_r, seed_offset=offset)
            return 0.0, path[:-1] + (updated_parent,)
        t1 = parent.t + prop.tof
        if self.config.tof_max_days is not None:
            if t1 > self.config.tof_max_days:
                return float("-inf"), path
        try:
            child_contrib, new_path = resolve_lambert_leg(
                self.config, self.registry, path, prop, self.scorer
            )
        except InfeasibleLeg:
            return float("-inf"), path
        return child_contrib, new_path


def make_score_fn(
    config: LambertConfig,
    registry: BodyRegistry,
    mode: str,
) -> Callable[[State, Proposal], Tuple[float, State]]:
    """Bind scoring mode and Lambert resolution into a callable for BeamSearch."""

    return BoundScoreFunction(config=config, registry=registry, mode=mode)


def key_fn(state: State) -> Hashable:
    """Coarse deduplication bucket keyed by encounter bins and visit history."""

    if not state:
        return ("root",)
    last = state[-1]
    try:
        period_days = _orbit_period_days(last.body)
    except ValueError:
        period_days = 0.0
    bin_width = max(1.0, 0.05 * period_days) if period_days > 0.0 else 10.0
    origin = 0.0
    tof_bin = int(math.floor((last.t - origin) / bin_width))
    vinf = -1.0 if last.vinf_in is None else last.vinf_in
    if vinf <= 0.0:
        vinf_bin = -1
    elif vinf < 1.0:
        vinf_bin = 0
    else:
        ratio = math.log(vinf / 1.0) / math.log(1.1)
        vinf_bin = 1 + int(math.floor(ratio))
    visited_bodies = tuple(sorted({enc.body for enc in state}))
    tail_bodies = tuple(enc.body for enc in state[-2:])
    offset = getattr(last, "seed_offset", None)
    if offset is None:
        offset_key = None
    else:
        offset_key = tuple(round(float(x), 6) for x in offset)
    return (last.body, visited_bodies, tail_bodies, tof_bin, vinf_bin, bin_width, offset_key)


__all__ = ["Proposal", "Vec3", "make_expand_fn", "make_score_fn", "key_fn", "BoundScoreFunction"]
