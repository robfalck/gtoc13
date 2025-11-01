"""Shared beam-search plumbing: expansion, scoring, and dedup helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable, Optional, Sequence, Tuple

import math
import numpy as np

from gtoc13.constants import MU_ALTAIRA

from .config import BASE_SEMI_MAJOR_AXES, BodyRegistry, LambertConfig
from .lambert import Encounter, InfeasibleLeg, State, resolve_lambert_leg
from .scoring import SCORING_FUNCTIONS, hohmann_bounds_for_bodies

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Proposal:
    """Cheap next-step proposal from expand_fn."""

    body: int  # candidate next target
    tof: float  # proposed time-of-flight (days)


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
) -> Callable[[State], Iterable[Proposal]]:
    """Return a cheap proposal generator bound to the given configuration."""

    def expand(path: State) -> Iterable[Proposal]:
        if not path:
            return []
        last_body = path[-1].body
        mission_start = path[0].t
        current_time = path[-1].t
        samples_same = same_body_samples if same_body_samples is not None else registry.tof_sample_count
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
                if not math.isfinite(tmin) or not math.isfinite(tmax) or tmax <= tmin:
                    continue
                tof_grid = np.linspace(tmin, tmax, max(2, samples_same))
            else:
                try:
                    tmin, tmax = hohmann_bounds_for_bodies(last_body, tgt, registry)
                except ValueError:
                    continue
                if not math.isfinite(tmin) or not math.isfinite(tmax) or tmax <= tmin:
                    continue
                tof_grid = np.linspace(tmin, tmax, registry.tof_sample_count)
            for tof_days in tof_grid:
                tof_days = float(tof_days)
                if tof_days <= 0.0:
                    continue
                arrival_time = current_time + tof_days
                total_duration = arrival_time - mission_start
                if config.tof_max_days is not None and total_duration > config.tof_max_days:
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
        t1 = parent.t + prop.tof
        if self.config.tof_max_days is not None:
            mission_start = path[0].t
            if (t1 - mission_start) > self.config.tof_max_days:
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
    """Coarsely bucket by latest encounter to collapse near-duplicates."""

    if not state:
        return ("root",)
    last = state[-1]
    tof_bin = int(round(last.t / 10.0))  # 50-day bins
    vinf = -1.0 if last.vinf_in is None else last.vinf_in
    vinf_bin = int(round(vinf / 1.0))  # 2 km/s bins
    return (last.body, tof_bin, vinf_bin)


__all__ = ["Proposal", "Vec3", "make_expand_fn", "make_score_fn", "key_fn", "BoundScoreFunction"]
