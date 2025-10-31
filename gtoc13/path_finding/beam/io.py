"""I/O helpers for beam search: resume loading and JSON export."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import json

from .config import LambertConfig
from .lambert import Encounter


def serialize_encounters(
    path: Sequence[Encounter],
    body_weights: Dict[int, float],
    body_names: Optional[Dict[int, str]] = None,
) -> Tuple[list[dict[str, Any]], float]:
    encounters_payload: list[dict[str, Any]] = []
    mission_start = path[0].t if path else 0.0
    prev_t = None
    total_dv = 0.0

    for idx, enc in enumerate(path):
        leg_tof = 0.0 if prev_t is None else enc.t - prev_t
        mission_tof = enc.t - mission_start
        encounters_payload.append(
            {
                "index": idx,
                "body_id": enc.body,
                "body_name": (body_names.get(enc.body) if body_names else None),
                "epoch_days": enc.t,
                "leg_tof_days": leg_tof,
                "mission_tof_days": mission_tof,
                "position_km": [float(x) for x in enc.r],
                "vinf_in_mag_km_s": enc.vinf_in,
                "vinf_in_vec_km_s": list(enc.vinf_in_vec) if enc.vinf_in_vec is not None else None,
                "vinf_out_mag_km_s": enc.vinf_out,
                "vinf_out_vec_km_s": list(enc.vinf_out_vec) if enc.vinf_out_vec is not None else None,
                "flyby_valid": enc.flyby_valid,
                "flyby_altitude_km": enc.flyby_altitude,
                "periapsis_dv_mag_km_s": enc.dv_periapsis,
                "periapsis_dv_vec_km_s": list(enc.dv_periapsis_vec) if enc.dv_periapsis_vec is not None else None,
                "body_weight": body_weights.get(enc.body),
                "cumulative_score": enc.J_total,
            }
        )
        if enc.dv_periapsis is not None:
            total_dv += enc.dv_periapsis
        prev_t = enc.t

    return encounters_payload, total_dv


def write_results(
    output_path: Path,
    *,
    start_body: int,
    start_epoch: float,
    start_vinf: Optional[Tuple[float, float, float]],
    beam_width: int,
    max_depth: int,
    score_mode: str,
    config: LambertConfig,
    tof_samples: int,
    body_types: Iterable[str],
    top_k: int,
    top_nodes: Sequence[Dict[str, Any]],
    resume_source: Optional[str] = None,
    resume_rank: Optional[int] = None,
    resume_index: Optional[int] = None,
    allow_repeat: bool = True,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "start_body": start_body,
            "start_epoch_days": start_epoch,
            "start_vinf": list(start_vinf) if start_vinf is not None else None,
            "beam_width": beam_width,
            "max_depth": max_depth,
            "score_mode": score_mode,
            "dv_max": config.dv_max,
            "vinf_max": config.vinf_max,
            "tof_max_days": config.tof_max_days,
            "tof_samples": tof_samples,
            "body_types": ",".join(sorted(body_types)),
            "top_k": top_k,
            "resume_source": resume_source,
            "resume_rank": resume_rank,
            "resume_index": resume_index,
            "allow_repeat": allow_repeat,
        },
        "solutions": top_nodes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_resume_solution(
    resume_file: Path,
    *,
    rank: int,
    encounter_index: int,
) -> Tuple[dict[str, Any], int, int]:
    if not resume_file.is_file():
        raise SystemExit(f"Resume file not found: {resume_file}")
    with resume_file.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    solutions = payload.get("solutions") or []
    if not solutions:
        raise SystemExit("Resume file contains no solutions.")
    rank_idx = max(0, min(len(solutions) - 1, rank - 1))
    solution = solutions[rank_idx]
    encounters = solution.get("encounters") or []
    if not encounters:
        raise SystemExit("Selected solution has no encounters.")
    if encounter_index < 0:
        enc_idx = len(encounters) + encounter_index
    else:
        enc_idx = encounter_index
    if enc_idx < 0:
        enc_idx = 0
    enc_idx = max(0, min(len(encounters) - 1, enc_idx))
    return encounters[enc_idx], rank_idx + 1, enc_idx


__all__ = [
    "serialize_encounters",
    "write_results",
    "load_resume_solution",
]
