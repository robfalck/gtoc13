"""
Lambert-powered beam search driver used for the GTOC13 spacecraft routing experiments.

The module wires the generic :class:`BeamSearch` engine to mission-specific logic:
Lambert transfers (via PyKEP), patched-conic flyby checks, and several scoring modes.

Example
-------
Run a small beam search that expands from Planet 1 with the "simple" score:

    python -m gtoc13.path_finding.bs_lambert \\
        --beam-width 20 \\
        --max-depth 4 \\
        --start-body 1 \\
        --start-epoch 0.0 \\
        --score-mode simple \\
        --body-types planets

This produces progress updates per depth and prints the highest-scoring paths found.
"""

from typing import Optional, Tuple, Any, Dict
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gtoc13.bodies import bodies_data
from gtoc13.path_finding.beam.beam_search import BeamSearch
from gtoc13.path_finding.beam.config import (
    BODY_TYPES,
    BodyRegistry,
    LambertConfig,
    DEFAULT_DV_MAX,
    DEFAULT_SCORE_MODE,
    DEFAULT_TOF_MAX_DAYS,
    DEFAULT_TOF_SAMPLE_COUNT,
    DEFAULT_VINF_MAX,
    build_body_registry,
    make_lambert_config,
    parse_body_type_string,
)
from gtoc13.path_finding.beam.lambert import Encounter, State, ephemeris_position
from gtoc13.path_finding.beam.scoring import (
    hohmann_bounds_for_bodies as scoring_hohmann_bounds_for_bodies,
    hohmann_tof_bounds as scoring_hohmann_tof_bounds,
    mission_score as scoring_mission_score,
)
from gtoc13.path_finding.beam import io as io_utils
from gtoc13.path_finding.beam.pipeline import (
    make_expand_fn,
    make_score_fn,
    key_fn,
)

# --------------------- Data shapes ---------------------

Vec3 = Tuple[float, float, float]

# Re-export selected helpers for external modules.
mission_score = scoring_mission_score
hohmann_tof_bounds = scoring_hohmann_tof_bounds
hohmann_bounds_for_bodies = scoring_hohmann_bounds_for_bodies



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
        default=DEFAULT_DV_MAX,
        help="Maximum allowed periapsis Δv (km/s). Use a negative value to disable pruning.",
    )
    parser.add_argument(
        "--vinf-max",
        type=float,
        default=DEFAULT_VINF_MAX,
        help="Maximum allowable |v∞| (km/s) before pruning (negative disables).",
    )
    parser.add_argument(
        "--tof-max",
        type=float,
        default=DEFAULT_TOF_MAX_DAYS,
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
        "--parallel",
        choices=("thread", "process", "none"),
        default="process",
        help="Parallel scoring backend (default: process).",
    )
    parser.add_argument(
        "--score-mode",
        choices=("mission", "medium", "simple", "depth"),
        default=DEFAULT_SCORE_MODE,
        help="Scoring model: 'mission' uses compute_score; 'medium' mixes science heuristics; "
        "'simple' uses weight/TOF; 'depth' prioritizes unique, rapid legs for longer chains.",
    )
    parser.add_argument(
        "--body-types",
        default="planets,asteroids,comets",
        help="Comma-separated list of body categories to include (planets, asteroids, comets).",
    )
    parser.add_argument(
        "--no-repeat",
        action="store_true",
        help="Disallow consecutive encounters of the same body during expansion.",
    )
    parser.add_argument("--resume-file", default=None, help="Path to a prior JSON results file to resume from.")
    parser.add_argument("--resume-rank", type=int, default=1, help="Rank (1-indexed) of the solution to resume from.")
    parser.add_argument("--resume-index", type=int, default=-1, help="Encounter index within the ranked solution (-1 for last).")
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional path for JSON results (defaults to results/beam/bs_<timestamp>.json).",
    )
    parser.add_argument(
        "--no-output-file",
        action="store_true",
        help="Disable JSON results export.",
    )
    parser.add_argument(
        "--tof-samples",
        type=int,
        default=DEFAULT_TOF_SAMPLE_COUNT,
        help="Number of sampled TOFs per candidate leg.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress per-depth progress logging.",
    )
    args = parser.parse_args()
    tof_sample_count = max(1, int(args.tof_samples))
    config = make_lambert_config(args.dv_max, args.vinf_max, args.tof_max)

    raw_body_types = list(parse_body_type_string(args.body_types))
    if not raw_body_types:
        raw_body_types = ["planet", "asteroid", "comet"]
    type_aliases = {
        "planet": "planet",
        "planets": "planet",
        "asteroid": "asteroid",
        "asteroids": "asteroid",
        "comet": "comet",
        "comets": "comet",
        "small": "small",
    }
    normalized_types: set[str] = set()
    for token in raw_body_types:
        mapped = type_aliases.get(token)
        if mapped is None:
            raise SystemExit(f"Unknown body type '{token}' in --body-types.")
        normalized_types.add(mapped)
    if not normalized_types:
        normalized_types = {"planet", "asteroid", "comet"}

    registry = build_body_registry(normalized_types, tof_sample_count)
    resume_data = None
    resume_meta: Optional[dict[str, Any]] = None
    resume_source: Optional[str] = None
    resume_rank_val: Optional[int] = None
    resume_index_val: Optional[int] = None
    if args.resume_file:
        resume_path = Path(args.resume_file)
        resume_source = str(resume_path)
        resume_encounter, resolved_rank, resolved_enc_idx = io_utils.load_resume_solution(
            resume_path,
            rank=args.resume_rank,
            encounter_index=args.resume_index,
        )
        resume_meta = {
            "body": int(resume_encounter["body_id"]),
            "epoch": float(resume_encounter["epoch_days"]),
            "vinf_vec": resume_encounter.get("vinf_in_vec_km_s"),
            "rank": resolved_rank,
            "encounter_index": resolved_enc_idx,
        }
        print(
            "Resuming from",
            resume_path,
            f"(rank={resume_meta['rank']}, encounter={resume_meta['encounter_index']})",
            f"body={resume_meta['body']} epoch={resume_meta['epoch']:.3f} days",
            flush=True,
        )
        resume_data = resume_meta
        resume_rank_val = resolved_rank
        resume_index_val = resolved_enc_idx
    if resume_data is not None:
        args.start_body = resume_data["body"]
        args.start_epoch = resume_data["epoch"]
        vec = resume_data["vinf_vec"]
        if vec is not None:
            args.start_vinf = tuple(float(x) for x in vec)

    if args.start_body not in bodies_data:
        available = ", ".join(str(k) for k in sorted(bodies_data.keys()))
        raise SystemExit(f"Unknown start body {args.start_body}. Available IDs: {available}")

    if args.no_output_file:
        output_path: Optional[Path] = None
    else:
        if args.output_file:
            output_path = Path(args.output_file)
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
            output_path = (
                Path("results") / "beam" /
                f"bs_{args.score_mode}_bw{args.beam_width}_d{args.max_depth}_top{args.top_k}_{timestamp}.json"
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
                f"dv_max={config.dv_max}, vinf_max={config.vinf_max}, tof_max={config.tof_max_days}, "
                f"tof_samples={registry.tof_sample_count}, body_types={','.join(sorted(normalized_types))}"
            )
            print(summary, flush=True)
            return
        current_clock = datetime.now(timezone.utc).strftime("%H:%M:%S")
        msg = (
            f"[depth {depth}] | parents={survivors} | expansions={expansions} "
            f"| depth_time={depth_time:.0f}s | total_elapsed={(total_elapsed)/60:.2f}min | clock={current_clock}"
        )
        print(msg, flush=True)

    expand = make_expand_fn(
        config,
        registry,
        allow_repeat=not args.no_repeat,
        same_body_samples=registry.tof_sample_count,
    )
    selected_score_fn = make_score_fn(config, registry, args.score_mode)

    beam = BeamSearch(
        expand_fn=expand,
        score_fn=selected_score_fn,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        key_fn=key_fn,
        max_workers=args.max_workers,
        score_chunksize=args.score_chunksize,
        parallel_backend=None if args.parallel == "none" else args.parallel,
        progress_fn=progress_logger,
    )

    final_nodes = beam.run(root_state)
    if not final_nodes:
        print("Beam search terminated without feasible nodes.")
        return

    final_nodes.sort(key=lambda n: n.cum_score, reverse=True)
    top = final_nodes[: args.top_k]

    # Collect full solution records for optional JSON export.
    solutions_payload: list[dict[str, Any]] = []
    body_name_lookup: Dict[int, str] = {
        bid: getattr(bodies_data.get(bid), "name", str(bid)) for bid in bodies_data.keys()
    }

    print(f"Beam search complete. Displaying top {len(top)} nodes (beam width={args.beam_width}).")
    for rank, node in enumerate(top, start=1):
        path_state: State = node.state
        path_depth = len(path_state)
        print(f"\n#{rank}: score={node.cum_score:.4f} depth={path_depth} node_id={node.id}")
        if not path_state:
            print("    <empty path>")
            continue
        mission_start = path_state[0].t
        prev_t = None

        for idx, enc in enumerate(path_state):
            print(_format_encounter(enc, idx))
            leg_tof = 0.0 if prev_t is None else enc.t - prev_t
            mission_tof = enc.t - mission_start
            prev_t = enc.t
        encounters_payload, dv_sum = io_utils.serialize_encounters(
            path_state,
            registry.weights,
            body_names=body_name_lookup,
        )
        print(f"    Total periapsis Δv along path: {dv_sum:.3f} km/s")

        total_tof = path_state[-1].t - mission_start if path_state else 0.0
        solutions_payload.append(
            {
                "rank": rank,
                "node_id": node.id,
                "score": node.cum_score,
                "depth": path_depth,
                "total_tof_days": total_tof,
                "total_periapsis_dv_km_s": dv_sum,
                "encounters": encounters_payload,
            }
        )

    if output_path is not None and solutions_payload:
        io_utils.write_results(
            output_path,
            start_body=args.start_body,
            start_epoch=args.start_epoch,
            start_vinf=tuple(args.start_vinf) if args.start_vinf is not None else None,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            score_mode=args.score_mode,
            config=config,
            tof_samples=registry.tof_sample_count,
            body_types=normalized_types,
            top_k=args.top_k,
            top_nodes=solutions_payload,
            resume_source=resume_source,
            resume_rank=resume_rank_val,
            resume_index=resume_index_val,
            allow_repeat=not args.no_repeat,
        )
        print(f"\nResults written to {output_path}")


if __name__ == '__main__':
    run_cli()
