"""Bridge beam-search JSON output into the Dymos solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gtoc13.constants import DAY, YEAR
from gtoc13.dymos_solver.dymos_solver2 import (
    get_dymos_serial_solver_problem,
    set_initial_guesses,
    create_solution,
)


def _days_to_gtoc_years(days: float) -> float:
    return (days * DAY) / YEAR


def _load_payload(json_path: Path) -> dict[str, Any]:
    if not json_path.is_file():
        raise FileNotFoundError(f"bs_lambert output not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _pick_solution(payload: dict[str, Any], rank: int) -> dict[str, Any]:
    solutions = payload.get("solutions") or []
    if not solutions:
        raise ValueError("JSON payload does not contain any solutions.")
    idx = max(0, min(len(solutions) - 1, rank - 1))
    return solutions[idx]


def _extract_epochs(solution: dict[str, Any]) -> Sequence[float]:
    encounters = solution.get("encounters") or []
    if not encounters:
        raise ValueError("Selected solution has no encounters to replay.")
    epochs = []
    for enc in encounters:
        epoch = enc.get("epoch_days")
        if epoch is None:
            raise ValueError("Encounter entry is missing 'epoch_days'.")
        epochs.append(float(epoch))
    return encounters, epochs


def _durations_from_epochs(
    epochs_days: Sequence[float],
    start_epoch_days: float,
) -> list[float]:
    durations_years: list[float] = []
    prev_epoch = start_epoch_days
    for epoch in epochs_days:
        dt_days = epoch - prev_epoch
        if dt_days < 0:
            raise ValueError("Encounter epochs are not in chronological order.")
        durations_years.append(_days_to_gtoc_years(dt_days))
        prev_epoch = epoch
    return durations_years


def _bodies_from_encounters(encounters: Sequence[dict[str, Any]]) -> list[int]:
    bodies: list[int] = []
    for enc in encounters:
        body_id = enc.get("body_id")
        if body_id is None:
            raise ValueError("Encounter entry is missing 'body_id'.")
        bodies.append(int(body_id))
    return bodies


def run_from_json(json_path: Path, *, rank: int, num_nodes: int, solve: bool) -> None:
    payload = _load_payload(json_path)
    solution = _pick_solution(payload, rank)
    encounters, epochs = _extract_epochs(solution)

    config = payload.get("config") or {}
    start_epoch_days = config.get("start_epoch_days")
    if start_epoch_days is None:
        start_epoch_days = epochs[0]

    epoch_years = [days / 365.25 for days in epochs]
    dt = _durations_from_epochs(epochs, float(start_epoch_days))
    bodies = _bodies_from_encounters(encounters)

    if bodies and bodies[0] == 0:
        if len(bodies) == 1:
            raise ValueError("Encounter list only contains the interstellar start body (id 0).")
        print("Encounter sequence starts with body id 0; skipping it for Dymos replay.")
        bodies = bodies[1:]
        epoch_years = epoch_years[1:]
        if dt:
            dt = dt[1:]

    if len(dt) != len(bodies):
        raise ValueError(
            f"Duration/body mismatch ({len(dt)} durations vs {len(bodies)} bodies).",
        )

    t0_years = _days_to_gtoc_years(float(start_epoch_days))

    print(
        f"Running Dymos solver for {len(bodies)} bodies from {json_path.name} "
        f"(rank {rank}, num_nodes={num_nodes}).",
    )
    bodies_str = " ".join(str(b) for b in bodies)
    flyby_str = " ".join(f"{val:.6f}" for val in epoch_years)
    node_str = " ".join(str(num_nodes) for _ in bodies)
    print(
        "Equivalent CLI:\n"
        f"  python -m gtoc13.dymos_solver "
        f"--bodies {bodies_str} --flyby-times {flyby_str} "
        f"--t0 {t0_years:.6f} --num-nodes {node_str}"
    )
    if not solve:
        print("Skipping solver run (use --solve to execute the Dymos optimization).")
        return

    _run_serial_solver(
        bodies=bodies,
        flyby_times=epoch_years,
        t0_years=t0_years,
        num_nodes=num_nodes,
    )


def _run_serial_solver(
    *,
    bodies: Sequence[int],
    flyby_times: Sequence[float],
    t0_years: float,
    num_nodes: int,
) -> None:
    controls = [0] * len(bodies)
    num_node_list = [num_nodes] * len(bodies)

    prob = get_dymos_serial_solver_problem(
        bodies=bodies,
        controls=controls,
        num_nodes=num_node_list,
        warm_start=False,
        default_opt_prob=True,
    )
    prob.setup()

    set_initial_guesses(
        prob,
        bodies=bodies,
        flyby_times=np.asarray(flyby_times, dtype=float),
        t0=t0_years,
        controls=controls,
    )

    result = prob.run_driver()
    if result.success:
        create_solution(prob, bodies)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a bs_lambert JSON solution inside the Dymos solver.",
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to a JSON file produced by gtoc13.path_finding.beam.bs_lambert.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="1-indexed rank inside the JSON 'solutions' array (default: 1).",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=20,
        help="Number of transcription nodes per arc (mirrors --num-nodes in the dymos CLI).",
    )
    parser.add_argument(
        "--solve",
        action="store_true",
        help="Actually run the Dymos solver; otherwise only print the CLI command.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    run_from_json(
        args.json_file,
        rank=args.rank,
        num_nodes=args.num_nodes,
        solve=args.solve,
    )


if __name__ == "__main__":
    main()
