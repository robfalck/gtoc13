"""Bridge beam-search JSON output into the Dymos solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gtoc13.constants import DAY, YEAR
from gtoc13.dymos_solver.initial_guesses import set_initial_guesses
from gtoc13.dymos_solver.solve_arcs import (
    get_dymos_serial_solver_problem,
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


def _parse_control_token(raw: Any) -> Any:
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token == "r":
            return "r"
        if token in {"0", "1"}:
            return int(token)
    elif raw in (0, 1):
        return int(raw)
    raise ValueError(f"Control entries must be 0, 1, or 'r'; received '{raw}'.")


def _resolve_controls(count: int, controls: Sequence[Any] | None) -> list[Any]:
    if count <= 0:
        return []
    if not controls:
        return [0] * count
    tokens = list(controls)
    if len(tokens) == 1:
        parsed = _parse_control_token(tokens[0])
        return [parsed] * count
    if len(tokens) != count:
        raise ValueError(
            f"Expected {count} control entries but received {len(tokens)}.",
        )
    return [_parse_control_token(token) for token in tokens]


def run_from_json(
    json_path: Path,
    *,
    rank: int,
    num_nodes: int,
    solve: bool,
    controls: Sequence[Any] | None,
    objective: str,
    max_time: float,
    save_name: str | None,
) -> None:
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

    trimmed = bodies
    trimmed_epochs = epoch_years
    trimmed_dt = dt
    has_interstellar_start = bool(trimmed) and trimmed[0] == 0
    if has_interstellar_start:
        if len(bodies) == 1:
            raise ValueError("Encounter list only contains the interstellar start body (id 0).")
        print("Encounter sequence starts with body id 0; skipping it for Dymos replay.")
        trimmed = bodies[1:]
        trimmed_epochs = epoch_years[1:]
        if trimmed_dt:
            trimmed_dt = trimmed_dt[1:]

    if len(trimmed_dt) != len(trimmed):
        raise ValueError(
            f"Duration/body mismatch ({len(trimmed_dt)} durations vs {len(trimmed)} bodies).",
        )

    resolved_controls = _resolve_controls(len(trimmed), controls)

    t0_years = _days_to_gtoc_years(float(start_epoch_days))
    t0_override = t0_years if has_interstellar_start else None

    print(
        f"Running Dymos solver for {len(trimmed)} bodies from {json_path.name} "
        f"(rank {rank}, num_nodes={num_nodes}, objective={objective}, t_max={max_time}).",
    )
    bodies_str = " ".join(str(b) for b in trimmed)
    flyby_str = " ".join(f"{val:.6f}" for val in trimmed_epochs)
    node_str = " ".join(str(num_nodes) for _ in trimmed)
    control_opts = "" if all(ctrl == 0 for ctrl in resolved_controls) else \
        " --controls " + " ".join(str(c) for c in resolved_controls)
    cli_parts = [
        "Equivalent CLI:\n"
        f"  python -m gtoc13.dymos_solver "
        f"--bodies {bodies_str} --flyby-times {flyby_str} "
    ]
    if t0_override is not None:
        cli_parts.append(f"--t0 {t0_override:.6f} ")
    cli_parts.append(
        f"--num-nodes {node_str} --max-time {max_time:.3f} --obj {objective}{control_opts}"
    )
    print("".join(cli_parts))
    if not solve:
        print("Skipping solver run (use --solve to execute the Dymos optimization).")
        return

    _run_serial_solver(
        bodies=trimmed,
        flyby_times=trimmed_epochs,
        t0_years=t0_override,
        num_nodes=num_nodes,
        controls=resolved_controls,
        objective=objective,
        max_time=max_time,
        save_name=save_name,
    )


def _run_serial_solver(
    *,
    bodies: Sequence[int],
    flyby_times: Sequence[float],
    t0_years: float | None,
    num_nodes: int,
    controls: Sequence[Any],
    objective: str,
    max_time: float,
    save_name: str | None,
) -> None:
    num_node_list = [num_nodes] * len(bodies)

    prob = get_dymos_serial_solver_problem(
        bodies=bodies,
        controls=controls,
        num_nodes=num_node_list,
        warm_start=False,
        default_opt_prob=True,
        t_max=max_time,
        obj=objective,
    )
    prob.setup()

    set_initial_guesses(
        prob,
        bodies=bodies,
        flyby_times=np.asarray(flyby_times, dtype=float),
        t0=t0_years if t0_years is not None else 0.0,
        controls=controls,
    )

    result = prob.run_driver()
    success = getattr(result, "success", bool(result))
    if success:
        create_solution(prob, bodies, controls=controls, filename=save_name)


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
        "--controls",
        nargs="+",
        default=None,
        help="Optional control mode per arc (0, 1, or 'r'); a single value is broadcast to all arcs.",
    )
    parser.add_argument(
        "--objective",
        choices=("J", "E"),
        default="J",
        help="Objective used by the Dymos solver (J: mission score, E: final energy).",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=199.999,
        help="Upper bound (years) on the final epoch passed to the solver.",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Optional basename for the saved Dymos solution (otherwise auto-incremented).",
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
        controls=args.controls,
        objective=args.objective,
        max_time=args.max_time,
        save_name=args.save_name,
    )


if __name__ == "__main__":
    main()
