"""Utility to visualise beam-search solutions with realistic Lambert trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pykep
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection

from gtoc13.bodies import bodies_data
from gtoc13.constants import MU_ALTAIRA
from gtoc13.path_finding.beam.config import BODY_TYPES
from gtoc13.path_finding.beam.lambert import Vec3, body_state


# ---------------------------------------------------------------------------
# Plot styling helpers
# ---------------------------------------------------------------------------

BODY_COLOURS: Dict[str, str] = {
    "planet": "black",
    "asteroid": "orange",
    "comet": "blue",
    "small": "gray",
    "unknown": "gray",
}


def _infer_body_type(body_id: int, body_name: Optional[str]) -> str:
    body_type = BODY_TYPES.get(body_id)
    if body_type:
        return body_type
    if body_name:
        lower = body_name.lower()
        if "comet" in lower:
            return "comet"
        if "asteroid" in lower:
            return "asteroid"
        if "planet" in lower:
            return "planet"
    return "unknown"


def _set_equal_aspect(ax, points: np.ndarray, padding: float = 0.25) -> None:
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    span = (maxs - mins).max()
    radius = 0.5 * span if span > 0 else 1.0
    radius *= 1.0 + max(0.0, padding)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


# ---------------------------------------------------------------------------
# Lambert trajectory sampling
# ---------------------------------------------------------------------------

def _sample_leg(
    body_depart: int,
    depart_enc: dict,
    arrive_enc: dict,
    samples: int,
) -> np.ndarray:
    """Sample the heliocentric trajectory between two encounters using v∞ data."""

    r0 = np.asarray(depart_enc["position_km"], dtype=float)
    t0 = float(depart_enc["epoch_days"])
    t1 = float(arrive_enc["epoch_days"])
    dt_sec = (t1 - t0) * 86400.0

    if dt_sec <= 0.0:
        return np.stack([r0, np.asarray(arrive_enc["position_km"], dtype=float)], axis=0)

    # Recover the heliocentric velocity at departure using the stored v∞.
    _, v_body = body_state(body_depart, t0)
    vinf_out = depart_enc.get("vinf_out_vec_km_s")
    if vinf_out is None:
        return np.stack([r0, np.asarray(arrive_enc["position_km"], dtype=float)], axis=0)

    v0 = v_body + np.asarray(vinf_out, dtype=float)
    times = np.linspace(0.0, dt_sec, max(2, samples))

    arc_points: list[np.ndarray] = []
    for tau in times:
        try:
            r_tau, _ = pykep.propagate_lagrangian(r0.tolist(), v0.tolist(), float(tau), MU_ALTAIRA)
        except Exception:
            return np.stack([r0, np.asarray(arrive_enc["position_km"], dtype=float)], axis=0)
        arc_points.append(np.asarray(r_tau, dtype=float))

    return np.asarray(arc_points, dtype=float)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_solution(
    solution: dict,
    *,
    samples_per_leg: int,
    save_path: Optional[Path] = None,
    show: bool = True,
    plot_2d: bool = False,
) -> None:
    encounters: Sequence[dict] = solution.get("encounters", [])
    if not encounters:
        print("Solution has no encounters; skipping plot.")
        return

    positions = np.asarray([enc["position_km"] for enc in encounters], dtype=float)
    colours: list[str] = []
    labels: list[str] = []
    for enc in encounters:
        body_id = int(enc["body_id"])
        body_obj = bodies_data.get(body_id)
        body_name = enc.get("body_name") or getattr(body_obj, "name", str(body_id))
        body_type = _infer_body_type(body_id, body_name)
        colours.append(BODY_COLOURS.get(body_type, BODY_COLOURS["unknown"]))
        labels.append(str(body_id))

    fig = plt.figure(figsize=(8, 6))
    if plot_2d:
        ax = fig.add_subplot(111)
        ax.grid(True, linestyle="--", alpha=0.3)
    else:
        ax = fig.add_subplot(111, projection="3d")
        ax.grid(True, linestyle="--", alpha=0.3)

    # Plot propagated arcs.
    for idx in range(len(encounters) - 1):
        depart = encounters[idx]
        arrive = encounters[idx + 1]
        leg_points = _sample_leg(
            int(depart["body_id"]),
            depart,
            arrive,
            samples=samples_per_leg,
        )
        if plot_2d:
            ax.plot(leg_points[:, 0], leg_points[:, 1], color="gray", alpha=0.6, linewidth=1.2)
        else:
            ax.plot(leg_points[:, 0], leg_points[:, 1], leg_points[:, 2], color="gray", alpha=0.6, linewidth=1.2)

    # Plot encounter markers.
    if positions.size:
        colours[0] = "green"
        colours[-1] = "red"

    if plot_2d:
        for idx in range(len(encounters) - 1, -1, -1):
            pos = positions[idx]
            ax.scatter(pos[0], pos[1], c=[colours[idx]], s=60)
            ax.text(pos[0], pos[1], f"{idx}: {labels[idx]}", fontsize=8)
        ax.scatter([0.0], [0.0], color="#e0c200", s=80, edgecolors="k")
    else:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colours, s=60, depthshade=True)
        for idx, (pos, label) in enumerate(zip(positions, labels)):
            ax.text(pos[0], pos[1], pos[2], f"{idx}: {label}", fontsize=8)
        ax.scatter([0.0], [0.0], [0.0], color="#e0c200", s=80, depthshade=False, edgecolors="k")

    ax.set_title(
        f"Solution #{solution['rank']} | score={solution['score']:.3f} | depth={solution['depth']}"
    )
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    if plot_2d:
        ax.set_aspect("equal", adjustable="datalim")
    else:
        ax.set_zlabel("z (km)")
        _set_equal_aspect(ax, positions)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_solutions(json_path: Path) -> Sequence[dict]:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("solutions", [])


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Plot beam-search solutions from a JSON results file.")
    parser.add_argument("json_path", type=Path, help="Path to the JSON file produced by bs_lambert.")
    parser.add_argument(
        "--rank",
        type=str,
        default="1",
        help="Solution rank to plot (1-indexed) or 'all' to plot every solution.",
    )
    parser.add_argument(
        "--samples-per-leg",
        type=int,
        default=2000,
        help="Number of propagated samples per transfer arc (>=2).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save figures instead of showing them interactively.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Suppress interactive display (useful when saving plots).",
    )
    parser.add_argument(
        "--2d",
        action="store_true",
        help="Plot only the x-y projection (ignore z).",
    )
    args = parser.parse_args(argv)

    solutions = _load_solutions(args.json_path)
    if not solutions:
        print(f"No solutions found in {args.json_path}")
        return

    if args.rank.lower() == "all":
        selected = solutions
    else:
        try:
            rank = int(args.rank)
        except ValueError as exc:  # pragma: no cover - defensive
            raise SystemExit(f"Invalid rank '{args.rank}'. Use an integer or 'all'.") from exc
        idx = max(0, min(len(solutions) - 1, rank - 1))
        selected = [solutions[idx]]

    show = not args.no_show and args.save_dir is None
    for sol in selected:
        save_path = None
        if args.save_dir is not None:
            filename = f"solution_rank{sol['rank']:02d}.png"
            save_path = args.save_dir / filename
        _plot_solution(
            sol,
            samples_per_leg=max(2, args.samples_per_leg),
            save_path=save_path,
            show=show,
            plot_2d=args.__dict__["2d"],
        )


if __name__ == "__main__":
    main()
