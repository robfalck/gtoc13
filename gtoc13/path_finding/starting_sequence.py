"""Multiple-segment ballistic starting-sequence optimizer.

This module builds a chain of ballistic legs that departs the -200 AU entry
plane and visits an arbitrary sequence of bodies. Each leg uses a two-point
shooting formulation: we propagate forward from the previous node and
backward from the next body for half of the segment time and enforce state
continuity at the midpoint. The optimiser (SciPy's SLSQP) adjusts the
boundary offsets, leg times, and v-infinity vectors to minimise the total
time of flight while enforcing inequality constraints that keep the forward
and backward states within user-defined tolerances. Velocity and position
mismatches are scaled independently to allow those tolerances to be tuned, and
optional flyby constraints ensure the turn angles are achievable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.integrate import solve_ivp
from scipy.optimize import Bounds, OptimizeResult, minimize

from gtoc13.bodies import bodies_data
from gtoc13.constants import KMPAU, MU_ALTAIRA, YEAR
from gtoc13.odes import ballistic_ode
from gtoc13.astrodynamics import patched_conic_flyby


AU_TO_KM = float(KMPAU)
MU_ALTAIRA_FLOAT = float(MU_ALTAIRA)
POSITION_TOL_KM = 100.0
VELOCITY_TOL_KMS = 0.01
VINF_TOL = 1e-3  # km/s tolerance for inbound/outbound v_inf magnitude match
SMOOTH_ABS_EPS = 1e-12


def _ballistic_ode_numpy(t: float, state: np.ndarray) -> np.ndarray:
    return np.asarray(ballistic_ode(t, jnp.asarray(state), (MU_ALTAIRA_FLOAT,)))


@dataclass
class SegmentResult:
    forward_state: np.ndarray
    backward_state: np.ndarray
    forward_samples: np.ndarray  # shape (N, 6)
    backward_samples: np.ndarray  # shape (N, 6)
    start_time: float
    match_time: float
    arrival_time: float


@dataclass
class PropagationResult:
    segments: List[SegmentResult]
    vinf_arrivals: List[np.ndarray]
    vinf_departures: List[np.ndarray]
    target_positions: List[np.ndarray]
    target_velocities: List[np.ndarray]
    leg_times_seconds: np.ndarray
    flyby_altitudes_km: List[float]
    flyby_valid: List[bool]

    def final_vinf_mag(self) -> float:
        return float(np.linalg.norm(self.vinf_arrivals[-1]))


class TrajectoryProblem:
    def __init__(self, body_ids: Sequence[int], scale_vector: np.ndarray):
        if len(body_ids) == 0:
            raise ValueError("At least one body is required")
        self.body_ids = tuple(int(b) for b in body_ids)
        self.bodies = [bodies_data[b] for b in self.body_ids]
        self.segment_count = len(self.body_ids)
        self.scale_vector = np.asarray(scale_vector, dtype=float)
        self._cache: dict[Tuple[float, ...], PropagationResult] = {}

    # ------------------------------------------------------------------
    # Decision-vector helpers
    # ------------------------------------------------------------------
    def to_internal(self, decision: Iterable[float]) -> np.ndarray:
        return np.asarray(decision, dtype=float) / self.scale_vector

    def to_physical(self, decision_internal: Iterable[float]) -> np.ndarray:
        return np.asarray(decision_internal, dtype=float) * self.scale_vector

    def _split_decision(self, decision: Sequence[float]) -> Tuple[
        float,
        float,
        float,
        np.ndarray,
        List[np.ndarray],
        List[np.ndarray],
    ]:
        idx = 0
        y_au = float(decision[idx]); idx += 1
        z_au = float(decision[idx]); idx += 1
        vx_kms = float(decision[idx]); idx += 1

        times_years = np.asarray(decision[idx: idx + self.segment_count], dtype=float)
        idx += self.segment_count

        departure_vinf: List[np.ndarray] = []
        for _ in range(self.segment_count - 1):
            departure_vinf.append(np.asarray(decision[idx: idx + 3], dtype=float))
            idx += 3

        arrival_vinf: List[np.ndarray] = []
        for _ in range(self.segment_count):
            arrival_vinf.append(np.asarray(decision[idx: idx + 3], dtype=float))
            idx += 3

        return y_au, z_au, vx_kms, times_years, departure_vinf, arrival_vinf

    # ------------------------------------------------------------------
    # Propagation helpers
    # ------------------------------------------------------------------
    def _integrate_forward(
        self,
        initial_state: np.ndarray,
        start_time: float,
        match_time: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        sol = solve_ivp(
            _ballistic_ode_numpy,
            (start_time, match_time),
            initial_state,
            rtol=1e-9,
            atol=1e-9,
            dense_output=True,
            method="DOP853",
        )
        if not sol.success:
            raise RuntimeError(f"Forward propagation failed: {sol.message}")
        samples = np.linspace(start_time, match_time, 200)
        states = sol.sol(samples).T
        return states[-1], states

    def _integrate_backward(
        self,
        arrival_state: np.ndarray,
        arrival_time: float,
        match_time: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        sol = solve_ivp(
            _ballistic_ode_numpy,
            (arrival_time, match_time),
            arrival_state,
            rtol=1e-9,
            atol=1e-9,
            dense_output=True,
            method="DOP853",
        )
        if not sol.success:
            raise RuntimeError(f"Backward propagation failed: {sol.message}")
        samples = np.linspace(arrival_time, match_time, 200)
        states = sol.sol(samples).T
        return states[-1], states

    def evaluate(self, decision: Sequence[float]) -> PropagationResult:
        key = tuple(np.asarray(decision, dtype=float))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        y_au, z_au, vx_kms, times_years, departure_vinf, arrival_vinf = self._split_decision(decision)
        total_times_seconds = np.asarray(times_years, dtype=float) * YEAR
        start_times = np.concatenate(([0.0], np.cumsum(total_times_seconds[:-1])))

        segments: List[SegmentResult] = []
        target_positions: List[np.ndarray] = []
        target_velocities: List[np.ndarray] = []
        flyby_altitudes: List[float] = []
        flyby_valid: List[bool] = []

        for leg in range(self.segment_count):
            start_time = start_times[leg]
            total_seconds = total_times_seconds[leg]
            match_time = start_time + 0.5 * total_seconds
            arrival_time = start_time + total_seconds

            if leg == 0:
                initial_state = np.array(
                    [
                        -200.0 * AU_TO_KM,
                        y_au * AU_TO_KM,
                        z_au * AU_TO_KM,
                        vx_kms,
                        0.0,
                        0.0,
                    ],
                    dtype=float,
                )
            else:
                body = self.bodies[leg - 1]
                body_state = body.get_state(start_time, time_units="s")
                pos = np.asarray(body_state.r, dtype=float)
                vel = np.asarray(body_state.v, dtype=float)
                vinf_depart = departure_vinf[leg - 1]
                initial_state = np.concatenate([pos, vel + vinf_depart])

            forward_state, forward_samples = self._integrate_forward(initial_state, start_time, match_time)

            target_body = self.bodies[leg]
            body_state = target_body.get_state(arrival_time, time_units="s")
            target_pos = np.asarray(body_state.r, dtype=float)
            target_vel = np.asarray(body_state.v, dtype=float)
            arrival_state = np.concatenate([target_pos, target_vel + arrival_vinf[leg]])

            backward_state, backward_samples = self._integrate_backward(arrival_state, arrival_time, match_time)

            segments.append(
                SegmentResult(
                    forward_state=forward_state,
                    backward_state=backward_state,
                    forward_samples=forward_samples,
                    backward_samples=backward_samples,
                    start_time=start_time,
                    match_time=match_time,
                    arrival_time=arrival_time,
                )
            )
            target_positions.append(target_pos)
            target_velocities.append(target_vel)
            if leg < self.segment_count - 1:
                body = self.bodies[leg]
                hp, is_valid = patched_conic_flyby(
                    jnp.asarray(arrival_vinf[leg]),
                    jnp.asarray(departure_vinf[leg]),
                    float(body.mu),
                    float(body.radius),
                )
                flyby_altitudes.append(float(hp))
                flyby_valid.append(bool(is_valid))

        result = PropagationResult(
            segments=segments,
            vinf_arrivals=arrival_vinf,
            vinf_departures=departure_vinf,
            target_positions=target_positions,
            target_velocities=target_velocities,
            leg_times_seconds=total_times_seconds,
            flyby_altitudes_km=flyby_altitudes,
            flyby_valid=flyby_valid,
        )
        self._cache[key] = result
        return result


# ----------------------------------------------------------------------
# Objective and constraints
# ----------------------------------------------------------------------


def _objective_internal(x_internal: np.ndarray, problem: TrajectoryProblem) -> float:
    phys = problem.to_physical(x_internal)
    _, _, _, times_years, _, _ = problem._split_decision(phys)
    return float(np.sum(times_years))


def _objective_grad_internal(x_internal: np.ndarray, problem: TrajectoryProblem) -> np.ndarray:
    grad_phys = np.zeros_like(problem.scale_vector)
    grad_phys[3:3 + problem.segment_count] = 1.0
    return grad_phys * problem.scale_vector


def _constraint_fun_internal(x_internal: np.ndarray, problem: TrajectoryProblem) -> np.ndarray:
    """Inequality residuals (>=0) for midpoint continuity and v_inf matching."""
    phys = problem.to_physical(x_internal)
    evaluation = problem.evaluate(phys)
    residuals = []
    for seg in evaluation.segments:
        diff = seg.forward_state - seg.backward_state
        pos_norm = np.linalg.norm(diff[:3])
        vel_norm = np.linalg.norm(diff[3:])
        residuals.append(1.0 - pos_norm / POSITION_TOL_KM)
        residuals.append(1.0 - vel_norm / VELOCITY_TOL_KMS)
    for idx in range(problem.segment_count - 1):
        vinf_in = evaluation.vinf_arrivals[idx]
        vinf_out = evaluation.vinf_departures[idx]
        mag_diff = np.linalg.norm(vinf_in) - np.linalg.norm(vinf_out)
        smooth_abs = (mag_diff ** 2 + SMOOTH_ABS_EPS) ** 0.5
        residuals.append(1.0 - smooth_abs / VINF_TOL)
    return np.asarray(residuals, dtype=float)


def _constraint_jac_internal(x_internal: np.ndarray, problem: TrajectoryProblem, eps: float = 1e-6) -> np.ndarray:
    base = _constraint_fun_internal(x_internal, problem)
    m = base.size
    n = x_internal.size
    jac = np.zeros((m, n), dtype=float)
    for i in range(n):
        step = np.array(x_internal, copy=True)
        step[i] += eps
        diff = _constraint_fun_internal(step, problem) - base
        jac[:, i] = diff / eps
    return jac


def _flyby_constraints_internal(x_internal: np.ndarray, problem: TrajectoryProblem) -> np.ndarray:
    phys = problem.to_physical(x_internal)
    evaluation = problem.evaluate(phys)
    constraints: List[float] = []
    for idx, hp in enumerate(evaluation.flyby_altitudes_km):
        body = problem.bodies[idx]
        r_body = float(body.radius)
        constraints.append((hp - 0.1 * r_body) / (0.1 * r_body))
        constraints.append((100.0 * r_body - hp) / (100.0 * r_body))
    return np.asarray(constraints, dtype=float)


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def _build_scale_vector(segment_count: int) -> np.ndarray:
    scales = [100.0, 100.0, 100.0]
    scales.extend([100.0] * segment_count)
    scales.extend([50.0] * 3 * (segment_count - 1))
    scales.extend([50.0] * 3 * segment_count)
    return np.asarray(scales, dtype=float)


def _default_initial_guess(segment_count: int) -> np.ndarray:
    guess = [0.0, 0.0, 10.0]
    guess.extend([40.0] * segment_count)
    guess.extend([0.0] * 3 * (segment_count - 1))
    guess.extend([0.0] * 3 * segment_count)
    return np.asarray(guess, dtype=float)


def _expand_initial_guess(segment_count: int, guess: np.ndarray) -> np.ndarray:
    guess = np.asarray(guess, dtype=float)
    expected = 3 + segment_count + 3 * (segment_count - 1) + 3 * segment_count
    if guess.size == expected:
        return guess
    if guess.size == 4:
        y_au, z_au, vx_kms, total_years = guess
        leg_time = total_years / segment_count
        full = [y_au, z_au, vx_kms]
        full.extend([leg_time] * segment_count)
        full.extend([0.0] * 3 * (segment_count - 1))
        full.extend([0.0] * 3 * segment_count)
        return np.asarray(full, dtype=float)
    raise ValueError(f"Initial guess must have length 4 or {expected}.")


def _build_bounds(segment_count: int) -> Tuple[np.ndarray, np.ndarray]:
    lower = [
        -100.0,  # y
        -100.0,  # z
        0.0,     # vx
    ]
    lower.extend([1.0] * segment_count)  # leg times
    lower.extend([-50.0] * 3 * (segment_count - 1))  # departure vinf
    lower.extend([-50.0] * 3 * segment_count)        # arrival vinf

    upper = [
        100.0,
        100.0,
        100.0,
    ]
    upper.extend([100.0] * segment_count)
    upper.extend([50.0] * 3 * (segment_count - 1))
    upper.extend([50.0] * 3 * segment_count)

    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _run_slsqp(
    problem: TrajectoryProblem,
    initial_internal: np.ndarray,
    bounds_internal: Bounds,
    max_iter: int | None,
) -> OptimizeResult:
    iteration = {"count": 0}

    def callback(xk: np.ndarray) -> None:  # pragma: no cover - diagnostics
        iteration["count"] += 1
        phys = problem.to_physical(xk)
        evaluation = problem.evaluate(phys)
        slsqp_obj = _objective_internal(xk, problem)
        pos_err = 0.0
        vel_err = 0.0
        for seg in evaluation.segments:
            diff = seg.forward_state - seg.backward_state
            pos_err = max(pos_err, np.linalg.norm(diff[:3]))
            vel_err = max(vel_err, np.linalg.norm(diff[3:]))
        total_years = np.sum(evaluation.leg_times_seconds) / YEAR
        pos_ok = "OK" if pos_err <= POSITION_TOL_KM else "FAIL"
        vel_ok = "OK" if vel_err <= VELOCITY_TOL_KMS else "FAIL"
        vinf_err = 0.0
        if evaluation.vinf_arrivals[:-1]:
            for idx in range(problem.segment_count - 1):
                vinf_in = evaluation.vinf_arrivals[idx]
                vinf_out = evaluation.vinf_departures[idx]
                vinf_err = max(vinf_err, abs(np.linalg.norm(vinf_in) - np.linalg.norm(vinf_out)))
        vinf_ok = "OK" if vinf_err <= VINF_TOL else "FAIL"
        hp_info = "n/a"
        if evaluation.flyby_altitudes_km:
            min_idx = int(np.argmin(evaluation.flyby_altitudes_km))
            hp = evaluation.flyby_altitudes_km[min_idx]
            body = problem.bodies[min_idx]
            r_body = float(body.radius)
            hp_radii = hp / r_body
            hp_ok = "OK" if evaluation.flyby_valid[min_idx] else "FAIL"
            hp_info = (
                f"{body.name}: h_p={hp_radii:.3f} R (limits [0.1, 100]) -> {hp_ok}"
            )
        continuity_margin = _constraint_fun_internal(xk, problem)
        flyby_margin = _flyby_constraints_internal(xk, problem)
        if flyby_margin.size:
            all_margin = np.concatenate([continuity_margin, flyby_margin])
        else:
            all_margin = continuity_margin
        min_margin = float(np.min(all_margin)) if all_margin.size else float("nan")
        max_violation = max(0.0, -min_margin) if all_margin.size else float("nan")
        print(
            f"[iter {iteration['count']:03d}] v_inf={evaluation.final_vinf_mag():.6f} km/s "
            f"| objective={slsqp_obj:.6f} yr | total_TOF={total_years:.3f} yr "
            f"| pos_err={pos_err:.3e} km ({pos_ok}) | vel_err={vel_err:.3e} km/s ({vel_ok}) "
            f"| vinf_err={vinf_err:.3e} km/s ({vinf_ok}) "
            f"| min_h={hp_info} | slack={min_margin:.3e} | violation={max_violation:.3e}"
        )

    options = {"disp": True, "ftol": 1e-6, "maxiter": max_iter or 1000, "eps": 1e-8}

    result = minimize(
        lambda x: _objective_internal(x, problem),
        initial_internal,
        method="SLSQP",
        jac=lambda x: _objective_grad_internal(x, problem),
        bounds=bounds_internal,
        constraints=[
            {
                "type": "ineq",
                "fun": lambda x: _constraint_fun_internal(x, problem),
            },
            {
                "type": "ineq",
                "fun": lambda x: _flyby_constraints_internal(x, problem),
            },
        ],
        callback=callback,
        options=options,
    )
    return result


# ----------------------------------------------------------------------
# Plotting and reporting
# ----------------------------------------------------------------------


def plot_trajectory(decision: np.ndarray, evaluation: PropagationResult, body_ids: Sequence[int]) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Build full trajectory path
    path_segments = []
    cumulative = []
    for idx, seg in enumerate(evaluation.segments):
        forward_pos = seg.forward_samples[:, :3] / AU_TO_KM
        backward_pos = seg.backward_samples[::-1, :3] / AU_TO_KM
        if idx == 0:
            start_point = np.array([-200.0, decision[0], decision[1]], dtype=float)
            ax.scatter(*start_point, color="green", s=40, label="Entry point")
            path_segments.append(forward_pos)
        else:
            path_segments.append(forward_pos)
        path_segments.append(backward_pos)
        match_point = seg.forward_state[:3] / AU_TO_KM
        ax.scatter(*match_point, color="purple", s=30, label="Match point" if idx == 0 else "")
        cumulative.append(match_point)

    for arr in path_segments:
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color="tab:gray")

    # Plot bodies with unique labels
    seen_labels = set()
    for i, body_id in enumerate(body_ids):
        body = bodies_data[body_id]
        label = body.name
        pos = evaluation.target_positions[i] / AU_TO_KM
        if label in seen_labels:
            ax.scatter(*pos, color="blue", marker="^", s=60)
        else:
            ax.scatter(*pos, color="blue", marker="^", s=60, label=label)
            seen_labels.add(label)

    ax.scatter(0.0, 0.0, 0.0, color="yellow", s=60, label="Origin")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_zlabel("z [AU]")
    ax.set_title("Multi-segment starting sequence")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Solver entry point
# ----------------------------------------------------------------------


def solve_starting_sequence(
    body_ids: Sequence[int],
    initial_guess: Iterable[float] | None = None,
    solver: str = "slsqp",
    max_iter: int | None = 1000,
    show_plot: bool = True,
) -> Tuple[OptimizeResult, PropagationResult]:
    body_ids = tuple(int(b) for b in body_ids)
    segment_count = len(body_ids)
    scale_vector = _build_scale_vector(segment_count)
    problem = TrajectoryProblem(body_ids, scale_vector)

    if initial_guess is None:
        physical_guess = _default_initial_guess(segment_count)
    else:
        physical_guess = _expand_initial_guess(segment_count, np.asarray(initial_guess, dtype=float))

    lower, upper = _build_bounds(segment_count)
    initial_internal = problem.to_internal(physical_guess)
    bounds_internal = Bounds(problem.to_internal(lower), problem.to_internal(upper))

    if solver.lower() != "slsqp":
        raise ValueError("Only 'slsqp' solver is currently supported.")

    result = _run_slsqp(problem, initial_internal, bounds_internal, max_iter)

    result_internal = np.asarray(result.x, dtype=float)
    result_physical = problem.to_physical(result_internal)
    evaluation = problem.evaluate(result_physical)

    result.x_internal = result_internal  # type: ignore[attr-defined]
    result.x = result_physical
    continuity_margin = _constraint_fun_internal(result_internal, problem)
    flyby_margin = _flyby_constraints_internal(result_internal, problem)
    if flyby_margin.size:
        combined_margin = np.concatenate([continuity_margin, flyby_margin])
    else:
        combined_margin = continuity_margin
    result.constraint_value = float(np.min(combined_margin))  # type: ignore[attr-defined]
    result.fun = evaluation.final_vinf_mag()

    if show_plot:
        plot_trajectory(result_physical, evaluation, body_ids)

    return result, evaluation


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimise a ballistic starting sequence with multiple bodies.")
    parser.add_argument(
        "--body-ids",
        type=int,
        nargs="+",
        required=True,
        help="Ordered list of target body identifiers (visit sequence).",
    )
    parser.add_argument(
        "--initial-guess",
        type=float,
        nargs="*",
        default=None,
        help="Optional initial guess: 4 values [y_AU z_AU vx_kms total_T_years] or full vector.",
    )
    parser.add_argument(
        "--solver",
        choices=("slsqp",),
        default="slsqp",
        help="Optimizer to use (only 'slsqp' is available).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum optimizer iterations (default: 1000).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable 3D trajectory plot.")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    initial_guess = None
    if args.initial_guess:
        initial_guess = np.asarray(args.initial_guess, dtype=float)

    result, evaluation = solve_starting_sequence(
        body_ids=args.body_ids,
        initial_guess=initial_guess,
        solver=args.solver,
        max_iter=args.max_iter,
        show_plot=not args.no_plot,
    )

    residual_blocks = [seg.forward_state - seg.backward_state for seg in evaluation.segments]
    residual_stack = np.stack(residual_blocks, axis=0)
    pos_err = np.max(np.linalg.norm(residual_stack[:, :3], axis=1))
    vel_err = np.max(np.linalg.norm(residual_stack[:, 3:], axis=1))
    total_years = np.sum(evaluation.leg_times_seconds) / YEAR
    pos_ok = "OK" if pos_err <= POSITION_TOL_KM else "FAIL"
    vel_ok = "OK" if vel_err <= VELOCITY_TOL_KMS else "FAIL"
    vinf_err = 0.0
    if evaluation.vinf_arrivals[:-1]:
        for idx in range(len(args.body_ids) - 1):
            vinf_in = evaluation.vinf_arrivals[idx]
            vinf_out = evaluation.vinf_departures[idx]
            vinf_err = max(vinf_err, abs(np.linalg.norm(vinf_in) - np.linalg.norm(vinf_out)))
    vinf_ok = "OK" if vinf_err <= VINF_TOL else "FAIL"

    print(f"Success: {result.success}, status: {result.status}, message: {result.message}")
    print(f"Decision vector (physical units): {result.x}")
    print(f"Maximum position mismatch (km): {pos_err:.6e} ({pos_ok})")
    print(f"Maximum velocity mismatch (km/s): {vel_err:.6e} ({vel_ok})")
    if len(args.body_ids) > 1:
        print(f"Maximum v_inf magnitude mismatch (km/s): {vinf_err:.6e} ({vinf_ok})")
    print(f"Total time of flight (years): {total_years:.6f}")
    slack_status = "OK" if result.constraint_value >= 0.0 else "FAIL"
    print(f"Minimum constraint slack: {result.constraint_value:.6e} ({slack_status})")
    print(f"Arrival v-infinity magnitude (km/s): {evaluation.final_vinf_mag():.6f}")
    if evaluation.flyby_altitudes_km:
        print("Flyby altitudes:")
        for idx, hp in enumerate(evaluation.flyby_altitudes_km):
            body = bodies_data[args.body_ids[idx]]
            r_body = float(body.radius)
            status = "OK" if evaluation.flyby_valid[idx] else "FAIL"
            print(
                f"  {body.name} (id {args.body_ids[idx]}): h_p={hp / r_body:.3f} R "
                f"(limits [0.1, 100]) -> {status}"
            )


if __name__ == "__main__":
    main()
