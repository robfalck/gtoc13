from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, basinhopping, differential_evolution

from gtoc13.astrodynamics import DAY
from gtoc13.bodies import bodies_data
from gtoc13.path_finding import bs_lambert


Vec3 = Tuple[float, float, float]
_PERIOD_CACHE: Dict[int, Optional[float]] = {}


@dataclass
class GAConfig:
    start_body: int
    start_epoch: float
    start_vinf_vec: Optional[Vec3]
    max_legs: int
    require_full_length: bool
    tof_min_leg: float
    tof_max_leg: float
    mission_tof_max: Optional[float]
    dv_max: Optional[float]
    vinf_max: Optional[float]
    max_revs: int
    branch_limit: int
    body_choices: Tuple[Optional[int], ...]  # final entry is sentinel (None)


@dataclass
class EvaluationResult:
    feasible: bool
    score: float
    total_tof: float
    total_dv: float
    max_leg_dv: float
    path: Tuple[bs_lambert.Encounter, ...]
    violations: Tuple[str, ...]
    penalty: float


def _build_body_choices(allowed: Sequence[int], include_sentinel: bool = True) -> Tuple[Optional[int], ...]:
    """Return allowed body ids, optionally appending a sentinel None for early termination."""
    ordered = tuple(int(b) for b in allowed)
    if include_sentinel:
        return ordered + (None,)
    return ordered


def _body_period_days(body_id: int) -> Optional[float]:
    cached = _PERIOD_CACHE.get(body_id)
    if cached is not None:
        return cached
    a = bs_lambert.ACTIVE_SEMI_MAJOR_AXES.get(body_id)
    if a is None or a <= 0.0:
        _PERIOD_CACHE[body_id] = None
        return None
    period_seconds = 2.0 * math.pi * math.sqrt((a**3) / bs_lambert.MU_ALTAIRA)
    period_days = period_seconds / DAY
    _PERIOD_CACHE[body_id] = period_days
    return period_days


def _norm(vec: Iterable[float]) -> float:
    arr = np.asarray(tuple(vec), dtype=float)
    return float(np.linalg.norm(arr))


def _create_root_encounter(start_body: int, start_epoch: float, start_vinf: Optional[Vec3]) -> bs_lambert.Encounter:
    r0 = bs_lambert.ephemeris_position(start_body, start_epoch)
    if start_vinf is None:
        return bs_lambert.Encounter(body=start_body, t=start_epoch, r=r0)
    vinf_vec = tuple(float(x) for x in start_vinf)
    vinf_mag = _norm(vinf_vec)
    return bs_lambert.Encounter(
        body=start_body,
        t=start_epoch,
        r=r0,
        vinf_in=vinf_mag,
        vinf_in_vec=vinf_vec,
    )


def _decode_vector(x: np.ndarray, config: GAConfig) -> List[Tuple[int, float, int]]:
    """Convert optimizer vector into list of (body_id, tof_days, branch_index)."""
    genes: List[Tuple[int, float, int]] = []
    x = np.asarray(x, dtype=float)
    segments = config.max_legs
    choices = config.body_choices
    choice_count = len(choices)
    choice_limit = max(0, choice_count - 1)
    allow_sentinel = (not config.require_full_length) and choice_count > 0 and choices[-1] is None
    sentinel_idx = choice_limit if allow_sentinel else None
    last_body_id: Optional[int] = config.start_body
    for idx in range(segments):
        base = 2 * idx
        body_idx = int(round(np.clip(x[base], 0, choice_limit)))
        if allow_sentinel and sentinel_idx is not None and body_idx == sentinel_idx:
            break  # explicit termination
        body_id = choices[body_idx]

        if last_body_id is not None and body_id == last_body_id:
            # Rotate to the next available body to avoid consecutive repeats.
            if choice_count > 1:
                body_idx = (body_idx + 1) % choice_count
                if allow_sentinel and sentinel_idx is not None and body_idx == sentinel_idx:
                    body_idx = (body_idx + 1) % choice_count
                body_id = choices[body_idx]
                if body_id == last_body_id and choice_count > 2:
                    body_idx = (body_idx + 1) % choice_count
                    if allow_sentinel and sentinel_idx is not None and body_idx == sentinel_idx:
                        body_idx = (body_idx + 1) % choice_count
                    body_id = choices[body_idx]
            else:
                break  # no alternative; stop expansion here.

        tof_val = float(np.clip(x[base + 1], config.tof_min_leg, config.tof_max_leg))
        genes.append((body_id, tof_val, 0))
        last_body_id = body_id

    return genes


class TrajectoryEvaluator:
    def __init__(self, config: GAConfig):
        self.config = config
        self.cache: Dict[Tuple[float, ...], EvaluationResult] = {}
        self.root = _create_root_encounter(config.start_body, config.start_epoch, config.start_vinf_vec)

    def evaluate(self, x: np.ndarray) -> EvaluationResult:
        key = tuple(np.asarray(x, dtype=float).round(6))
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        try:
            result = self._evaluate_raw(x)
        except Exception as exc:  # pragma: no cover - defensive
            result = EvaluationResult(
                feasible=False,
                score=0.0,
                total_tof=float("inf"),
                total_dv=float("inf"),
                max_leg_dv=float("inf"),
                path=(self.root,),
                violations=(f"exception:{exc}",),
                penalty=1e7,
            )
        self.cache[key] = result
        return result

    def _evaluate_raw(self, x: np.ndarray) -> EvaluationResult:
        config = self.config
        genes = _decode_vector(x, config)

        path: Tuple[bs_lambert.Encounter, ...] = (self.root,)
        total_dv = 0.0
        max_leg_dv = 0.0
        mission_start = self.root.t
        violations: List[str] = []
        penalty = 0.0

        if not genes:
            return EvaluationResult(
                feasible=(penalty == 0.0),
                score=0.0,
                total_tof=0.0,
                total_dv=0.0,
                max_leg_dv=0.0,
                path=path,
                violations=tuple(violations),
                penalty=penalty,
            )

        for leg_index, (body_id, tof_days, branch_idx) in enumerate(genes, start=1):
            parent = path[-1]
            arrival_time = parent.t + tof_days

            if arrival_time <= parent.t + 1e-6:
                violations.append("nonpositive_tof")
                penalty += 1e7
                break

            total_elapsed = arrival_time - mission_start
            if config.mission_tof_max is not None and total_elapsed > config.mission_tof_max:
                if "mission_tof_limit" not in violations:
                    violations.append("mission_tof_limit")
                penalty += 1e3 * (total_elapsed - config.mission_tof_max + 1.0)

            try:
                tmin, tmax = bs_lambert.hohmann_bounds_for_bodies(parent.body, body_id)
            except Exception:
                tmin, tmax = None, None
            if tmin is not None and tmax is not None:
                window_min = 0.5 * tmin
                window_max = 1.5 * tmax
                if tof_days < window_min or tof_days > window_max:
                    violations.append("tof_out_of_window")
                    penalty += 1e3 * (abs(tof_days - np.clip(tof_days, window_min, window_max)) + 1.0)

            sol_entries = bs_lambert._enumerate_lambert_solutions(
                parent.body,
                body_id,
                parent.t * DAY,
                arrival_time * DAY,
                max_revs=config.max_revs,
            )
            if not sol_entries:
                violations.append("no_lambert_solution")
                penalty += 5e4
                break
            sol_entries.sort(key=lambda s: float(np.linalg.norm(np.asarray(s["v1"])) + np.linalg.norm(np.asarray(s["v2"]))))
            sol = sol_entries[0]
            _, v_body_depart = bs_lambert.body_state(parent.body, parent.t)
            _, v_body_arrive = bs_lambert.body_state(body_id, arrival_time)

            vinf_out_vec = np.asarray(sol["v1"], dtype=float) - v_body_depart
            vinf_in_vec = np.asarray(sol["v2"], dtype=float) - v_body_arrive

            vinf_out = float(np.linalg.norm(vinf_out_vec))
            vinf_in = float(np.linalg.norm(vinf_in_vec))

            vinf_min = 0.1
            min_violation = min(vinf_out, vinf_in) - vinf_min
            if min_violation < 0.0:
                violations.append("vinf_min")
                penalty += 1e3 * (abs(min_violation) + 0.01)

            if config.vinf_max is not None:
                excess = max(vinf_out, vinf_in) - config.vinf_max
                if excess > 0.0:
                    violations.append("vinf_limit")
                    penalty += 1e3 * excess**2

            flyby_valid, flyby_altitude, dv_mag, dv_vec = bs_lambert.evaluate_flyby(
                parent.body,
                parent.vinf_in_vec,
                tuple(float(x) for x in vinf_out_vec),
            )

            if dv_mag is None:
                dv_mag = 0.0

            if config.dv_max is not None:
                dv_excess = dv_mag - config.dv_max
                if dv_excess > 0.0:
                    violations.append("dv_limit")
                    penalty += 2e3 * dv_excess**2

            parent_resolved = bs_lambert.replace(
                parent,
                vinf_out=vinf_out,
                vinf_out_vec=tuple(float(x) for x in vinf_out_vec),
                flyby_valid=flyby_valid,
                flyby_altitude=flyby_altitude,
                dv_periapsis=dv_mag,
                dv_periapsis_vec=dv_vec,
            )
            prefix = path[:-1] + (parent_resolved,)

            child = bs_lambert.Encounter(
                body=body_id,
                t=arrival_time,
                r=tuple(float(x) for x in sol["r2"]),
                vinf_in=vinf_in,
                vinf_in_vec=tuple(float(x) for x in vinf_in_vec),
                J_total=parent_resolved.J_total,
            )

            prev_same = next((enc for enc in reversed(prefix) if enc.body == body_id), None)
            if prev_same is not None:
                dt = arrival_time - prev_same.t
                period_days = _body_period_days(body_id)
                if period_days is not None and period_days > 0.0:
                    min_sep = 0.4 * period_days
                    if dt <= min_sep:
                        violations.append("repeat_too_soon")
                        shortfall = max(0.0, min_sep - dt)
                        penalty += 1e3 * (shortfall + 1.0)

            candidate_path = prefix + (child,)
            total_score = bs_lambert.mission_score(candidate_path)
            if not math.isfinite(total_score):
                violations.append("score_nan")
                penalty += 1e5
                break

            child = bs_lambert.replace(child, J_total=total_score)
            path = candidate_path[:-1] + (child,)

            total_dv += dv_mag
            max_leg_dv = max(max_leg_dv, dv_mag)

        total_tof = path[-1].t - mission_start
        if config.mission_tof_max is not None and total_tof > config.mission_tof_max:
            if "mission_tof_limit" not in violations:
                violations.append("mission_tof_limit")
            excess = total_tof - config.mission_tof_max
            penalty += 1e3 * excess**2 + 1e5

        actual_legs = max(0, len(path) - 1)
        if config.require_full_length and actual_legs < config.max_legs:
            deficit = config.max_legs - actual_legs
            if "too_short" not in violations:
                violations.append("too_short")
            penalty += 1e5 * (deficit + 1)

        feasible = penalty == 0.0
        score = path[-1].J_total if path else 0.0
        return EvaluationResult(
            feasible=feasible,
            score=score,
            total_tof=total_tof,
            total_dv=total_dv,
            max_leg_dv=max_leg_dv,
            path=path,
            violations=tuple(violations),
            penalty=penalty,
        )


@dataclass
class DEProgressPrinter:
    evaluator: TrajectoryEvaluator
    verbose: bool = True
    start_time: float = time.monotonic()
    iteration: int = 0

    def __call__(self, xk: np.ndarray, convergence: float) -> bool:
        if not self.verbose:
            return False
        self.iteration += 1
        res_eval = self.evaluator.evaluate(xk)
        score_display = f"{res_eval.score:.4f}" if res_eval.feasible else "—"
        elapsed = time.monotonic() - self.start_time
        sequence = "->".join(str(enc.body) for enc in res_eval.path)
        print(
            f"[gen {self.iteration:3d}] feasible={res_eval.feasible} best_score={score_display} "
            f"total_tof={res_eval.total_tof:.2f}d dv_total={res_eval.total_dv:.3f}km/s penalty={res_eval.penalty:.2f} "
            f"seq={sequence} conv={convergence:.3e} elapsed={elapsed:.1f}s",
            flush=True,
        )
        return False


def objective(vec: np.ndarray, evaluator: TrajectoryEvaluator) -> float:
    res_eval = evaluator.evaluate(vec)
    return -res_eval.score + res_eval.penalty


def _format_path(path: Tuple[bs_lambert.Encounter, ...]) -> str:
    lines = []
    for idx, enc in enumerate(path):
        vinf_in = "—" if enc.vinf_in is None else f"{enc.vinf_in:.3f}"
        vinf_out = "—" if enc.vinf_out is None else f"{enc.vinf_out:.3f}"
        dv = "—" if enc.dv_periapsis is None else f"{enc.dv_periapsis:.3f}"
        flyby = "?" if enc.flyby_valid is None else ("✓" if enc.flyby_valid else "×")
        score = enc.J_total if idx == len(path) - 1 else enc.J_total
        lines.append(
            f"    [{idx}] body={enc.body:4d} t={enc.t:10.2f} d  score={score:10.4f}  "
            f"v∞in={vinf_in:>7}  v∞out={vinf_out:>7}  flyby={flyby}  dvp={dv}"
        )
    return "\n".join(lines)


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Differential evolution search over Lambert sequences.")
    parser.add_argument("--start-body", type=int, default=10, help="Starting body ID (default PlanetX=10).")
    parser.add_argument("--start-epoch", type=float, default=0.0, help="Starting epoch in days.")
    parser.add_argument(
        "--start-vinf",
        nargs=3,
        type=float,
        metavar=("VX", "VY", "VZ"),
        default=None,
        help="Optional inbound v-infinity vector (km/s) at the starting body.",
    )
    parser.add_argument("--max-legs", type=int, default=5, help="Maximum number of legs/encounters.")
    parser.add_argument(
        "--require-full-length",
        action="store_true",
        help="Force chromosomes to use all max-legs entries (no early termination).",
    )
    parser.add_argument("--tof-min-leg", type=float, default=10.0, help="Minimum leg time-of-flight (days).")
    parser.add_argument("--tof-max-leg", type=float, default=400.0, help="Maximum leg time-of-flight (days).")
    parser.add_argument("--tof-max", type=float, default=bs_lambert.TOF_MISSION_MAX_DAYS, help="Mission TOF cap (days).")
    parser.add_argument("--dv-max", type=float, default=bs_lambert.DV_PERIAPSIS_MAX, help="Periapsis Δv limit (km/s).")
    parser.add_argument("--vinf-max", type=float, default=bs_lambert.VINF_MAX, help="Maximum |v∞| (km/s).")
    parser.add_argument("--max-revs", type=int, default=2, help="Maximum revolutions for Lambert solutions.")
    parser.add_argument(
        "--branch-limit",
        type=int,
        default=6,
        help="Maximum Lambert branch index considered per leg (higher values allow more multi-rev options).",
    )
    parser.add_argument(
        "--body-types",
        default="planets,asteroids,comets",
        help="Comma-separated list of body categories to include (planets, asteroids, comets).",
    )
    parser.add_argument(
        "--optimizer",
        choices=("de", "basinhopping"),
        default="de",
        help="Global optimizer to use: differential evolution (de) or basin hopping.",
    )
    parser.add_argument("--strategy", default="best1bin", help="Differential evolution strategy.")
    parser.add_argument("--popsize", type=int, default=15, help="Population size multiplier for DE.")
    parser.add_argument("--maxiter", type=int, default=80, help="Maximum DE generations.")
    parser.add_argument("--mutation", type=float, default=0.7, help="Mutation factor for DE.")
    parser.add_argument("--recombination", type=float, default=0.9, help="Recombination (crossover) rate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (-1 for all cores).")
    parser.add_argument("--tol", type=float, default=0.01, help="Relative convergence tolerance.")
    parser.add_argument("--polish", action="store_true", help="Enable polishing with trust-constr at the end.")
    parser.add_argument("--bh-iter", type=int, default=150, help="Basin hopping iterations (niter).")
    parser.add_argument("--bh-stepsize", type=float, default=1.0, help="Basin hopping step size.")
    parser.add_argument("--bh-temperature", type=float, default=1.0, help="Basin hopping temperature.")
    parser.add_argument(
        "--bh-local-maxiter",
        type=int,
        default=100,
        help="Maximum iterations for the local minimizer within basin hopping.",
    )
    parser.add_argument(
        "--bh-pareto-alpha",
        type=float,
        default=1.5,
        help="Shape parameter α for the Pareto-distributed hop lengths (larger makes hops shorter).",
    )
    args = parser.parse_args()

    type_aliases = {
        "planet": "planet",
        "planets": "planet",
        "asteroid": "asteroid",
        "asteroids": "asteroid",
        "comet": "comet",
        "comets": "comet",
        "small": "small",
        "all": "all",
    }
    requested_types: set[str] = set()
    for token in (args.body_types or "").split(","):
        token = token.strip().lower()
        if not token:
            continue
        alias = type_aliases.get(token)
        if alias is None:
            raise SystemExit(f"Unknown body type '{token}'.")
        if alias == "all":
            requested_types.update({"planet", "asteroid", "comet", "small"})
        else:
            requested_types.add(alias)
    if not requested_types:
        requested_types = {"planet", "asteroid", "comet"}

    bs_lambert._activate_body_subset(requested_types)

    if args.start_body not in bodies_data:
        available = ", ".join(str(k) for k in sorted(bodies_data.keys()))
        raise SystemExit(f"Unknown start body {args.start_body}. IDs: {available}")

    allowed_ids = bs_lambert.ACTIVE_BODY_IDS
    if args.start_body not in allowed_ids:
        raise SystemExit("Start body must be within the selected body subset.")

    if args.max_legs <= 0:
        raise SystemExit("--max-legs must be positive.")

    body_choices = _build_body_choices(allowed_ids, include_sentinel=not args.require_full_length)
    config = GAConfig(
        start_body=args.start_body,
        start_epoch=args.start_epoch,
        start_vinf_vec=tuple(args.start_vinf) if args.start_vinf is not None else None,
        max_legs=args.max_legs,
        require_full_length=bool(args.require_full_length),
        tof_min_leg=max(1e-3, args.tof_min_leg),
        tof_max_leg=max(args.tof_min_leg, args.tof_max_leg),
        mission_tof_max=None if args.tof_max is None or args.tof_max < 0 else args.tof_max,
        dv_max=None if args.dv_max is None or args.dv_max < 0 else args.dv_max,
        vinf_max=None if args.vinf_max is None or args.vinf_max < 0 else args.vinf_max,
        max_revs=max(0, args.max_revs),
        branch_limit=max(1, args.branch_limit),
        body_choices=body_choices,
    )

    evaluator = TrajectoryEvaluator(config)

    dim = 2 * config.max_legs
    lower: List[float] = []
    upper: List[float] = []
    integrality: List[bool] = []
    choice_limit = len(config.body_choices) - 1
    for _ in range(config.max_legs):
        lower.extend([0.0, config.tof_min_leg])
        upper.extend([float(choice_limit), config.tof_max_leg])
        integrality.extend([True, False])

    bounds = Bounds(lower, upper)
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)

    if args.optimizer == "de":
        print(
            "Differential evolution start:",
            f"start_body={config.start_body}",
            f"max_legs={config.max_legs}",
            f"population={args.popsize}",
            f"maxiter={args.maxiter}",
            f"mission_tof_max={config.mission_tof_max}",
            f"dv_max={config.dv_max}",
            f"vinf_max={config.vinf_max}",
        )

        progress = DEProgressPrinter(evaluator=evaluator)

        opt_result = differential_evolution(
            partial(objective, evaluator=evaluator),
            bounds=bounds,
            strategy=args.strategy,
            maxiter=args.maxiter,
            popsize=args.popsize,
            mutation=args.mutation,
            recombination=args.recombination,
            seed=args.seed,
            polish=args.polish,
            workers=args.workers,
            integrality=np.array(integrality, dtype=bool),
            tol=args.tol,
            updating="deferred" if args.workers and args.workers != 1 else "immediate",
            callback=progress,
        )
        best_vec = opt_result.x
        success = opt_result.success
        message = opt_result.message
        nfev = opt_result.nfev
    else:
        print(
            "Basin hopping start:",
            f"start_body={config.start_body}",
            f"max_legs={config.max_legs}",
            f"niter={args.bh_iter}",
            f"stepsize={args.bh_stepsize}",
            f"mission_tof_max={config.mission_tof_max}",
            f"dv_max={config.dv_max}",
            f"vinf_max={config.vinf_max}",
        )

        rng = np.random.default_rng(args.seed)
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in zip(lower_arr, upper_arr)])

        def bounded_objective(vec):
            vec = np.asarray(vec, dtype=float)
            vec = np.clip(vec, lower_arr, upper_arr)
            return objective(vec, evaluator)

        class ParetoStep:
            __slots__ = ("rng", "stepsize", "lower", "upper", "alpha")

            def __init__(self, rng, stepsize, lower, upper, alpha):
                self.rng = rng
                self.stepsize = stepsize
                self.lower = lower
                self.upper = upper
                self.alpha = alpha

            def __call__(self, x):
                x = np.asarray(x, dtype=float)
                direction = self.rng.normal(size=x.shape)
                norm = np.linalg.norm(direction)
                if norm < 1e-12:
                    direction = np.ones_like(direction)
                    norm = np.linalg.norm(direction)
                direction /= norm
                radius = self.rng.pareto(self.alpha) + 1.0
                step = direction * (self.stepsize * radius)
                proposal = x + step
                return np.clip(proposal, self.lower, self.upper)

        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": list(zip(lower_arr, upper_arr)),
            "options": {"maxiter": args.bh_local_maxiter},
        }

        take_step = ParetoStep(rng=rng, stepsize=args.bh_stepsize, lower=lower_arr, upper=upper_arr, alpha=args.bh_pareto_alpha)

        opt_result = basinhopping(
            bounded_objective,
            x0,
            niter=args.bh_iter,
            stepsize=args.bh_stepsize,
            T=args.bh_temperature,
            minimizer_kwargs=minimizer_kwargs,
            seed=args.seed,
            disp=True,
            take_step=take_step,
        )
        best_vec = np.clip(opt_result.x, lower_arr, upper_arr)
        lowest = opt_result.lowest_optimization_result
        success = bool(getattr(lowest, "success", True))
        message = opt_result.message
        nfev = getattr(lowest, "nfev", opt_result.nfev)

    best = evaluator.evaluate(best_vec)
    print("\nOptimization complete.")
    print(f"Optimizer: {args.optimizer}")
    print(f"Success: {success}  message='{message}'")
    print(f"Best objective: {-best.score + best.penalty:.4f}")
    print(f"Function evaluations: {nfev}")
    print(f"Feasible: {best.feasible}  Score: {best.score:.6f}  Penalty: {best.penalty:.4f}")
    print(f"Total TOF: {best.total_tof:.2f} days  Total Δv: {best.total_dv:.3f} km/s  Max leg Δv: {best.max_leg_dv:.3f} km/s")
    if not best.feasible:
        print(f"Violations: {', '.join(best.violations) or 'none'}")

    print("\nBest sequence:")
    print(_format_path(best.path))


if __name__ == "__main__":
    run_cli()
