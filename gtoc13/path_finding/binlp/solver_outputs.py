from pyomo.opt import TerminationCondition, SolverFactory, SolverResults
import pyomo.environ as pyo
from b_utils import SolverParams, SequenceTarget, timer
from pathlib import Path
from typing import Any
from numpy import round
from gtoc13 import YEAR, SPTU
from pprint import pprint
from build_model import nogood_cuts_iteration_constrs


@timer
def run_solver(
    model: pyo.ConcreteModel, solver_params: SolverParams, iter: int
) -> tuple[SolverResults, Any]:
    if solver_params.write_nl:
        print("...writing .nl file for debugging...")
        output_path = Path.cwd() / "outputs"
        Path.mkdir(output_path, exist_ok=True)
        model.write(output_path / f"sequence_{iter}.nl", format="nl")
    print(f"...solving iteration {iter}...")
    solver = SolverFactory(solver_params.solver_name, solver_io="nl")
    results = solver.solve(
        model,
        tee=solver_params.toconsole,
        options={"limits/gap": solver_params.soln_gap}
        if solver_params.solver_name == "scip"
        else None,
        logfile=output_path / f"solverlog_{iter}.txt" if solver_params.write_log else None,
    )
    print(f"...iteration {iter} solved...")
    return results, solver


def process_sequence(
    model: pyo.ConcreteModel,
) -> tuple[list[SequenceTarget], list[tuple[int, int]]]:
    sequence = [
        (h, (model.name_k[k], (k, i)), round((model.tu_ki[k, i] * SPTU / YEAR).tolist(), 3))
        for (k, i, h), v in model.x_kih.items()
        if pyo.value(v) > 0.5
    ]
    sequence = sorted(sequence, key=lambda x: (x[2], x[0]))
    print("Sequence (h-th position, k-th body, i-th timestep in years):")
    pprint(sequence)
    print("\n")
    short_seq = [each[1][1] for each in sequence]
    sequence = [
        SequenceTarget(
            place=each[0], name=each[1][0], ID=each[1][1], t_idx=each[1][2], year=each[2]
        )
        for each in sequence
    ]
    return sequence, short_seq


def process_arcs(model: pyo.ConcreteModel, short_seq: list[tuple[int, int]]):
    if model.find_component("L_kimj"):
        print("Number of lambert arcs: ", int(round(pyo.value(pyo.summation(model.L_kimj)))), "\n")
        print("Lambert arcs (k, i) to (m, j):")
        lambert_arcs = [
            [
                short_seq.index((k, i)) + 1,
                f"({k}, {i}) to ({m}, {j})",
                f"dv_tot: {round(model.dv_kimj[k, i, m, j], 3)}",
            ]
            for (k, i, m, j), v in model.L_kimj.items()
            if pyo.value(v) > 0.5
        ]
        lambert_arcs = sorted(lambert_arcs)
        for arc in lambert_arcs:
            print(arc)
        print("\n")
    else:
        lambert_arcs = None
    return lambert_arcs


def process_flybys(model: pyo.ConcreteModel):
    pass


def print_solution(
    model: pyo.ConcreteModel, results: SolverResults, iter: int
) -> list[tuple[int, tuple[str, tuple[int, int]], float]] | None:
    if results.solver.termination_condition == TerminationCondition.infeasible:
        print("Infeasible, sorry :(\n")
        sequence = None
    else:
        sequence, short_seq = process_sequence(model=model)
        arcs = process_arcs(model=model)

        ######
        print("Number of repeated flybys:", int(round(pyo.value(pyo.summation(model.y_kij)))), "\n")
        print("First flyby keys (k, i-th):")
        first_flybys = [ki for ki, v in model.z_ki.items() if pyo.value(v) > 0.5]
        pprint(sorted(first_flybys, key=lambda x: model.tu_ki[x]))
        if pyo.value(pyo.summation(model.y_kij) > 0):
            print("\n")
            print("Repeat flyby keys (k, i-th, j-th prev):")
            for kij, v in model.y_kij.items():
                if pyo.value(v) > 0.5:
                    print(kij)

        print(f"\n...iteration {iter} complete...\n")
    return sequence


@timer
def generate_iterative_solutions(
    model: pyo.ConcreteModel, solver_params: SolverParams
) -> list[list[SequenceTarget]]:
    print(f">>>>> RUN SOLVER FOR {solver_params.solv_iter} ITERATIONS(S) >>>>>")
    for iter in range(solver_params.solv_iter):
        results, solver = run_solver(model=model, solver_params=solver_params, iter=iter + 1)
        soln_seq = print_solution(model=model, results=results, iter=iter + 1)
        soln_seqs = []
        if not soln_seq:
            break
        soln_seqs.append(soln_seq)
        if iter < solver_params.solv_iter - 2:
            ### create no-good cuts for multiple solutions of the same problem ###
            print(f"...start no-good cuts for iteration {iter + 2}...")
            nogood_cuts_iteration_constrs(seq_model=model)
            print("<<<<< FINISHED RUNNING ALL ITERATIONS <<<<<")
    return soln_seqs


def generate_segment_t():
    pass
