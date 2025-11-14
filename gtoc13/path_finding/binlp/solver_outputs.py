from pyomo.opt import TerminationCondition, SolverFactory, SolverResults
import pyomo.environ as pyo
from gtoc13.path_finding.binlp.b_utils import SolverParams, SequenceTarget, timer
from pathlib import Path
from typing import Any
from numpy import round
from gtoc13 import YEAR, SPTU, KMPDU
from pprint import pprint
from gtoc13.path_finding.binlp.build_model import nogood_cuts_constrs
from numpy import ndarray


@timer
def run_solver(
    model: pyo.ConcreteModel, solver_params: SolverParams, iter: int | None = None
) -> tuple[SolverResults, Any]:
    output_path = Path.cwd() / "outputs"
    if not iter:
        iter = ""
    if solver_params.write_nl:
        print("...writing .nl file for debugging...")
        Path.mkdir(output_path, exist_ok=True)
        model.write(output_path / f"sequence_{iter}.nl", format="nl")
    solver = SolverFactory(solver_params.solver_name, solver_io="nl")
    results = solver.solve(
        model,
        tee=solver_params.toconsole,
        options={"limits/gap": solver_params.soln_gap}
        if solver_params.solver_name == "scip"
        else None,
        logfile=output_path / f"solverlog_{iter}.txt" if solver_params.write_log else None,
    )
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
            place=each[0],
            name=each[1][0],
            body_id=each[1][1][0],
            ts_idx=each[1][1][1],
            year=each[2],
        )
        for each in sequence
    ]
    return sequence, short_seq


def process_arcs(model: pyo.ConcreteModel, short_sequence: list[tuple[int, int]]):
    if model.find_component("Lp_kimj"):
        print(
            "Number of lambert arcs: ",
            int(round(pyo.value(pyo.summation(model.Lp_kimj))))
            + int(round(pyo.value(pyo.summation(model.Lr_kimj)))),
            "\n",
        )
        print("Lambert arcs (k, i) to (m, j):")
        lambert_arcs = []
        for k, i, m, j in model.KIMJ:
            if pyo.value(model.Lp_kimj[k, i, m, j]) > 0.5:
                energy = (
                    "prograde",
                    round(model.p_se_kimj[k, i, m, j] * float(KMPDU / SPTU) ** 2, 3),
                )
            elif pyo.value(model.Lr_kimj[k, i, m, j]) > 0.5:
                energy = (
                    "retrograde",
                    round(model.r_se_kimj[k, i, m, j] * float(KMPDU / SPTU) ** 2, 3),
                )
            else:
                continue
            lambert_arcs.append(
                (
                    short_sequence.index((k, i)) + 1,
                    f"({k}, {i}) to ({m}, {j})",
                    f"spec_en: {energy}",
                )
            )

        # lambert_arcs = [
        #     [
        #         short_sequence.index((k, i)) + 1,
        #         f"({k}, {i}) to ({m}, {j})",
        #         f"dv_tot: {round((model.dvout_kimj[k, i, m, j] + model.dvin_kimj[k, i, m, j]) * KMPDU / SPTU, 3)}",
        #     ]
        #     for (k, i, m, j), v in model.L_kimj.items()
        #     if pyo.value(v) > 0.5
        # ]
        lambert_arcs = sorted(lambert_arcs)
        for arc in lambert_arcs:
            print(arc)
        print("\n")
    else:
        lambert_arcs = None
    return lambert_arcs


def process_flybys(
    model: pyo.ConcreteModel, flyby_history: dict[int : list[ndarray]] | None = None
) -> dict[int : list[ndarray]]:
    print("Number of repeated flybys:", int(round(pyo.value(pyo.summation(model.y_kij)))), "\n")
    print("First flyby keys (k, i-th):")
    first_flybys = [ki for ki, v in model.z_ki.items() if pyo.value(v) > 0.5]
    pprint(sorted(first_flybys, key=lambda x: model.tu_ki[x]))
    if flyby_history:
        bodies = flyby_history.keys()
        for k, i in first_flybys:
            if k in bodies:
                flyby_history[k].append(model.rdu_ki[k, i])
            else:
                flyby_history.update({k: [model.rdu_ki[k, i]]})
    else:
        flyby_history = {k: [model.rdu_ki[k, i]] for (k, i) in first_flybys}
    # flyby_history = {}
    if pyo.value(pyo.summation(model.y_kij) > 0):
        # all_dupe
        print("\n")
        print("Repeat flyby keys (k, i-th, j-th prev):")
        i_prev = 0
        for (k, i, j), v in model.y_kij.items():
            if pyo.value(v) > 0.5:
                if i_prev != i:
                    flyby_history[k].append(model.rdu_ki[k, i])
                    i_prev = i
                print((k, i, j))
    print("\n")
    return flyby_history


@timer
def generate_iterative_solutions(
    model: pyo.ConcreteModel, solver_params: SolverParams
) -> list[list[SequenceTarget]]:
    print(f">>>>> RUN SOLVER FOR {solver_params.solv_iter} ITERATIONS(S) >>>>>")
    for iter in range(solver_params.solv_iter):
        print(f"...solving iteration {iter + 1}...")
        results, solver = run_solver(model, solver_params, iter + 1)
        print(f"...iteration {iter + 1} solved...")
        if results.solver.termination_condition == TerminationCondition.infeasible:
            print("Infeasible, sorry :(\n")
            sequence = None
        else:
            sequence, short_seq = process_sequence(model)
            __ = process_arcs(model, short_seq)
            __ = process_flybys(model)
            print(f"\n...iteration {iter + 1} processed...\n")
        soln_seqs = []
        if not sequence:
            break
        soln_seqs.append(sequence)
        if iter < solver_params.solv_iter - 2:
            ### create no-good cuts for multiple solutions of the same problem ###
            print(f"...start no-good cuts for iteration {iter + 2}...")
            nogood_cuts_constrs(model)
            print("<<<<< FINISHED RUNNING ALL ITERATIONS <<<<<")
    return soln_seqs


@timer
def generate_segment(
    model: pyo.ConcreteModel,
    solver_params: SolverParams,
    flyby_history: dict[int : list[ndarray]] | None = None,
) -> tuple[list[SequenceTarget], dict[int : list[ndarray]]]:
    results, solver = run_solver(model, solver_params)
    if results.solver.termination_condition == TerminationCondition.infeasible:
        print("Infeasible, sorry :(\n")
        sequence = None
        flyby_history = None
    else:
        sequence, short_seq = process_sequence(model)
        __ = process_arcs(model, short_seq)
        flyby_history = process_flybys(model, flyby_history)

    return sequence, flyby_history, model
