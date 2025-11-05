"""
Binary Integer Non-Linear Programming formulation compute the most "valuable" sequences
for scientific flybys. Assisting flybys not counted; they should be calculated during
sequence trajectory evaluation.

Assumes a starting year and body; solve_first_arc.py should determine which body and when.

Flyby object

NOTE: users need to update their environment according to the pyproject.toml and install 'scip' via conda.

TODO:
    - compute or intake hyperbolic excess velocity per planet <- that might just be the dv_limit
    - clean up script to yield sequence(s) for validation
    - config file for parameters

Variables
-------
x_tkh : binary variable for

"""

import pyomo.environ as pyo
import pyomo.contrib.iis as iis
import pyomo.util.report_scaling as scaling
import pyomo.util.model_size as m_size
from pyomo.opt import SolverFactory, TerminationCondition
import os
import numpy as np

# from pathlib import Path

from gtoc13 import DAY, KMPDU, SPTU, YEAR, MU_ALTAIRA, bodies_data

from pykep import lambert_problem
from tqdm import tqdm
import time
from itertools import product
from pprint import pprint
from math import factorial

############### CONFIG ###############
Yo = 0
Yf = 6
perYear = 2
dv_limit = 3000
h_tot = 3  # sequence length
gt_bodies = 3
gt_smalls = 13
Nk_lim = 3  # 13
sol_iter = 1
solver_log = True
solver = "scip"  # AMPL-format solvers
############# END CONFIG #############


class Timer:
    def __enter__(self):
        self._enter_time = time.time()

    def __exit__(self, *exc_args):
        self._exit_time = time.time()
        print(f"{self._exit_time - self._enter_time:.2f} seconds elapsed\n")


def lin_dots_penalty(r_i, r_j):
    dots = np.clip(np.dot(r_i, r_j) / (np.linalg.norm(r_i) * np.linalg.norm(r_j)), -1.0, 1.0)
    if dots < 0.875:
        return 0.0
    else:
        return np.clip(  # polynomial fit of penalty
            3e-07 * dots**6
            - 2e-05 * dots**5
            + 0.0006 * dots**4
            - 0.0072 * dots**3
            + 0.0426 * dots**2
            - 0.108 * dots
            + 1.0822,
            0,
            1,
        )


##############################################################################################
print(">>>>> CREATE DISCRETIZED DATASET <<<<<\n")
with Timer():
    #############
    To = Yo * YEAR
    Tf = Yf * YEAR  # years in seconds
    num = int(np.ceil((Tf - To) / (YEAR / perYear)))
    dv_limit *= SPTU.tolist() / KMPDU  # km/s

    ## Generate position tables for just the bodies
    print("...discretizing body data...")
    with Timer():
        discrete_data = dict()
        for b_idx, body in tqdm(bodies_data.items()):
            if body.is_planet() or body.name == "Yandi":
                timestep = np.linspace(To, Tf, num) / SPTU
                discrete_data[b_idx] = dict(
                    r_du=np.array([body.get_state(timestep[idx]).r / KMPDU for idx in range(num)]),
                    t_tu=timestep,
                )

    k_body = list(discrete_data.keys())
    print("...calculating lambert delta-vs...")
    with Timer():
        dv_1 = {}
        dv_2 = {}
        for kimj in tqdm(
            [
                (k, i, m, j)
                for (k, m) in list(product(k_body, repeat=2))
                for i, tu_i in enumerate(discrete_data[k]["t_tu"])
                for j, tu_j in enumerate(discrete_data[m]["t_tu"])
            ]
        ):
            k, i, m, j = kimj
            tof = (discrete_data[m]["t_tu"][j] - discrete_data[k]["t_tu"][i]).tolist()
            # if tof > 0:
            #     ki_to_mj = lambert_problem(
            #         r1=discrete_data[k]["r_du"][i].tolist(),
            #         r2=discrete_data[m]["r_du"][j].tolist(),
            #         tof=(discrete_data[m]["t_tu"][j] - discrete_data[k]["t_tu"][i]).tolist(),
            #     )

            #     dv_1[(k, i + 1, m, j + 1)] = np.array(ki_to_mj.get_v1()[0])
            #     dv_2[(k, i + 1, m, j + 1)] = np.array(ki_to_mj.get_v2()[0])
            # else:
            dv_1[(k, i + 1, m, j + 1)], dv_2[(k, i + 1, m, j + 1)] = np.random.rand(2) * (
                dv_limit + 50.0
            )
    # [(v * KMPDU / SPTU).tolist() for v in test_3.get_v1()[0]]
print(">>>>> DISCRETIZED DATASET GENERATED <<<<<\n\n")

dvi_check = [val <= dv_limit for key, val in dv_1.items()]
dvf_check = [val <= dv_limit for key, val in dv_2.items()]
print("dvi % :", sum(dvi_check) * 100 / len(dvi_check))
print("dvf % :", sum(dvf_check) * 100 / len(dvf_check))
print("\n")
##############################################################################################

print(">>>>> WRITE PYOMO MODEL <<<<<\n")
with Timer():
    print("...create indices and parameters...")
    with Timer():
        S = pyo.ConcreteModel()  # ConcreteModel instantiates during construction
        S.name = "SequenceSearch"
        S.K = pyo.Set(initialize=k_body)  # body index
        S.T = pyo.RangeSet(num)  # timestep index
        S.H = pyo.RangeSet(h_tot)  # sequence position index

        S.w_k = pyo.Param(
            S.K, initialize=lambda model, k: bodies_data[k].weight, within=pyo.PositiveReals
        )  # scoring weights

        # Cartesian product sets
        S.KT = pyo.Set(initialize=[(k, t) for k in S.K for t in S.T])
        S.KIJ = pyo.Set(initialize=[(k, i, j) for k in S.K for i in S.T for j in range(1, i)])
        S.KIMJ = pyo.Set(initialize=list(dv_1.keys()))

        # Parameters requiring the Cartesian product sets
        S.tu_kt = pyo.Param(
            S.KT,
            initialize=lambda model, k, t: discrete_data[k]["t_tu"][t - 1].tolist(),
            within=pyo.NonNegativeReals,
        )
        S.rdu_kt = pyo.Param(
            S.KT, initialize=lambda model, k, t: discrete_data[k]["r_du"][t - 1], within=pyo.Any
        )

        S.dv1_kimj = pyo.Param(
            S.KIMJ, initialize=lambda model, k, i, m, j: dv_1[k, i, m, j], within=pyo.Any
        )
        S.dv2_kimj = pyo.Param(
            S.KIMJ, initialize=lambda model, k, i, m, j: dv_2[k, i, m, j], within=pyo.Any
        )

    # x variables and constraints
    print("...create x_kth binary variable...")
    with Timer():
        # X Binary Decision Variable: k-th body, t-th timestep, and h-th position in the sequence
        S.x_kth = pyo.Var(S.K * S.T * S.H, within=pyo.Binary)

    print("...create x parition constraint...")
    with Timer():
        # x partition constraint: selection of bodies must equal to h_tot
        S.x_partition = pyo.Constraint(rule=pyo.summation(S.x_kth) == h_tot)

    print("...create x_kt* packing constraints...")
    with Timer():
        # for each (k,t), it can only exist in one position at most
        S.x_kt_packing = pyo.ConstraintList()
        for kt in tqdm(S.KT):
            S.x_kt_packing.add(pyo.quicksum(S.x_kth[kt, :]) <= 1)

    print("...create x_**h partition constraints...")
    with Timer():
        # for each h, there must only be one body at some time
        S.x_h_packing = pyo.Constraint(
            S.H,
            rule=lambda model, h: pyo.quicksum(model.x_kth[..., h]) == 1,
        )

    print("...create x_k** max number of scientific flybys...")
    with Timer():
        # for each k, there can only be a total of N_k flybys counted
        S.flyby_limit = pyo.Constraint(
            S.K,
            rule=lambda model, k: pyo.quicksum(model.x_kth[k, ...]) <= Nk_lim,
        )

    print("...create x_*th monotonic time constraints...")
    with Timer():
        # for each h, the time must be greater than the previous h
        S.monotonic_time = pyo.ConstraintList()
        dt_tol = (DAY / SPTU).tolist()

        def time_expr(model, h):
            return pyo.quicksum(
                model.tu_kt[k, t] * model.x_kth[k, t, h] for k in model.K for t in model.T
            )

        for h in S.H:
            if h > 1:
                h_lhs = time_expr(S, h) - time_expr(S, h - 1)
                S.monotonic_time.add(h_lhs >= dt_tol)

    # y variables and constraints
    print("...create y_kij indicator variable of previous j flybys for i-th flyby...")
    with Timer():
        # Y Binary Indicator Variable: k-th body, i-th timestep, j-th previous timestep
        S.y_kij = pyo.Var(S.KIJ, within=pyo.Binary)

    print("...create y_k** packing constraints...")
    with Timer():
        # the amount of total previous flybys cannot be greater than h_tot - 1
        S.y_k_packing = pyo.ConstraintList()
        for k in tqdm(S.K):
            S.y_k_packing.add(pyo.quicksum(S.y_kij[k, ...]) <= factorial(Nk_lim - 1))

    print("...create y_kij big-M constraints...")
    with Timer():
        S.y_kij_bigm = pyo.ConstraintList()
        for kij in tqdm(S.KIJ):
            k, i, j = kij
            y_kij_lhs = pyo.quicksum(S.x_kth[k, i, h] + S.x_kth[k, j, h] for h in S.H)
            S.y_kij_bigm.add(y_kij_lhs <= 10 * S.y_kij[kij] + 1)
            S.y_kij_bigm.add(y_kij_lhs >= 2 - 10 * (1 - S.y_kij[kij]))

    print("...create z_kt indicator variable of first flyby at t for body k...")
    with Timer():
        # Z Binary Indicator Variable: k-th body, j-th first flyby timestep
        S.z_kt = pyo.Var(S.KT, within=pyo.Binary)

    print("...create z_k* packing constraints...")
    with Timer():
        # at most only ONE first flyby for body k
        S.z_k_packing = pyo.Constraint(S.K, rule=lambda model, k: pyo.quicksum(S.z_kt[k, :]) <= 1)

    print("...create z_k* and z_kt implication constraints...")
    with Timer():
        S.z_implies_x = pyo.ConstraintList()
        # if there is a flyby at that time, then (k, t) can be a first flyby
        for kt in tqdm(S.KT):
            S.z_implies_x.add(S.z_kt[kt] <= pyo.quicksum(S.x_kth[kt, :]))

        S.z_bigm_x = pyo.ConstraintList()
        # if there is a flyby for body k, then there must be a first flyby for k.
        for k in tqdm(S.K):
            S.z_bigm_x.add(h_tot * pyo.quicksum(S.z_kt[k, :]) >= pyo.quicksum(S.x_kth[k, ...]))

        S.z_implies_not_y = pyo.ConstraintList()
        # if there are previous flybys at time i, i cannot be a first flyby
        for kt in tqdm(S.KT):
            k, t = kt
            if t > 1:
                S.z_implies_not_y.add(S.z_kt[k, t] <= 1 - pyo.quicksum(S.y_kij[k, t, :]))

    print("...create (z_kt, y_kij, k_kth) -> S(r_kij) seasonal penalty terms...")
    with Timer():
        flyby_kt = {kt: [] for kt in S.KT}
        # first flyby score
        for kt in tqdm(S.KT):
            flyby_kt[kt] = S.z_kt[kt]

        # subsequent flybys
        lin_term = 0
        for kij in tqdm(S.KIJ):
            k, i, j = kij
            lin_term += lin_dots_penalty(S.rdu_kt[k, i], S.rdu_kt[k, j]) * S.y_kij[kij]
            if j == i - 1:
                flyby_kt[k, i] = lin_term
                lin_term = 0

    #### LAMBERT VARIABLES #####
    print("...create L_kimj dv table indicator variables...")
    with Timer():
        # bodies k and m, times i and j
        S.L_kimj = pyo.Var(S.KIMJ, within=pyo.Binary)

    print("...create L_ki** and L_**mj implication constraints...")
    with Timer():
        # if x_kt* isn't 1, then there can't be an L_kt** or L_**kt
        S.L_implies_x_nodes = pyo.Constraint(
            S.KIMJ,
            rule=lambda model, k, i, m, j: model.L_kimj[k, i, m, j] * 2
            <= pyo.quicksum(model.x_kth[k, i, :]) + pyo.quicksum(model.x_kth[m, j, :]),
        )

    print("...create L_kt** and L_**kt packing constraints...")
    with Timer():
        # each k, t node can only have at most one start and end.
        S.L_single_start = pyo.Constraint(
            S.KT, rule=lambda model, k, t: pyo.quicksum(model.L_kimj[k, t, ...]) <= 1
        )
        S.L_single_end = pyo.Constraint(
            S.KT,
            rule=lambda model, k, t: 1 >= pyo.quicksum(model.L_kimj[..., k, t]),
        )

    print("...create L_kimj positive dt constraints...")
    with Timer():
        # if L_kimj is selected, it must have a positive delta-t
        S.positive_dt = pyo.Constraint(
            S.KIMJ,
            rule=lambda model, k, i, m, j: model.L_kimj[k, i, m, j]
            * (model.tu_kt[m, j] - model.tu_kt[k, i])
            >= 0,
        )

    print("...create L_kimj partition constraints...")
    with Timer():
        # must have lambert checks up to h_tot - 1
        S.L_partition = pyo.Constraint(rule=pyo.summation(S.L_kimj) == h_tot - 1)

    print("...create L_kimj implication constraints for x_ki(h) and x_mj(h+1)...")
    with Timer():
        # if kimj aren't connected by h and h+1, it is not a valid lambert arc
        def h_arcs_expr(model, k, i, m, j, h):
            if h > 1:
                term = (
                    model.x_kth[k, i, h - 1] + model.x_kth[m, j, h] <= 1 + model.L_kimj[k, i, m, j]
                )
            else:
                term = pyo.Constraint.Skip
            return term

        S.L_arcs = pyo.Constraint(S.KIMJ * S.H, rule=h_arcs_expr)

    # print("...create L_kimj indicator and delta-v limit constraints...")
    # with Timer():
    # toggle lambert idx
    # S.L_ind_con = pyo.ConstraintList()
    # S.L_dv_con = pyo.ConstraintList()
    # S.L_implies_x = pyo.ConstraintList()
    # for h in tqdm(S.H):
    #     if h < S.H.at(-1):
    #         for kimj in tqdm(S.KIMJ):
    #             k, i, m, j = kimj
    # S.L_implies_x.add(S.L_kimj[kimj] <= S.x_kth[k, i, h])
    # S.L_implies_x.add(S.L_kimj[kimj] <= S.x_kth[m, j, h + 1])
    # S.L_ind_con.add(S.x_kth[k, i, h] + S.x_kth[m, j, h + 1] <= 10 * S.L_kimj[kimj] + 1)
    #             M.L_dv_con.add(M.L_kimj[kimj] * M.DV_i[kimj] <= dv_lim)
    #             M.L_dv_con.add(M.L_kimj[kimj] * M.DV_j[kimj] <= dv_lim)

    print("...create grand tour bonus indicator variables Zp, Gp, Zc, Gc, and big-M constraints...")
    with Timer():
        S.count_k_planets = pyo.Var(S.K, within=pyo.Binary)  # planets and yandi
        S.all_planets = pyo.Var(initialize=0, within=pyo.Binary)  # all planets indicator
        S.count_p_bigm = pyo.ConstraintList()
        for k in S.K:
            count_p_lhs = pyo.quicksum(S.x_kth[k, t, h] for h in S.H for t in S.T)
            S.count_p_bigm.add(count_p_lhs <= 2 * num * S.count_k_planets[k])
            S.count_p_bigm.add(count_p_lhs - 1 >= 2 * num * (S.count_k_planets[k] - 1))

        S.all_planets_bigm = pyo.ConstraintList()
        S.all_planets_bigm.add(
            pyo.quicksum(S.count_k_planets[k] for k in S.K) - gt_bodies <= S.all_planets * 20
        )
        S.all_planets_bigm.add(
            pyo.quicksum(S.count_k_planets[k] for k in S.K) - gt_bodies >= (S.all_planets - 1) * 20
        )
        GT_bonus = 1.0 + 0.3 * S.all_planets

    # Objective function
    print("...create objective function...")
    with Timer():
        S.maximize_score = pyo.Objective(
            rule=GT_bonus * pyo.quicksum(S.w_k[k] * flyby_kt[kt] for kt in S.KT),
            sense=pyo.maximize,
        )

    print("...total model setup time...")
print(">>>>> MODEL SETUP COMPLETE <<<<<\n\n")

print(">>>>> SET UP ASSUMED INITIAL ARCS USING FIXED VARIABLES AND CONSTRAINTS <<<<<\n")
with Timer():
    # Solver setup
    # from solve first arc, the intial body is Planet X at Yo = 3 years.
    # # Hit Rogue One and then Wakonyingo
    S.first_arcs = pyo.ConstraintList()
    S.first_arcs.add(pyo.quicksum(S.x_kth[10, t, 1] for t in range(1, 3 * perYear + 1)) == 1)
    S.first_arcs.add(pyo.quicksum(S.x_kth[9, :, 2]) == 1)
    S.first_arcs.add(pyo.quicksum(S.x_kth[8, :, 3]) == 1)
print(">>>>> ASSUMED INITIAL ARCS CONSTRAINED <<<<<\n\n")

## Run
print(">>>>> RUN SOLVER <<<<<\n")
solver = SolverFactory(solver, solver_io="nl")
results = solver.solve(S, tee=True)


print("\n...iteration 1 solved...\n")

if results.solver.termination_condition == TerminationCondition.infeasible:
    print("Infeasible, sorry :(\n")
    # print("...generate miniminal intractable system...")
    # iis.compute_infeasibility_explanation(S, solver="scip")
else:
    sequence = []
    for kt in S.KT:

        k, t = kt
        for h in S.H:
            value = pyo.value(S.x_kth[kt, h])
            if value > 0.5:
                sequence.append(
                    (
                        h,
                        (bodies_data[k].name, kt),
                        np.round((discrete_data[k]["t_tu"][t] * SPTU / YEAR), 3).tolist(),
                    )
                )
    sequence = sorted(sequence, key=lambda x: x[2])
    print("Sequence (h-th position, k-th body, t in years):")
    pprint(sequence)
    short_seq = [each[1][1] for each in sequence]
    print("\n")
    if S.find_component("L_kimj"):
        print("Number of lambert arcs: ", int(np.round(pyo.value(pyo.summation(S.L_kimj)))), "\n")
        print("Lambert arcs (from k @ t[i], to m @ t[j]):")
        lambert_arcs = [
            [short_seq.index((k, i)), (k, i), (m, j)]
            for (k, i, m, j), v in S.L_kimj.items()
            if pyo.value(v) > 0.5
        ]
        lambert_arcs = sorted(lambert_arcs)
        for arc in lambert_arcs:
            print(arc[1:])
        print("\n")
    print("Number of repeated flybys:", int(np.round(pyo.value(pyo.summation(S.y_kij)))), "\n")
    print("First flyby keys (k, i-th):")
    for k, v in S.z_kt.items():
        if pyo.value(v) > 0.5:
            print(k)

    print("\n")
    print("Repeat flyby keys (k, i-th, j-th prev):")
    for k, v in S.y_kij.items():
        if pyo.value(v) > 0.5:
            print(k)
    print("\n...iteration 1 complete...\n")

    if sol_iter > 1:
        print("...start no-good cuts for iteration 2...")

        # No-good cuts for multiple solutions
        S.ng_cuts = pyo.ConstraintList()
        for sol in range(1, sol_iter):
            with Timer():
                expr = 0
                for kth in S.x_kth:
                    if pyo.value(S.x_kth[kth]) < 0.5:
                        expr += S.x_kth[kth]
                    else:
                        expr += 1 - S.x_kth[kth]
                S.ng_cuts.add(expr >= 1)

            print("\n>>>>> RUN SOLVER <<<<<\n")
            results = solver.solve(S, tee=solver_log)
            print(f"\n...iteration {sol+1} solved...\n")
            if results.solver.termination_condition == TerminationCondition.infeasible:
                print("Infeasible, sorry :(")
                # print("...generate miniminal intractable system...")
                # iis.compute_infeasibility_explanation(S, solver="scip")
            else:
                sequence = []
                for kt in S.KT:
                    k, t = kt
                    for h in S.H:
                        value = pyo.value(S.x_kth[kt, h])
                        if value > 0.5:
                            sequence.append(
                                (
                                    h,
                                    (bodies_data[k].name, kt),
                                    np.round(
                                        (discrete_data[k]["t_tu"][t] * SPTU / YEAR), 3
                                    ).tolist(),
                                )
                            )
                sequence = sorted(sequence, key=lambda x: x[2])
                print("Sequence (h-th position, k-th body, t in years):")
                pprint(sequence)
                short_seq = [each[1][1] for each in sequence]
                print("\n")
                if S.find_component("L_kimj"):
                    print(
                        "Number of lambert arcs: ",
                        int(np.round(pyo.value(pyo.summation(S.L_kimj)))),
                        "\n",
                    )
                    print("Lambert arcs (from k @ t[i], to m @ t[j]):")
                    lambert_arcs = [
                        [short_seq.index((k, i)), (k, i), (m, j)]
                        for (k, i, m, j), v in S.L_kimj.items()
                        if pyo.value(v) > 0.5
                    ]
                    lambert_arcs = sorted(lambert_arcs)
                    for arc in lambert_arcs:
                        print(arc[1:])
                    print("\n")
                print(
                    "Number of repeated flybys: ",
                    int(np.round(pyo.value(pyo.summation(S.y_kij)))),
                    "\n",
                )
                print("First flyby keys (k, i-th):")
                for k, v in S.z_kt.items():
                    if pyo.value(v) > 0.5:
                        print(k)
                print("\n")
                print("Repeat flyby keys (k, i-th, j-th prev):")
                for k, v in S.y_kij.items():
                    if pyo.value(v) > 0.5:
                        print(k)

                print(f"\n...iteration {sol+1} complete...\n")
                if sol + 2 <= sol_iter:
                    print(f"...start no-good cuts for iteration {sol+2}...")


print(">>>>> FINISHED RUNNING SOLVER <<<<<\n\n")
