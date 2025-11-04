"""
Binary Integer Non-Linear Programming formulation compute the most "valuable" sequences
for scientific flybys. Assisting flybys not counted; they should be calculated during
sequence trajectory evaluation.

Assumes a starting year and body; solve_first_arc.py should determine which body and when.

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
from pyomo.opt import TerminationCondition as TermCond

import numpy as np


# from pathlib import Path

from gtoc13 import DAY, KMPDU, SPTU, YEAR, MU_ALTAIRA, bodies_data

from pykep import lambert_problem
from tqdm import tqdm
import time
from itertools import product
from pprint import pprint
from math import factorial


class Timer:
    def __enter__(self):
        self._enter_time = time.time()

    def __exit__(self, *exc_args):
        self._exit_time = time.time()
        print(f"{self._exit_time - self._enter_time:.2f} seconds elapsed\n")


##############################################################################################
print(">>>>> CREATE DISCRETIZED DATASET <<<<<\n")
with Timer():
    ############# CONFIG #############
    Yo = 0
    Yf = 10
    perYear = 4
    dv_limit = 5000
    h_tot = 5  # sequence length
    sol_iter = 1
    solver_log = True
    #############
    To = Yo * YEAR
    Tf = Yf * YEAR  # years in seconds
    num = int(np.ceil((Tf - To) / (YEAR / perYear)))
    #############
    dv_limit *= SPTU.tolist() / KMPDU  # km/s
    gt_bodies = 4
    gt_smalls = 13
    Nk_lim = 4  # 13

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
    # print("...calculating lambert delta-vs...")
    # with Timer():
    #     dv_1 = {}
    #     dv_2 = {}
    #     for kimj in tqdm(
    #         [
    #             (k, i, m, j)
    #             for (k, m) in list(product(k_body, repeat=2))
    #             for i, tu_i in enumerate(discrete_data[k]["t_tu"])
    #             for j, tu_j in enumerate(discrete_data[m]["t_tu"])
    #         ]
    #     ):
    #         k, i, m, j = kimj
    #         tof = (discrete_data[m]["t_tu"][j] - discrete_data[k]["t_tu"][i]).tolist()
    #         if tof > 0:
    #             ki_to_mj = lambert_problem(
    #                 r1=discrete_data[k]["r_du"][i].tolist(),
    #                 r2=discrete_data[m]["r_du"][j].tolist(),
    #                 tof=(discrete_data[m]["t_tu"][j] - discrete_data[k]["t_tu"][i]).tolist(),
    #             )
    #             dv_1[kimj] = np.array(ki_to_mj.get_v1()[0])
    #             dv_2[kimj] = np.array(ki_to_mj.get_v2()[0])
    #         else:
    #             dv_1[kimj], dv_2[kimj] = np.random.rand(2) * (dv_limit + 5.0)
    #     del ki_to_mj
    # [(v  * KMPDU/SPTU).tolist() for v in test_3.get_v1()[0]]
print(">>>>> DISCRETIZED DATASET GENERATED <<<<<\n\n")

# dvi_check = [val <= dv_limit for key, val in dv_1.items()]
# dvf_check = [val <= dv_limit for key, val in dv_2.items()]
# print("dvi % :", sum(dvi_check) * 100 / len(dvi_check))
# print("dvf % :", sum(dvf_check) * 100 / len(dvf_check))
# print("\n")
##############################################################################################

print(">>>>> WRITE PYOMO MODEL <<<<<\n")
with Timer():
    print("...create indices and parameters...")
    with Timer():
        S = pyo.ConcreteModel()  # ConcreteModel instantiates during construction
        S.K = pyo.Set(initialize=k_body)  # body index
        S.T = pyo.RangeSet(num)  # timestep index
        S.H = pyo.RangeSet(h_tot)  # sequence position index

        S.w_k = pyo.Param(
            S.K, initialize=lambda model, k: bodies_data[k].weight, within=pyo.PositiveReals
        )  # scoring weights

        # Cartesian product sets
        S.KT = pyo.Set(initialize=[(k, t) for k in S.K for t in S.T])
        # S.KJ = pyo.Set(initialize=[(k, j) for k in S.K for j in range(1, S.T.at(-1))])
        S.KIJ = pyo.Set(initialize=[(k, i, j) for k in S.K for i in S.T for j in range(1, i)])
        # S.KIMJ = pyo.Set(initialize=list(dv_1.keys()))

        # Parameters requiring the Cartesian product sets
        S.tu_kt = pyo.Param(
            S.KT,
            initialize=lambda model, k, t: discrete_data[k]["t_tu"][t - 1].tolist(),
            within=pyo.NonNegativeReals,
        )
        S.rdu_kt = pyo.Param(
            S.KT, initialize=lambda model, k, t: discrete_data[k]["r_du"][t - 1], within=pyo.Any
        )

        # S.dv1_kimj = pyo.Param(
        #     S.KIMJ, initialize=lambda model, k, i, m, j: dv_1[k, i, m, j], within=pyo.Any
        # )
        # S.dv2_kimj = pyo.Param(
        #     S.KIMJ, initialize=lambda model, k, i, m, j: dv_2[k, i, m, j], within=pyo.Any
        # )

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

        # S.z_x_and_y = pyo.ConstraintList()
        # for k in tqdm(S.K):
        #     for i in S.T:
        #         if i < S.T.at(-1):
        #             Y_term = pyo.quicksum(S.y_kij[k, :, i])
        #         else:
        #             Y_term = 1
        #         X_term = pyo.quicksum(S.x_kth[k, i, :])
        #         S.z_x_and_y.add(S.z_kt[k, i] + Nk_lim >= X_term + Y_term)

    # print("...create (Z_kj, Y_kij, X_kth) -> S(r_kij) seasonal penalty terms...")
    # with Timer():
    #     flyby_terms = {k: {} for k in S.K}
    #     for k in tqdm(S.K):
    #         for i in S.T:
    #             if i > 1:
    #                 expn_term = 0
    #                 # first flyby term
    #                 first_term = S.z_kj[k, i - 1]
    #                 for j in range(1, i):
    #                     # subsequent penalty terms
    #                     rhat_i = S.rdu_kt[k, i] / np.linalg.norm(S.rdu_kt[k, i])
    #                     rhat_j = S.rdu_kt[k, j] / np.linalg.norm(S.rdu_kt[k, j])
    #                     dot_products = np.dot(rhat_i, rhat_j).tolist()
    #                     angles_deg = np.arccos(dot_products) * 180 / np.pi
    #                     expn_term += np.exp(-(angles_deg**2) / 50.0) * S.y_kij[k, i, j]
    #                 penalty_term = (0.1 + 0.9 / (1 + 10 * expn_term)) * pyo.quicksum(
    #                     S.x_kth[k, i, :]
    #                 )
    #                 flyby_terms[k][i] = (first_term + penalty_term) * S.w_k[k]

    # #### LAMBERT VARIABLES #####
    # print("...create L_kimj dv table indicator variables...")
    # with Timer():
    #     # bodies k and m, times i and j
    #     M.L_kimj = pyo.Var(M.kimj, within=pyo.Binary)

    # print("...create L_kimj partition constraints...")
    # with Timer():
    #     # must have lambert checks up to h_tot - 1
    #     M.L_partition_con = pyo.Constraint(rule=pyo.summation(M.L_kimj) == h_tot - 1)

    # print("...create L_ki** and L_**mj start and end constraints...")
    # with Timer():
    #     # can't start from the same place more than once
    #     M.L_start_con = pyo.ConstraintList()
    #     M.L_end_con = pyo.ConstraintList()
    #     for kt in tqdm(M.kt):
    #         M.L_start_con.add(pyo.quicksum(M.L_kimj[kt, ...]) <= 1)
    #         M.L_end_con.add(pyo.quicksum(M.L_kimj[..., kt]) <= 1)

    # print("...create L_kimj indicator and delta-v limit constraints...")
    # with Timer():
    #     # toggle lambert idx
    #     M.L_ind_con = pyo.ConstraintList()
    #     M.L_dv_con = pyo.ConstraintList()
    #     for h in tqdm(range(1, len(M.h))):
    #         for kimj in tqdm(M.kimj):
    #             k, i, m, j = kimj
    #             M.L_ind_con.add(M.x_kth[k, i, h] + M.x_kth[m, j, h + 1] <= 10 * M.L_kimj[kimj] + 1)

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
        # GT_bonus = pyo.quicksum(M.zp_k[k] for k in M.k)
        # GT_bonus = 1.3

    ## Objective function
    # print("...create objective function...")
    # with Timer():
    #     S.maximize_score = pyo.Objective(
    #         rule=GT_bonus * pyo.quicksum(flyby_terms[k][j] for kj in S.KJ),
    #         sense=pyo.maximize,
    #     )
    S.quick_max = pyo.Objective(
        rule=GT_bonus * pyo.quicksum(S.w_k[k] * pyo.quicksum(S.x_kth[k, ...]) for k in S.K),
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
    # S.first_arcs.add(pyo.quicksum(S.x_kth[9, :, 2]) == 1)
    # S.first_arcs.add(pyo.quicksum(S.x_kth[8, :, 3]) == 1)
print(">>>>> ASSUMED INITIAL ARCS CONSTRAINED <<<<<\n\n")

## Run
print(">>>>> RUN SOLVER <<<<<\n")
solver = pyo.SolverFactory("scip", solver_io="nl")
results = solver.solve(S, tee=True)
print("\n...iteration 1 solved...\n")

if results.solver.termination_condition == TermCond.infeasible:
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
                        f"position {h}",
                        bodies_data[k].name,
                        (discrete_data[k]["t_tu"][t] * SPTU / YEAR).tolist(),
                    )
                )
    print("Sequence:")
    pprint(sorted(sequence, key=lambda x: x[2]))
    print("\n")
    # print("Number of lambert arcs: ", int(np.ceil(pyo.value(pyo.summation(M.L_kimj)))),"\n")
    print("Number of repeated flybys: ", int(np.ceil(pyo.value(pyo.summation(S.y_kij)))), "\n")
    print("Flyby keys:")
    for k, v in S.y_kij.items():
        if pyo.value(v) > 0.5:
            print(k)
    print("\n")
    print("First flyby keys:")
    for k, v in S.z_kt.items():
        if pyo.value(v) > 0.5:
            print(k)
    print("\n...iteration 1 complete...\n")

    if sol_iter > 1:
        print("...start no-good cuts for iteration 2...")
        with Timer():
            # No-good cuts for multiple solutions
            S.ng_cuts = pyo.ConstraintList()
            for sol in range(1, sol_iter):
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
                if results.solver.termination_condition == TermCond.infeasible:
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
                                        f"position {h}",
                                        bodies_data[k].name,
                                        (discrete_data[k]["t_tu"][t] * SPTU / YEAR).tolist(),
                                    )
                                )
                    print("Sequence:")
                    pprint(sorted(sequence, key=lambda x: x[2]))
                    print("\n")
                    # print("Number of lambert arcs: ", int(np.ceil(pyo.value(pyo.summation(M.L_kimj)))),"\n")
                    print(
                        "Number of repeated flybys: ",
                        int(np.round(pyo.value(pyo.summation(S.y_kij)))),
                        "\n",
                    )
                    print("Flyby keys:")
                    for k, v in S.y_kij.items():
                        if pyo.value(v) > 0.5:
                            print(k)
                    print("\n")
                    print("First flyby keys:")
                    for k, v in S.z_kt.items():
                        if pyo.value(v) > 0.5:
                            print(k)
                    print(f"\n...iteration {sol+1} complete...\n")
                    if sol + 2 <= sol_iter:
                        print(f"...start no-good cuts for iteration {sol+2}...")


print(">>>>> FINISHED RUNNING SOLVER <<<<<\n\n")
