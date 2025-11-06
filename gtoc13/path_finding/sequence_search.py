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
from pyomo.opt import SolverFactory, TerminationCondition
import numpy as np

# from pathlib import Path

from gtoc13 import DAY, KMPDU, SPTU, YEAR, bodies_data

from tqdm import tqdm
import time
from itertools import product
from pprint import pprint
from math import factorial

np.set_printoptions(legacy="1.25")
############### CONFIG ###############
Yo = 0
Yf = 10
perYear = 4
dv_limit = 2000
h_tot = 5  # sequence length
gt_bodies = 5
gt_smalls = 13
Nk_lim = 3  # 13
sol_iter = 1
solver_log = True
solver = "gurobi"  # AMPL-format solvers
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
                    v_dtu=np.array(
                        [body.get_state(timestep[idx]).v * SPTU / KMPDU for idx in range(num)]
                    ),
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
                for i, __ in enumerate(discrete_data[k]["t_tu"])
                for j, __ in enumerate(discrete_data[m]["t_tu"])
            ]
        ):
            k, i, m, j = kimj
            tu_i = discrete_data[k]["t_tu"][i]
            tu_j = discrete_data[m]["t_tu"][j]
            vk_dtu_i = discrete_data[k]["v_dtu"][i]
            vm_dtu_j = discrete_data[m]["v_dtu"][j]
            tof = (tu_j - tu_i).tolist()
            # if tof > 0:
            #     ki_to_mj = lambert_problem(
            #         r1=discrete_data[k]["r_du"][i].tolist(),
            #         r2=discrete_data[m]["r_du"][j].tolist(),
            #         tof=(tu_j - tu_i).tolist(),
            #     )
            #     dv_1[(k, i + 1, m, j + 1)] = np.linalg.norm(
            #         np.array(ki_to_mj.get_v1()[0]) - vk_dtu_i
            #     )
            #     dv_2[(k, i + 1, m, j + 1)] = np.linalg.norm(
            #         np.array(ki_to_mj.get_v2()[0]) - vm_dtu_j
            #     )

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
            tqdm(S.K), initialize=lambda model, k: bodies_data[k].weight, within=pyo.PositiveReals
        )  # scoring weights

        # Cartesian product sets
        S.KT = pyo.Set(initialize=tqdm([(k, t) for k in S.K for t in S.T]))
        S.KIJ = pyo.Set(initialize=tqdm([(k, i, j) for k in S.K for i in S.T for j in range(1, i)]))
        S.KIMJ = pyo.Set(initialize=tqdm(list(dv_1.keys())))

        # Parameters requiring the Cartesian product sets
        S.tu_kt = pyo.Param(
            tqdm(S.KT),
            initialize=lambda model, k, t: discrete_data[k]["t_tu"][t - 1].tolist(),
            within=pyo.NonNegativeReals,
        )
        S.rdu_kt = pyo.Param(
            tqdm(S.KT),
            initialize=lambda model, k, t: discrete_data[k]["r_du"][t - 1],
            within=pyo.Any,
        )

        S.dv1_kimj = pyo.Param(
            tqdm(S.KIMJ), initialize=lambda model, k, i, m, j: dv_1[k, i, m, j], within=pyo.Any
        )
        S.dv2_kimj = pyo.Param(
            tqdm(S.KIMJ), initialize=lambda model, k, i, m, j: dv_2[k, i, m, j], within=pyo.Any
        )

    # x variables and constraints
    print("...create x_kth binary variable...")
    with Timer():
        # X Binary Decision Variable: k-th body, t-th timestep, and h-th position in the sequence
        S.x_kth = pyo.Var(tqdm(S.K * S.T * S.H), within=pyo.Binary)

    print("...create x parition constraint...")
    with Timer():
        # x partition constraint: selection of bodies must equal to h_tot
        S.x_partition = pyo.Constraint(rule=pyo.summation(S.x_kth) == h_tot)

    print("...create x_kt* packing constraints...")
    with Timer():
        # for each (k,t), it can only exist in one position at most
        S.x_kt_packing = pyo.Constraint(
            tqdm(S.KT), rule=lambda model, k, t: pyo.quicksum(model.x_kth[k, t, :]) <= 1
        )

    print("...create x_**h partition constraints...")
    with Timer():
        # for each h, there must only be one body at some time
        S.x_h_packing = pyo.Constraint(
            tqdm(S.H),
            rule=lambda model, h: pyo.quicksum(model.x_kth[..., h]) == 1,
        )

    print("...create x_k** max number of scientific flybys...")
    with Timer():
        # for each k, there can only be a total of N_k flybys counted
        S.flyby_limit = pyo.Constraint(
            tqdm(S.K),
            rule=lambda model, k: pyo.quicksum(model.x_kth[k, ...]) <= Nk_lim,
        )

    print("...create x_*th monotonic time constraints...")
    with Timer():
        # for each h, the time must be greater than the previous h

        dt_tol = (DAY / SPTU).tolist()

        def monotime_rule(model, h):
            if h > 1:
                term = (
                    pyo.quicksum(
                        model.tu_kt[k, t] * model.x_kth[k, t, h] for k in model.K for t in model.T
                    )
                    - pyo.quicksum(
                        model.tu_kt[k, t] * model.x_kth[k, t, h - 1]
                        for k in model.K
                        for t in model.T
                    )
                    >= dt_tol
                )
            else:
                term = pyo.Constraint.Skip
            return term

        S.monotonic_time = pyo.Constraint(tqdm(S.H), rule=monotime_rule)

    # y variables and constraints
    print("...create y_kij indicator variable of previous j flybys for i-th flyby...")
    with Timer():
        # Y Binary Indicator Variable: k-th body, i-th timestep, j-th previous timestep
        S.y_kij = pyo.Var(tqdm(S.KIJ), within=pyo.Binary)

    print("...create y_k** packing constraints...")
    with Timer():
        # the amount of total previous flybys cannot be greater than h_tot - 1
        S.y_k_packing = pyo.Constraint(
            tqdm(S.K),
            rule=lambda model, k: pyo.quicksum(model.y_kij[k, ...]) <= factorial(Nk_lim - 1),
        )

    print("...create y_kij big-M constraints...")
    with Timer():
        # if there is both x_ki* and x_kj*, then there must be a y_kij
        S.y_bigm1_x = pyo.Constraint(
            tqdm(S.KIJ),
            rule=lambda model, k, i, j: pyo.quicksum(
                model.x_kth[k, i, h] + model.x_kth[k, j, h] for h in model.H
            )
            <= 10 * model.y_kij[k, i, j] + 1,
        )
        # if there isn't both x_ki* and x_kj*, then there cannot be a y_kij
        S.y_bigm2_x = pyo.Constraint(
            tqdm(S.KIJ),
            rule=lambda model, k, i, j: pyo.quicksum(
                model.x_kth[k, i, h] + model.x_kth[k, j, h] for h in model.H
            )
            >= 2 - 10 * (1 - model.y_kij[k, i, j]),
        )

    print("...create z_kt indicator variable of first flyby at t for body k...")
    with Timer():
        # Z Binary Indicator Variable: k-th body, j-th first flyby timestep
        S.z_kt = pyo.Var(tqdm(S.KT), within=pyo.Binary)

    print("...create z_k* packing constraints...")
    with Timer():
        # at most only ONE first flyby for body k
        S.z_k_packing = pyo.Constraint(
            tqdm(S.K), rule=lambda model, k: pyo.quicksum(model.z_kt[k, :]) <= 1
        )

    print("...create z_k* and z_kt implication constraints...")
    with Timer():
        # if there is a flyby at that time, then (k, t) can be a first flyby
        S.z_implies_x = pyo.Constraint(
            tqdm(S.KT),
            rule=lambda model, k, t: model.z_kt[k, t] <= pyo.quicksum(model.x_kth[k, t, :]),
        )
        # if there is a flyby for body k, then there must be a first flyby for k.
        S.z_bigm_x = pyo.Constraint(
            tqdm(S.K),
            rule=lambda model, k: h_tot * pyo.quicksum(model.z_kt[k, :])
            >= pyo.quicksum(model.x_kth[k, ...]),
        )
        # if there are previous flybys at time i, i cannot be a first flyby
        S.z_implies_not_y = pyo.Constraint(
            tqdm(S.KT),
            rule=lambda model, k, t: (
                model.z_kt[k, t] <= 1 - pyo.quicksum(model.y_kij[k, t, :])
                if t > 1
                else pyo.Constraint.Feasible
            ),
        )

    print("...create L_kimj arc indicator variables...")
    with Timer():
        # bodies k and m, times i and j
        S.L_kimj = pyo.Var(tqdm(S.KIMJ), within=pyo.Binary)

    print("...create L_ki** and L_**mj implication constraints...")
    with Timer():
        # if x_kt* isn't 1, then there can't be an L_kt** or L_**kt
        S.L_implies_x_nodes = pyo.Constraint(
            tqdm(S.KIMJ),
            rule=lambda model, k, i, m, j: model.L_kimj[k, i, m, j] * 2
            <= pyo.quicksum(model.x_kth[k, i, h] + model.x_kth[m, j, h] for h in model.H),
        )

    print("...create L_kt** and L_**kt packing constraints...")
    with Timer():
        # each k, t node can only have at most one start and end.
        S.L_single_start = pyo.Constraint(
            tqdm(S.KT), rule=lambda model, k, t: pyo.quicksum(model.L_kimj[k, t, ...]) <= 1
        )
        S.L_single_end = pyo.Constraint(
            tqdm(S.KT), rule=lambda model, k, t: 1 >= pyo.quicksum(model.L_kimj[..., k, t])
        )

    print("...create L_kimj positive dt constraints...")
    with Timer():
        # if L_kimj is selected, it must have a positive delta-t
        S.positive_dt = pyo.Constraint(
            tqdm(S.KIMJ),
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

        S.L_arcs = pyo.Constraint(tqdm(S.KIMJ * S.H), rule=h_arcs_expr)

    print("...create L_kimj delta-v limit constraints...")
    with Timer():
        # if the lambert arc is used, it must not exceed the dv limits.
        S.dv1_limit = pyo.Constraint(
            tqdm(S.KIMJ),
            rule=lambda model, k, i, m, j: model.L_kimj[k, i, m, j] * model.dv1_kimj[k, i, m, j]
            <= dv_limit,
        )
        S.dv2_limit = pyo.Constraint(
            tqdm(S.KIMJ),
            rule=lambda model, k, i, m, j: model.L_kimj[kimj] * model.dv2_kimj[kimj] <= dv_limit,
        )

    print("...create grand tour bonus indicator variables Zp, Gp, Zc, Gc, and big-M constraints...")
    with Timer():
        S.count_k_planets = pyo.Var(tqdm(S.K), within=pyo.Binary)  # planets and yandi
        S.all_planets = pyo.Var(initialize=0, within=pyo.Binary)  # all planets indicator
        S.count_p_bigm = pyo.ConstraintList()

        S.count_p_bigm1 = pyo.Constraint(
            tqdm(S.K),
            rule=lambda model, k: pyo.quicksum(model.x_kth[k, ...])
            <= 2 * num * model.count_k_planets[k],
        )
        S.count_p_bigm2 = pyo.Constraint(
            tqdm(S.K),
            rule=lambda model, k: pyo.quicksum(model.x_kth[k, ...]) - 1
            >= 2 * num * (model.count_k_planets[k] - 1),
        )
        S.all_planets_bigm1 = pyo.Constraint(
            rule=pyo.summation(S.count_k_planets) - gt_bodies <= S.all_planets * 20
        )
        S.all_planets_bigm2 = pyo.Constraint(
            rule=pyo.summation(S.count_k_planets) - gt_bodies >= (S.all_planets - 1) * 20
        )

    ##### Objective Function and Scoring #####
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

    print("...create objective function...")
    with Timer():
        GT_bonus = 1.0 + 0.3 * S.all_planets
        S.maximize_score = pyo.Objective(
            rule=GT_bonus * pyo.quicksum(S.w_k[k] * flyby_kt[kt] for kt in tqdm(S.KT)),
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

else:
    sequence = []
    for kt in S.KT:

        k, t = kt
        for h in S.H:
            value = pyo.value(S.x_kth[kt, h])
            if value > 0.5:
                tu = np.round((discrete_data[k]["t_tu"][t] * SPTU / YEAR).tolist(), 3)
                sequence.append(
                    (
                        h,
                        [bodies_data[k].name, kt],
                        f"{tu} years",
                    )
                )
    sequence = sorted(sequence, key=lambda x: x[2])
    print("Sequence (h-th position, k-th body, t in years):")
    pprint(sequence)
    short_seq = [each[1][1] for each in sequence]
    print("\n")
    if S.find_component("L_kimj"):
        print("Number of lambert arcs: ", int(np.round(pyo.value(pyo.summation(S.L_kimj)))), "\n")
        print("Lambert arcs (k, i) to (m, j):")
        lambert_arcs = [
            [short_seq.index((k, i)) + 1, f"({k}, {i}) to ({m}, {j})"]
            for (k, i, m, j), v in S.L_kimj.items()
            if pyo.value(v) > 0.5
        ]
        lambert_arcs = sorted(lambert_arcs)
        for arc in lambert_arcs:
            print(arc)
        print("\n")
    print("Number of repeated flybys:", int(np.round(pyo.value(pyo.summation(S.y_kij)))), "\n")
    print("First flyby keys (k, i-th):")
    for k, v in S.z_kt.items():
        if pyo.value(v) > 0.5:
            print(k)
    if pyo.value(pyo.summation(S.y_kij) > 0):
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
            else:
                sequence = []
                for kt in S.KT:
                    k, t = kt
                    for h in S.H:
                        value = pyo.value(S.x_kth[kt, h])
                        if value > 0.5:
                            tu = np.round((discrete_data[k]["t_tu"][t] * SPTU / YEAR).tolist(), 3)
                            sequence.append(
                                (
                                    h,
                                    [bodies_data[k].name, kt],
                                    f"{tu} years",
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
                    print("Lambert arcs (k, i) to (m, j):")
                    lambert_arcs = [
                        [short_seq.index((k, i)) + 1, f"({k}, {i}) to ({m}, {j})"]
                        for (k, i, m, j), v in S.L_kimj.items()
                        if pyo.value(v) > 0.5
                    ]
                    lambert_arcs = sorted(lambert_arcs)
                    for arc in lambert_arcs:
                        print(arc)
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
                if pyo.value(pyo.summation(S.y_kij) > 0):
                    print("\n")
                    print("Repeat flyby keys (k, i-th, j-th prev):")
                    for k, v in S.y_kij.items():
                        if pyo.value(v) > 0.5:
                            print(k)

                print(f"\n...iteration {sol+1} complete...\n")
                if sol + 2 <= sol_iter:
                    print(f"...start no-good cuts for iteration {sol+2}...")


print(">>>>> FINISHED RUNNING SOLVER <<<<<\n\n")
