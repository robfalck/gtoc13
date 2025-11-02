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
from pyomo.opt import TerminationCondition
import numpy as np
from math import factorial

# from pathlib import Path

from gtoc13 import (
    DAY,
    KMPDU,
    SPTU,
    YEAR,
    MU_ALTAIRA,
    bodies_data,
    lambert_delta_v,
)

from tqdm import tqdm
import time
from itertools import product
from pprint import pprint


class Timer:
    def __enter__(self):
        self._enter_time = time.time()

    def __exit__(self, *exc_args):
        self._exit_time = time.time()
        print(f"{self._exit_time - self._enter_time:.2f} seconds elapsed\n")


def orbital_period(body):
    """
    Returns orbital period in seconds.
    """
    return 2 * np.pi * (body.elements.a) ** (3 / 2) / np.sqrt(MU_ALTAIRA)


print("...CREATE DISCRETIZED DATASET...")
with Timer():
    ## Generate position tables for just the bodies
    discrete_rs = dict()
    To = 5 * YEAR
    Tf = 15 * YEAR  # years in seconds
    Nk_lim = 3  # 13
    dv_lim = 2500  # km/s
    gt_bodies = 5
    gt_smalls = 13
    h_tot = 6  # sequence length
    b_idxs = discrete_rs.keys()

    ## Should tune the granularity of the discretization
    print("...discretizing body data...")
    with Timer():
        for b_idx, body in tqdm(bodies_data.items()):
            if body.is_planet() or body.name == "Yandi":
                TP = orbital_period(body)
                if TP / 2 > YEAR:
                    num = int(np.ceil((Tf - To) / (YEAR)))
                elif TP * 1.75 > YEAR:
                    num = int(np.ceil((Tf - To) / (YEAR / 2)))
                else:
                    num = int(np.ceil((Tf - To) / (TP * 12)))
                # num = int(np.ceil(Tf / (YEAR / 4))) if YEAR / 4 < TP else int(np.ceil(Tf / (TP)))
                ts = np.linspace(To, Tf, num)
                discrete_rs[b_idx] = dict(
                    TP=TP,
                    rf=np.array([body.get_state(ts[idx]).r / KMPDU for idx in range(num)]),
                    ts=ts,
                    t_k=num,
                )

    print("...calculating lambert delta-vs...")
    with Timer():
        dv_i = {}
        dv_f = {}
        lambert_from_to = list(product(b_idxs, repeat=2))
        for kimj in tqdm(
            [
                (k, i, m, j)
                for (k, m) in lambert_from_to
                for i in range(discrete_rs[k]["t_k"])
                for j in range(discrete_rs[m]["t_k"])
            ]
        ):
            k, i, m, j = kimj
            t_i = discrete_rs[k]["ts"][i]
            t_j = discrete_rs[m]["ts"][j]
            dv_i[kimj], dv_f[kimj] = np.random.rand(2) * 5000

    print("...DISCRETIZED DATASET GENERATED...")

dvi_check = [val <= dv_lim for key, val in dv_i.items()]
dvf_check = [val <= dv_lim for key, val in dv_f.items()]
print("dvi % :", sum(dvi_check) * 100 / len(dvi_check))
print("dvf % :", sum(dvf_check) * 100 / len(dvf_check))
print("\n")

print("...WRITE PYOMO MODEL...\n")
## WRITE PYOMO MODEL ##
with Timer():
    print("...initialize model parameters...")
    with Timer():
        M = pyo.ConcreteModel()
        M.k = pyo.Set(initialize=b_idxs)
        M.h = pyo.RangeSet(1, h_tot)

        M.w_k = pyo.Param(
            M.k, initialize=lambda model, k: bodies_data[k].weight, within=pyo.PositiveReals
        )
        M.t_num = pyo.Param(
            M.k, initialize=lambda model, k: discrete_rs[k]["t_k"], within=pyo.PositiveIntegers
        )

        # Cartesian product sets
        M.kt = pyo.Set(initialize=[(k, t) for k in M.k for t in range(M.t_num[k])])
        M.kij = pyo.Set(
            initialize=[(k, i, j) for k in M.k for i in range(M.t_num[k]) for j in range(i)]
        )
        M.kj = pyo.Set(initialize=[(k, j) for k in M.k for j in range(M.t_num[k] - 1)])
        M.kimj = pyo.Set(initialize=list(dv_i.keys()))

        # Parameters requiring the Cartesian product sets
        M.tfs = pyo.Param(
            M.kt, initialize=lambda model, k, t: discrete_rs[k]["ts"][t], within=pyo.PositiveReals
        )
        M.rfs = pyo.Param(
            M.kt, initialize=lambda model, k, t: discrete_rs[k]["rf"][t], within=pyo.Any
        )

        M.DV_i = pyo.Param(
            M.kimj, initialize=lambda model, k, i, m, j: dv_i[k, i, m, j], within=pyo.Reals
        )
        M.DV_j = pyo.Param(
            M.kimj, initialize=lambda model, k, i, m, j: dv_f[k, i, m, j], within=pyo.Reals
        )

    print("...create X_kth binary variables...")
    with Timer():
        # X Binary Decision Variable: k-th body, t-th timestep, and h-th position in the sequence
        M.x_kth = pyo.Var(M.kt * M.h, within=pyo.Binary)

    ## Constraints
    print("...create X_kth packing constraints...")
    with Timer():
        # X partition constraint: do h_tot
        M.x_partition_con = pyo.Constraint(rule=pyo.summation(M.x_kth) == h_tot)

    print("...create X_kt* packing constraints...")
    with Timer():
        # for each (k,t), it can only exist in one position at most
        M.x_kt_packing_con = pyo.ConstraintList()
        for kt in tqdm(M.kt):
            M.x_kt_packing_con.add(pyo.quicksum(M.x_kth[kt, :]) <= 1)

    print("...create X_**h packing constraints...")
    with Timer():
        # for each h, there can only be one body at some time at most
        M.x_h_packing_con = pyo.Constraint(
            M.h, rule=lambda model, h: pyo.quicksum(model.x_kth[kt, h] for kt in model.kt) <= 1
        )

    print("...create X_k** max number of scientific flybys...")
    with Timer():
        # for each k, there can only be a total of N_k flybys counted
        M.NK_limit_con = pyo.Constraint(
            M.k,
            rule=lambda model, k: pyo.quicksum(
                pyo.quicksum(model.x_kth[k, t, h] for h in model.h) for t in range(model.t_num[k])
            )
            <= Nk_lim,
        )

    print("...create X_*th monotonic time constraints...")
    with Timer():
        # for each h, the time must be greater than the previous h
        def th_expr(model, position):
            # position h in sequence
            return pyo.quicksum(model.tfs[kt] * model.x_kth[kt, position] for kt in model.kt)

        M.h_time_con = pyo.ConstraintList()
        for h in M.h:
            if h > 1:
                M.h_time_con.add(th_expr(M, h) >= th_expr(M, h - 1) + DAY)

    print("...create Y_kij indicator variables of previous flybys...")
    with Timer():
        # Y Binary Indicator Variable: k-th body, i-th timestep, j-th previous timestep
        M.y_kij = pyo.Var(M.kij, within=pyo.Binary)

    print("...create Y_k** packing constraints...")
    with Timer():
        M.y_kij_packing_con = pyo.ConstraintList()
        # for k in M.k:
        #     M.y_kij_packing_con.add(pyo.quicksum(M.y_kij[k, ...]) <= factorial(h_tot - 1))
        for kj in M.kj:
            k, j = kj
            M.y_kij_packing_con.add(pyo.quicksum(M.y_kij[k, j + 1, :]) <= h_tot - 1)

    print("...create Y_kij big-M constraints...")
    with Timer():
        M.y_kij_ind_con = pyo.ConstraintList()
        for kij in tqdm(M.kij):
            k, i, j = kij
            y_kij_lhs = pyo.quicksum(M.x_kth[k, i, h] + M.x_kth[k, j, h] for h in M.h)
            M.y_kij_ind_con.add(y_kij_lhs <= 3 * M.y_kij[kij] + 1)
            M.y_kij_ind_con.add(y_kij_lhs >= 2 - 3 * (1 - M.y_kij[kij]))

    print("...create Z_kj indicator variables of first flyby...")
    with Timer():
        # Z Binary Indicator Variable: k-th body, j-th first flyby timestep
        M.z_kj = pyo.Var(M.kj, within=pyo.Binary)

    print("...create Z_k* packing constraints...")
    with Timer():
        # if there are flybys for k, then there can only be ONE first flyby
        M.z_kj_packing_con = pyo.ConstraintList()
        for k in M.k:
            M.z_kj_packing_con.add(pyo.quicksum(M.z_kj[k, :]) <= 1)

    print("...create Z_kj big-M constraints...")
    with Timer():
        M.z_kj_ind_con = pyo.ConstraintList()
        for kj in tqdm(M.kj):
            k, j = kj
            # z_kj_lhs = pyo.quicksum(M.x_kth[k, ...]) - pyo.quicksum(
            #     M.y_kij[k, :, j]
            # )  # should equal to 0 if j is the first flyby index, otherwise greater than 0

            M.z_kj_ind_con.add(pyo.quicksum(M.y_kij[k, :, j]) <= h_tot * M.z_kj[kj])
            # M.z_kj_ind_con.add(pyo.quicksum(M.y_kij[k, :, j]) >= h_tot * ((M.z_kj[kj]) - 1) + 1)

            # M.z_kj_ind_con.add(z_kj_lhs <= M.t_num[k] * (1 - M.z_kj[kj]))

    print("...create (Z_kj, Y_kij, X_kth) -> S(r_kij) seasonal penalty terms...")
    with Timer():
        flyby_terms = {k: {} for k in M.k}
        for k in tqdm(M.k):
            for i in range(1, M.t_num[k]):
                expn_term = 0
                # first flyby term
                first_term = M.z_kj[k, i - 1]
                for j in range(i):
                    # subsequent penalty terms
                    dot_products = np.clip(np.dot(M.rfs[k, i], M.rfs[k, j]), -1.0, 1.0)
                    angles_deg = np.arccos(dot_products) * 180 / np.pi
                    expn_term += np.exp(-(angles_deg**2) / 50.0) * M.y_kij[k, i, j]
                penalty_term = (0.1 + 0.9 / (1 + 10 * expn_term)) * pyo.quicksum(M.x_kth[k, i, :])
                flyby_terms[k][i] = (first_term + penalty_term) * M.w_k[k]

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
        M.zp_k = pyo.Var(M.k, within=pyo.Binary)  # planets and yandi
        # M.Gp = pyo.Var(initialize=1, within=pyo.Binary)  # all planets indicator
        M.zp_ind_con = pyo.ConstraintList()
        for k in M.k:
            zp_sum_term = pyo.quicksum(
                pyo.quicksum(M.x_kth[k, t, h] for h in M.h) for t in range(M.t_num[k])
            )
            M.zp_ind_con.add(zp_sum_term <= 2 * M.t_num[k] * M.zp_k[k])
            M.zp_ind_con.add(zp_sum_term - 1 >= 2 * M.t_num[k] * (M.zp_k[k] - 1))

        # M.gt_p_ind_con = pyo.ConstraintList()
        # M.gt_p_ind_con.add(pyo.quicksum(M.zp_k[k] for k in M.k) - gt_bodies <= M.Gp * 20)
        # M.gt_p_ind_con.add(pyo.quicksum(M.zp_k[k] for k in M.k) - gt_bodies >= (M.Gp - 1) * 20)
        # GT_bonus = 1.0 + 0.3 * M.Gp
        GT_bonus = pyo.quicksum(M.zp_k[k] for k in M.k)
        # GT_bonus = 1.3

    ## Objective function
    print("...create objective function...")
    with Timer():
        M.max_score = pyo.Objective(
            rule=GT_bonus
            * pyo.quicksum(flyby_terms[k][i] for k in M.k for i in range(1, M.t_num[k])),
            sense=pyo.maximize,
        )
    # M.max = pyo.Objective(rule=pyo.quicksum(M.w_k[k] * pyo.quicksum(M.x_kth[k, ...]) for k in M.k))

    print("...total model setup time...")


print("...SET UP SOLVER...")
with Timer():
    # Solver setup
    # from solve first arc, the intial body is Planet X at To = 5 years.
    for k in M.k:
        M.x_kth[k, 0, 1].fix(0)
    M.x_kth[10, 0, 1].fix(1)

    solver = pyo.SolverFactory("scip", solver_io="nl")


## Run
print("...RUN SOLVER...")
results = solver.solve(M, tee=True)
print(f"\n\n...iteration {0} solved...\n\n")

if results.solver.termination_condition == TerminationCondition.infeasible:
    print("Infeasible, sorry :(")
    print(f"\n...iteration {0} complete...\n")
else:
    sequence = []
    for kt in M.kt:
        k, t = kt
        for h in M.h:
            value = pyo.value(M.x_kth[kt, h])
            if value > 0.5:
                sequence.append((bodies_data[k].name, discrete_rs[k]["ts"][t] / SPTU))

    pprint(sorted(sequence, key=lambda x: x[1]))
    print("Number of lambert arcs: ", int(pyo.value(pyo.summation(M.L_kimj))))
    print("Number of repeated flybys: ", int(pyo.value(pyo.summation(M.y_kij))))
    print("Flyby keys:\n")
    for k, v in M.y_kij.items():
        if pyo.value(v) > 0.5:
            print(k)
    print("First flyby keys:\n")
    for k, v in M.z_kj.items():
        if pyo.value(v) > 0.5:
            print(k)
    print(f"\n...iteration {0} complete...\n")
