"""
BINLP to roughly find the most "valuable" sequences for scientific flybys.
Assisting flybys not counted.
Assumes a starting year, first arc solver should determine feasibility of the starting body.
- Grand tour bonus scaling rather 1.0 or 1.3 multiplier.

NOTE: users need to update their environment according to the pyproject.toml and install 'scip' via conda.
TODO: check what else is implicitly overcoming the "up to h_total" constraint

Example
-------

"""

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import numpy as np

# from pathlib import Path

from gtoc13 import DAY, KMPDU, SPTU, YEAR, MU_ALTAIRA, bodies_data, Body, lambert_delta_v

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


def orbital_period(body: Body):
    """
    Returns orbital period in seconds.
    """
    return 2 * np.pi * (body.elements.a) ** (3 / 2) / np.sqrt(MU_ALTAIRA)


## Generate position tables for just the bodies
discrete_rs = dict()
To = 5 * YEAR
Tf = 10 * YEAR  # years in seconds
Nk_lim = 3  # 13
dv_lim = 10000  # km/s
gt_bodies = 11
h_tot = 12  # sequence length
b_idxs = discrete_rs.keys()

## Should tune the granularity of the discretization
for b_idx, body in bodies_data.items():
    if body.is_planet() or body.name == "Yandi":
        TP = orbital_period(body)
        if TP / 2 > YEAR:
            num = int(np.ceil((Tf - To) / (YEAR / 3)))
        elif TP * 2 > YEAR:
            num = int(np.ceil((Tf - To) / (YEAR / 6)))
        else:
            num = int(np.ceil((Tf - To) / (TP * 18)))
        # num = int(np.ceil(Tf / (YEAR / 4))) if YEAR / 4 < TP else int(np.ceil(Tf / (TP)))
        ts = np.linspace(To, Tf, num)
        discrete_rs[b_idx] = dict(
            TP=TP,
            rf=np.array([body.get_state(ts[idx]).r / KMPDU for idx in range(num)]),
            ts=ts,
            t_k=num,
        )

#### GENERATE LAMBERT TABLES ####
dv_i_dict = {}
dv_f_dict = {}

print("...generating lambert tables...")
with Timer():
    lambert_from_to = list(product(b_idxs, repeat=2))
    for from_to in tqdm(lambert_from_to):
        k_i, k_f = from_to
        for ti_i, ts_i in enumerate(discrete_rs[k_i]["ts"]):
            for ti_f, ts_f in enumerate(discrete_rs[k_f]["ts"]):
                if ts_f <= ts_i:
                    dv_i_dict[(k_i, ti_i, k_f, ti_f)] = -10000
                    dv_f_dict[(k_i, ti_i, k_f, ti_f)] = -10000
                    continue
                else:
                    dv_i, dv_f, conv = lambert_delta_v(
                        k_i, k_f, ts_i, ts_f, bodies_data=bodies_data
                    )
                    dv_i_dict[(k_i, ti_i, k_f, ti_f)] = dv_i if conv else 10e16
                    dv_f_dict[(k_i, ti_i, k_f, ti_f)] = dv_f if conv else 10e16


## Generate decision variables
print("...initialize model parameters...")
with Timer():
    M = pyo.ConcreteModel()
    M.k = pyo.Set(initialize=b_idxs)
    M.h = pyo.RangeSet(1, h_tot)

    kt_idxs = [(b_idx, t_idx) for b_idx in b_idxs for t_idx in range(discrete_rs[b_idx]["t_k"])]
    kj_idxs = [(b_idx, t_idx) for b_idx in b_idxs for t_idx in range(discrete_rs[b_idx]["t_k"] - 1)]
    kij_idxs = [
        (b_idx, i_idx, j_idx)
        for b_idx in b_idxs
        for i_idx in range(discrete_rs[b_idx]["t_k"])
        for j_idx in range(i_idx)
    ]

    def ks_rule(model, i, j):
        return i

    def ts_rule(model, i, j):
        return discrete_rs[i]["ts"][j]

    def ts_idx(model, k, i):
        return i

    def rf_rule(model, i, j):
        return discrete_rs[i]["rf"][j]

    M.kfs = pyo.Param(kt_idxs, initialize=ks_rule, within=pyo.PositiveIntegers)
    M.tfs = pyo.Param(kt_idxs, initialize=ts_rule, within=pyo.PositiveReals)
    M.t_idxs = pyo.Param(kt_idxs, initialize=ts_idx, within=pyo.NonNegativeIntegers)
    M.rfs = pyo.Param(kt_idxs, initialize=rf_rule, within=pyo.Any)

print("...create x binary variables...")
with Timer():
    # X Binary Decision Variable: k-th body, t-th timestep, and h-th position in the sequence
    M.x_kth = pyo.Var(kt_idxs * M.h, within=pyo.Binary)

## Constraints
print("...create x packing constraints...")
with Timer():
    # X packing constraint: up to h_tot
    M.x_packing_con = pyo.Constraint(rule=pyo.summation(M.x_kth) <= h_tot)

# TODO: h in order (1,2,3,4,... not 1, 2, 5, 6,...) ...doesn't matter.

print("...create x_kt packing constraints...")
with Timer():
    # for each (k,t), it can only exist in one position at most
    def kt_pack_constraint(model, body, timestep):
        return pyo.quicksum(model.x_kth[body, timestep, :]) <= 1

    M.kt_packing_con = pyo.Constraint(kt_idxs, rule=kt_pack_constraint)

print("...create x_h packing constraints...")
with Timer():
    # for each h, there can only be one body at some time at most
    def h_pack_constraint(model, position):
        return pyo.quicksum(M.x_kth[kt, position] for kt in kt_idxs) <= 1

    M.h_packing_con = pyo.Constraint(M.h, rule=h_pack_constraint)


print("...create x_k max number of scientific flybys...")
with Timer():
    # for each k, there can only be a total of N_k flybys counted
    def NK_limit_constraint(model, body):
        t_num = discrete_rs[body]["t_k"]
        return (
            pyo.quicksum(pyo.quicksum(M.x_kth[body, t, h] for h in M.h) for t in range(t_num))
            <= Nk_lim
        )

    M.NK_limit_con = pyo.Constraint(M.k, rule=NK_limit_constraint)


def rh_expr(model, position):
    # position h in sequence
    return pyo.quicksum(model.rfs[kt] * model.x_kth[kt, position] for kt in kt_idxs)


def th_expr(model, position):
    # position h in sequence
    return pyo.quicksum(model.tfs[kt] * model.x_kth[kt, position] for kt in kt_idxs)


def kh_expr(model, position):
    # position h in sequence
    return pyo.quicksum(model.kfs[kt] * model.x_kth[kt, position] for kt in kt_idxs)


print("...create h monotonic time constraints...")
with Timer():
    # for each h, the time must be greater than the previous h
    M.h_time_con = pyo.ConstraintList()
    for h in M.h:
        if h > 1:
            M.h_time_con.add(th_expr(M, h) >= th_expr(M, h - 1) + DAY)

print("...create y indicator variables...")
with Timer():
    # Y Binary Indicator Variable: k-th body, i-th timestep, j-th previous timestep
    M.y_kij0 = pyo.Var(kij_idxs, within=pyo.Binary)
    M.y_kij1 = pyo.Var(kij_idxs, within=pyo.Binary)

print("...create y partition constraints...")
with Timer():
    M.y_partition_con = pyo.ConstraintList()
    for kij in kij_idxs:
        M.y_partition_con.add(M.y_kij0[kij] + M.y_kij1[kij] == 1)

print("...create y big-M constraints...")
with Timer():
    M.y_ind_con = pyo.ConstraintList()
    for kij in tqdm(kij_idxs):
        k, i, j = kij
        y_sum_term = pyo.quicksum(M.x_kth[k, i, :]) + pyo.quicksum(M.x_kth[k, j, :])
        M.y_ind_con.add(5 * (1 - M.y_kij1[kij]) <= y_sum_term)
        M.y_ind_con.add(5 * (M.y_kij0[kij] + M.y_kij1[kij]) >= y_sum_term)

print("...create z indicator variables...")
with Timer():
    # Z Binary Indicator Variable: k-th body, j-th first flyby timestep
    M.z_kj0 = pyo.Var(kj_idxs, within=pyo.Binary)
    M.z_kj1 = pyo.Var(kj_idxs, within=pyo.Binary)

print("...create z partition constraints...")
# for every kj, either z_kj0 or z_kj1 must be 1
with Timer():
    M.z_partition_con = pyo.ConstraintList()
    for kj in kj_idxs:
        M.z_partition_con.add(M.z_kj0[kj] + M.z_kj1[kj] == 1)

print("...create z packing constraints...")
with Timer():
    # if there are flybys for k, then there can only be ONE first flyby
    M.z_packing_con = pyo.ConstraintList()
    for k in M.k:
        M.z_packing_con.add(pyo.quicksum(M.z_kj1[k, :]) <= 1)

print("...create z big-M constraints...")
with Timer():
    M.z_ind_con = pyo.ConstraintList()
    for kj in tqdm(kj_idxs):
        k, j = kj
        t_num = discrete_rs[k]["t_k"]
        z_sum_term = (
            pyo.quicksum(pyo.quicksum(M.x_kth[k, t, h] for h in M.h) for t in range(t_num))
            - pyo.quicksum(M.y_kij1[k, :, j])
            - 1
        )
        M.z_ind_con.add(z_sum_term <= t_num * (1 - M.z_kj1[kj]))
        M.z_ind_con.add(z_sum_term <= t_num * (M.z_kj0[kj] + M.z_kj1[kj]))


print("...create seasonal penalty terms...")
with Timer():
    flyby_terms = {k: {} for k in M.k}
    for k in M.k:
        weight_k = bodies_data[k].weight
        num = discrete_rs[k]["t_k"]
        for i in range(1, num):
            expn_term = 0
            # first flyby term
            first_term = M.z_kj1[k, i - 1]
            for j in range(i):
                # subsequent penalty terms
                dot_products = np.clip(np.dot(M.rfs[k, i], M.rfs[k, j]), -1.0, 1.0)
                angles_deg = np.arccos(dot_products) * 180 / np.pi
                expn_term += np.exp(-(angles_deg**2) / 50.0) * M.y_kij1[k, i, j]
            penalty_term = (0.1 + 0.9 / (1 + 10 * expn_term)) * pyo.quicksum(
                M.x_kth[k, i, h] for h in M.h
            )
            flyby_terms[k][i] = (first_term + penalty_term) * weight_k


print("...create grand tour bonus indicator variables and big-M constraints...")
with Timer():
    M.zp_k = pyo.Var(M.k, within=pyo.Binary)  # planets and yandi
    M.Gp = pyo.Var(initialize=1, within=pyo.Binary)  # all planets indicator
    M.zp_ind_con = pyo.ConstraintList()
    for k in M.k:
        t_num = discrete_rs[k]["t_k"]
        zp_sum_term = pyo.quicksum(
            pyo.quicksum(M.x_kth[k, t, h] for h in M.h) for t in range(t_num)
        )
        M.zp_ind_con.add(zp_sum_term <= 2 * t_num * M.zp_k[k])
        M.zp_ind_con.add(zp_sum_term - 1 >= 2 * t_num * (M.zp_k[k] - 1))

    M.grandtour_ind_con = pyo.ConstraintList()
    M.grandtour_ind_con.add(pyo.quicksum(M.zp_k[k] for k in M.k) - gt_bodies <= M.Gp * 200)
    M.grandtour_ind_con.add(pyo.quicksum(M.zp_k[k] for k in M.k) - gt_bodies >= (M.Gp - 1) * 200)
    GT_bonus = 1.0 + 0.3 * M.Gp
# GT_bonus = pyo.quicksum(M.zp_k[k] for k in M.k)


#### LAMBERT VARIABLES #####
print("...create lambert table indicator variables and big-M constraints...")
with Timer():
    # bodies k and m, times i and j
    kimj_idxs = dv_i_dict.keys()
    print("...create lambert table indicator variables...")
    M.L_kimj0 = pyo.Var(kimj_idxs, within=pyo.Binary)
    M.L_kimj1 = pyo.Var(kimj_idxs, within=pyo.Binary)
    print("...create lambert variables partition constraints...")
    # toggle on or off
    M.L_partition_con = pyo.ConstraintList()
    for kimj in tqdm(kimj_idxs):
        M.L_partition_con.add(M.L_kimj0[kimj] + M.L_kimj1[kimj] == 1)

    print("...create lambert transfer partition constraints...")
    # must have lambert checks equal to sum(x_kth) - 1
    M.L_packing_con = pyo.Constraint(rule=pyo.summation(M.L_kimj1) <= pyo.summation(M.x_kth) - 1)
    print("...create lambert transfer start and end constraints...")
    # can't start from the same place more than once
    M.L_start_con = pyo.ConstraintList()
    M.L_end_con = pyo.ConstraintList()
    for kt in tqdm(kt_idxs):
        M.L_start_con.add(pyo.quicksum(M.L_kimj1[kt, ki] for ki in kt_idxs) <= 1)
        M.L_end_con.add(pyo.quicksum(M.L_kimj1[ki, kt] for ki in kt_idxs) <= 1)
    print("...create lambert indicator and delta-v constraints...")
    # toggle lambert idx
    M.L_ind_con = pyo.ConstraintList()
    M.L_dv_con = pyo.ConstraintList()
    for h in tqdm(range(2, len(M.h) + 1)):
        for kimj in tqdm(kimj_idxs):
            k, i, m, j = kimj
            M.L_ind_con.add(
                3 * (1 - M.L_kimj1[kimj]) >= 2 - M.x_kth[k, i, h - 1] - M.x_kth[m, j, h]
            )
            M.L_ind_con.add(
                3 * (M.L_kimj0[kimj] - M.L_kimj1[kimj])
                >= 2 - M.x_kth[k, i, h - 1] - M.x_kth[m, j, h]
            )
            M.L_dv_con.add(
                M.L_kimj1[kimj] * dv_i_dict[kimj] + M.L_kimj1[kimj] * dv_f_dict[kimj] <= dv_lim
            )
            M.L_dv_con.add(0 <= M.L_kimj1[kimj] * dv_i_dict[kimj])
            M.L_dv_con.add(0 <= M.L_kimj1[kimj] * dv_f_dict[kimj])


## Objective function
print("...create objective function...")
with Timer():
    M.max_score = pyo.Objective(
        rule=GT_bonus
        * pyo.quicksum(flyby_terms[k][i] for k in M.k for i in range(1, discrete_rs[k]["t_k"])),
        sense=pyo.maximize,
    )


# Solver setup
# from solve first arc, the intial body is Planet X at To = 5 years.
M.x_kth[10, 0, 1].fix(1)

solver = pyo.SolverFactory("scip", solver_io="nl")


## Run
print("...run solver...")
results = solver.solve(M, tee=True)
print(f"\n\n...iteration {0} solved...\n\n")

if results.solver.termination_condition == TerminationCondition.infeasible:
    print("Infeasible, sorry :(")
    print(f"\n\n...iteration {0} complete...\n\n")
else:
    sequence = []
    for kt in kt_idxs:
        k, t = kt
        for h in M.h:
            value = pyo.value(M.x_kth[kt, h])
            if value > 0:
                sequence.append((bodies_data[k].name, discrete_rs[k]["ts"][t] / SPTU))

    pprint(sorted(sequence, key=lambda x: x[1]))
    print(f"\n\n...iteration {0} complete...\n\n")

    # No-good cuts for multiple solutions
    # M.cuts_con = pyo.ConstraintList()
    # for sol in range(2):
    #     expr = 0
    #     for x_idx in M.x_kth:
    #         if pyo.value(M.x_kth[x_idx]) < 0.5:
    #             expr += M.x_kth[x_idx]
    #         else:
    #             expr += 1 - M.x_kth[x_idx]
    #     M.cuts_con.add(expr >= 1)

    #     print("...run solver...")
    #     results = solver.solve(m, tee=True)
    #     print(f"\n\n...iteration {sol+1} solved...\n\n")
    #     if results.solver.termination_condition == TerminationCondition.infeasible:
    #         print("Infeasible, sorry :(")
    #         print(f"\n\n...iteration {sol+1} complete...\n\n")
    #         break
    #     else:
    #         sequence = []
    #         for kt in kt_idxs:
    #             k, t = kt
    #             for h in M.h:
    #                 value = pyo.value(M.x_kth[kt, h])
    #                 if value > 0:
    #                     sequence.append((bodies_data[k].name, discrete_rs[k]["ts"][t] / SPTU))
    #         pprint(sorted(sequence, key=lambda x: x[1]))
    #         print(f"\n\n...iteration {sol+1} complete...\n\n")
