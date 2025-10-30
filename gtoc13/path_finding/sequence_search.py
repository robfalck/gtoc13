"""
BINLP to roughly find the most "valuable" sequences for scientific flybys.
Assisting flybys not counted.

Example
-------

"""

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import numpy as np
from pathlib import Path

from gtoc13 import KMPDU, SPTU, YPTU, YEAR, MU_ALTAIRA, bodies_data, lambert_tof, lambert_v, Body

from tqdm import tqdm
import time


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
Tf = 15 * YEAR  # years in seconds
Nk_lim = 5  # 13

for b_idx, body in bodies_data.items():
    if body.is_planet() or body.name == "Yandi":
        TP = orbital_period(body)
        if TP / 2 > YEAR:
            num = int(np.ceil((Tf - To) / (YEAR / 2)))
        elif TP * 2 > YEAR:
            num = int(np.ceil((Tf - To) / (YEAR / 4)))
        else:
            num = int(np.ceil((Tf - To) / (TP * 4)))
        # num = int(np.ceil(Tf / (YEAR / 4))) if YEAR / 4 < TP else int(np.ceil(Tf / (TP)))
        ts = np.linspace(To, Tf, num)
        discrete_rs[b_idx] = dict(
            TP=TP,
            rf=np.array([body.get_state(ts[idx]).r / KMPDU for idx in range(num)]),
            ts=ts,
            t_k=num,
        )

## Generate decision variables
h_tot = 6  # sequence length
b_idxs = discrete_rs.keys()

print("...initialize model parameters...")
with Timer():
    m = pyo.ConcreteModel()
    m.k = pyo.Set(initialize=b_idxs)
    m.h = pyo.RangeSet(1, h_tot)

    kt_idxs = [(b_idx, t_idx) for b_idx in b_idxs for t_idx in range(discrete_rs[b_idx]["t_k"])]
    kj_idxs = [(b_idx, t_idx) for b_idx in b_idxs for t_idx in range(discrete_rs[b_idx]["t_k"] - 1)]
    kij_idxs = [
        (b_idx, i_idx, j_idx)
        for b_idx in b_idxs
        for i_idx in range(discrete_rs[b_idx]["t_k"])
        for j_idx in range(i_idx)
    ]

    def ts_rule(model, i, j):
        return discrete_rs[i]["ts"][j]

    def ts_idx(model, i, j):
        return j + 1

    def rf_rule(model, i, j):
        return discrete_rs[i]["rf"][j]

    m.tfs = pyo.Param(kt_idxs, initialize=ts_rule, within=pyo.PositiveReals)
    m.t_idxs = pyo.Param(kt_idxs, initialize=ts_idx, within=pyo.NonNegativeIntegers)
    m.rfs = pyo.Param(kt_idxs, initialize=rf_rule, within=pyo.Any)

print("...create x binary variables...")
with Timer():
    # X Binary Decision Variable: k-th body, t-th timestep, and h-th position in the sequence
    m.x_kth = pyo.Var(kt_idxs * m.h, within=pyo.Binary)

## Constraints
print("...create x packing constraints...")
with Timer():
    # X packing constraint: up to h_tot
    m.x_packing_con = pyo.Constraint(rule=pyo.summation(m.x_kth) <= h_tot)

# TODO: h in order (1,2,3,4,... not 1, 2, 5, 6,...) ...doesn't matter.

print("...create x_kt packing constraints...")
with Timer():
    # for each (k,t), it can only exist in one position at most
    m.kt_packing_con = pyo.ConstraintList()

    def kt_pack_rule(model, body, timestep):
        return pyo.quicksum(model.x_kth[body, timestep, :])

    for kt in tqdm(kt_idxs):
        m.kt_packing_con.add(kt_pack_rule(m, *kt) <= 1)

print("...create x_h packing constraints...")
with Timer():
    # for each h, there can only be one body at some time at most
    m.h_packing_con = pyo.ConstraintList()

    for h in m.h:
        m.h_packing_con.add(pyo.quicksum(m.x_kth[kt, h] for kt in kt_idxs) <= 1)

print("...create x_k max number of scientific flybys...")
with Timer():
    # for each k, there can only be a total of N_k flybys counted
    m.Nk_limit_con = pyo.ConstraintList()
    for k in m.k:
        num = discrete_rs[k]["t_k"]
        m.Nk_limit_con.add(
            pyo.quicksum(
                pyo.quicksum(
                    m.x_kth[
                        k,
                        t,
                        h,
                    ]
                    for h in m.h
                )
                for t in range(num)
            )
            <= Nk_lim
        )

print("...create h monotonic time constraints...")
with Timer():
    # for each h, the time must be greater than the previous h
    m.h_time_con = pyo.ConstraintList()

    times = {}
    for h in m.h:
        times[h] = pyo.quicksum(m.tfs[kt] * m.x_kth[kt, h] for kt in kt_idxs)
        if h > 1:
            m.h_time_con.add(times[h] >= times[h - 1])

print("...create y indicator variables...")
with Timer():
    # Y Binary Indicator Variable: k-th body, i-th timestep, j-th previous timestep
    # toggle big-M constraints
    m.y_kij0 = pyo.Var(kij_idxs, within=pyo.Binary)
    m.y_kij1 = pyo.Var(kij_idxs, within=pyo.Binary)

print("...create y partition constraints...")
with Timer():
    m.y_partition_con = pyo.ConstraintList()
    for kij in kij_idxs:
        m.y_partition_con.add(m.y_kij0[kij] + m.y_kij1[kij] == 1)

print("...create y big-M constraints...")
with Timer():
    m.y_ind_con = pyo.ConstraintList()
    for kij in tqdm(kij_idxs):
        k, i, j = kij
        y_sum_term = pyo.quicksum(m.x_kth[k, i, :]) + pyo.quicksum(m.x_kth[k, j, :])
        m.y_ind_con.add(5 * (1 - m.y_kij1[kij]) <= y_sum_term)
        m.y_ind_con.add(5 * (m.y_kij0[kij] + m.y_kij1[kij]) >= y_sum_term)

print("...create z indicator variables...")
with Timer():
    # Z Binary Indicator Variable: k-th body, j-th first flyby timestep
    m.z_kj0 = pyo.Var(kj_idxs, within=pyo.Binary)
    m.z_kj1 = pyo.Var(kj_idxs, within=pyo.Binary)

print("...create z partition constraints...")
# for every kj, either z_kj0 or z_kj1 must be 1
with Timer():
    m.z_partition_con = pyo.ConstraintList()
    for kj in kj_idxs:
        m.z_partition_con.add(m.z_kj0[kj] + m.z_kj1[kj] == 1)

print("...create z packing constraints...")
with Timer():
    # if there are flybys for k, then there can only be ONE first flyby
    m.z_packing_con = pyo.ConstraintList()
    for k in m.k:
        m.z_packing_con.add(pyo.quicksum(m.z_kj1[k, :]) <= 1)

print("...create z big-M constraints...")
with Timer():
    m.z_ind_con = pyo.ConstraintList()
    for kj in tqdm(kj_idxs):
        k, j = kj
        M = discrete_rs[k]["t_k"]
        z_sum_term = (
            pyo.quicksum(pyo.quicksum(m.x_kth[k, t, h] for h in m.h) for t in range(M))
            - pyo.quicksum(m.y_kij1[k, :, j])
            - 1
        )
        m.z_ind_con.add(z_sum_term <= M * (1 - m.z_kj1[kj]))
        m.z_ind_con.add(z_sum_term <= M * (m.z_kj0[kj] + m.z_kj1[kj]))


print("...create seasonal penalty terms constraints...")
with Timer():
    flyby_terms = {k: {} for k in m.k}
    for k in m.k:
        weight_k = bodies_data[k].weight
        num = discrete_rs[k]["t_k"]
        for i in range(1, num):
            expn_term = 0
            # first flyby term
            first_term = m.z_kj1[k, i - 1]
            for j in range(i):
                # subsequent penalty terms
                dot_products = np.clip(np.dot(m.rfs[k, i], m.rfs[k, j]), -1.0, 1.0)
                angles_deg = np.arccos(dot_products) * 180 / np.pi
                expn_term += np.exp(-(angles_deg**2) / 50.0) * m.y_kij1[k, i, j]
            penalty_term = (0.1 + 0.9 / (1 + 10 * expn_term)) * pyo.quicksum(
                m.x_kth[k, i, h] for h in m.h
            )
            flyby_terms[k][i] = (first_term + penalty_term) * weight_k


## Objective function
print("...create objective function...")
with Timer():
    m.max_score = pyo.Objective(
        rule=pyo.quicksum(flyby_terms[k][i] for k in m.k for i in range(1, discrete_rs[k]["t_k"])),
        sense=pyo.maximize,
    )


# Solver setup
# from solve first arc, the intial body is Planet X at To = 5 years.
m.x_kth[10, 0, 1].fix(1)

solver = pyo.SolverFactory("scip", solver_io="nl")
results = solver.solve(m, tee=True)
print(f"\n\n...iteration {0} solved...\n\n")

if results.solver.termination_condition == TerminationCondition.infeasible:
    print("Infeasible, sorry :(")
    print(f"\n\n...iteration {0} complete...\n\n")
else:
    sequence = []
    for kt in kt_idxs:
        k, t = kt
        for h in m.h:
            value = pyo.value(m.x_kth[kt, h])
            if value > 0:
                sequence.append((bodies_data[k].name, discrete_rs[k]["ts"][t] / SPTU))

    print(sorted(sequence, key=lambda x: x[1]))
    print(f"\n\n...iteration {0} complete...\n\n")

    m.cuts_con = pyo.ConstraintList()
    for sol in range(3):
        expr = 0
        for x_idx in m.x_kth:
            if pyo.value(m.x_kth[x_idx]) < 0.5:
                expr += m.x_kth[x_idx]
            else:
                expr += 1 - m.x_kth[x_idx]
        m.cuts_con.add(expr >= 1)
        results = solver.solve(m, tee=True)
        print(f"\n\n...iteration {sol+1} solved...\n\n")
        if results.solver.termination_condition == TerminationCondition.infeasible:
            print("Infeasible, sorry :(")
            print(f"\n\n...iteration {sol+1} complete...\n\n")
            break
        else:
            sequence = []
            for kt in kt_idxs:
                k, t = kt
                for h in m.h:
                    value = pyo.value(m.x_kth[kt, h])
                    if value > 0:
                        sequence.append((bodies_data[k].name, discrete_rs[k]["ts"][t] / SPTU))
            print(sorted(sequence, key=lambda x: x[1]))
            print(f"\n\n...iteration {sol+1} complete...\n\n")
