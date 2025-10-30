"""
Description

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

for b_idx, body in bodies_data.items():
    if body.is_planet() or body.name == "Yandi":
        TP = orbital_period(body)
        if TP / 2 > YEAR:
            num = int(np.ceil((Tf - To) / (YEAR / 2)))
        elif TP * 2 > YEAR:
            num = int(np.ceil((Tf - To) / (YEAR / 4)))
        else:
            num = int(np.ceil((Tf - To) / (TP)))
        # num = int(np.ceil(Tf / (YEAR / 4))) if YEAR / 4 < TP else int(np.ceil(Tf / (TP)))
        ts = np.linspace(To, Tf, num)
        discrete_rs[b_idx] = dict(
            TP=TP,
            rf=np.array([body.get_state(ts[idx]).r / KMPDU for idx in range(num)]),
            ts=ts,
            t_k=num,
        )

## Generate decision variables
h_tot = 3  # sequence length
b_idxs = discrete_rs.keys()

print("...initialize model parameters...")
with Timer():
    m = pyo.ConcreteModel()
    m.k = pyo.Set(initialize=b_idxs)
    m.h = pyo.RangeSet(1, h_tot)

    kt_idxs = [(b_idx, t_idx) for b_idx in b_idxs for t_idx in range(discrete_rs[b_idx]["t_k"])]
    kij_idxs = [
        (b_idx, i_idx, j_idx)
        for b_idx in b_idxs
        for i_idx in range(discrete_rs[b_idx]["t_k"])
        for j_idx in range(i_idx)
    ]

    def ts_rule(model, i, j):
        return discrete_rs[i]["ts"][j]

    def rf_rule(model, i, j):
        return discrete_rs[i]["rf"][j]

    m.ts = pyo.Param(kt_idxs, initialize=ts_rule, within=pyo.PositiveReals)
    m.rfs = pyo.Param(kt_idxs, initialize=rf_rule, within=pyo.Any)

print("...create x binary variables...")
with Timer():
    # X Binary Decision Variable: k-th body, t-th timestep, and h-th position in the sequence
    m.x_kth = pyo.Var(kt_idxs * m.h, within=pyo.Binary)

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

print("...create y partition constraints...")
with Timer():
    m.y_ind_con = pyo.ConstraintList()
    for kij in tqdm(kij_idxs):
        k, i, j = kij
        m.y_ind_con.add(
            5 * (1 - m.y_kij1[kij])
            <= pyo.quicksum(m.x_kth[k, i, :]) + pyo.quicksum(m.x_kth[k, j, :])
        )
        m.y_ind_con.add(
            5 * (m.y_kij0[kij] + m.y_kij1[kij])
            >= pyo.quicksum(m.x_kth[k, i, :]) + pyo.quicksum(m.x_kth[k, j, :])
        )


## Constraints
print("...create x packing constraints...")
with Timer():
    # X packing constraint: up to h_tot
    m.x_packing_con = pyo.Constraint(rule=pyo.summation(m.x_kth) <= h_tot)

# TODO: h in order (1,2,3,4,... not 1, 2, 5, 6,...) ...doesn't matter.

print("...create kt packing constraints...")
with Timer():
    # for each (k,t), it can only exist in one position at most
    m.kt_packing_con = pyo.ConstraintList()

    def kt_pack_rule(model, body, timestep):
        return pyo.quicksum(model.x_kth[body, timestep, :])

    for kt in kt_idxs:
        m.kt_packing_con.add(kt_pack_rule(m, *kt) <= 1)

print("...create h packing constraints...")
with Timer():
    # for each h, there can only be one body at some time at most
    m.h_packing_con = pyo.ConstraintList()

    for h in m.h:
        m.h_packing_con.add(pyo.quicksum(m.x_kth[kt, h] for kt in kt_idxs) <= 1)

print("...create h monotonic time constraints...")
with Timer():
    # for each h, the time must be greater than the previous h
    m.h_time_con = pyo.ConstraintList()

    times = {}
    for h in m.h:
        times[h] = pyo.quicksum(m.ts[kt] * m.x_kth[kt, h] for kt in kt_idxs)
        if h > 1:
            m.h_time_con.add(times[h] >= times[h - 1])

print("...create seasonal penalty terms constraints...")
with Timer():
    # Seasonal penalty; add first term as 1.
    flyby_terms = {k: {} for k in m.k}
    old_k = 0
    old_i = 0
    for kij in kij_idxs:
        k, i, j = kij
        if old_k != k or old_i != i:
            flyby_terms[k][i] = []
        dot_prods = np.clip(np.dot(m.rfs[k, i], m.rfs[k, j]), -1.0, 1.0)
        angles_deg = np.arccos(dot_prods) * 180.0 / np.pi
        exp_term = np.exp(-(angles_deg**2) / 50.0) * m.y_kij1[kij]

        flyby_terms[k][i].append(exp_term)
        old_k = k
        old_i = i

    flybys = {}
    for k in flyby_terms:
        for i in flyby_terms[k]:
            flybys[k, i] = (
                (0.1 + 0.9 / (sum(flyby_terms[k][i]) * 10 + 1))
                * pyo.quicksum(m.x_kth[k, i, :])
                * bodies_data[k].weight
            )

## Objective function
with Timer():
    m.max_score = pyo.Objective(rule=pyo.quicksum(flybys[k] for k in flybys), sense=pyo.maximize)


# Solver setup
# from solve first arc, the intial body is Planet X at To = 5 years.
m.x_kth[10, 0, 1].fix(1)

solver = pyo.SolverFactory("scip", solver_io="nl")
results = solver.solve(m, tee=True)

if results.solver.termination_condition == TerminationCondition.infeasible:
    print("Infeasible, sorry :(")
else:
    sequence = []
    for kt in kt_idxs:
        k, t = kt
        for h in m.h:
            value = pyo.value(m.x_kth[kt, h])
            if value > 0:
                sequence.append((k, discrete_rs[k]["ts"][t] / SPTU))
