"""
Description

Example
-------

"""

import pyomo.environ as pyo
import numpy as np
from pathlib import Path

from gtoc13 import KMPDU, SPTU, YPTU, YEAR, MU_ALTAIRA, bodies_data, lambert_tof, lambert_v, Body


def orbital_period(body: Body):
    """
    Returns orbital period in seconds.
    """
    return 2 * np.pi * (body.elements.a) ** (3 / 2) / np.sqrt(MU_ALTAIRA)


## Generate position tables for just the bodies
discrete_rs = dict()
To = 0.5 * YEAR
Tf = 6 * YEAR  # years in seconds

for b_idx, body in bodies_data.items():
    if body.is_planet() or body.name == "Yandi":
        TP = orbital_period(body)
        num = int(np.ceil(Tf / (YEAR / 4))) if YEAR / 4 < TP / 3 else int(np.ceil(Tf / (TP / 3)))
        ts = np.linspace(To * YEAR, Tf, num)
        discrete_rs[b_idx] = dict(
            rf=np.array([body.get_state(ts[idx]).r / KMPDU for idx in range(num)]),
            ts=ts,
            t_k=num,
        )

## Generate decision variables
h_tot = 20  # sequence length
b_idxs = discrete_rs.keys()

m = pyo.ConcreteModel()
m.k = pyo.Set(initialize=b_idxs)
m.h = pyo.RangeSet(1, h_tot)

kt_idxs = [(b_idx, t_idx) for b_idx in b_idxs for t_idx in range(discrete_rs[b_idx]["t_k"])]


def ts_rule(model, i, j):
    return discrete_rs[i]["ts"][j]


def rf_rule(model, i, j):
    return discrete_rs[i]["rf"][j]


m.ts = pyo.Param(kt_idxs, initialize=ts_rule, within=pyo.PositiveReals)
m.rfs = pyo.Param(kt_idxs, initialize=rf_rule, within=pyo.Reals)

# X Binary Decision Variable: k-th body, t-th timestep, and h-th position in the sequence
m.x_kth = pyo.Var(kt_idxs * m.h, within=pyo.Binary)


## Constraints
# X packing constraint: up to h_tot
m.x_packing_con = pyo.Constraint(rule=pyo.summation(m.x_kth) <= h_tot)

# for each (k,t), it can only exist in one position at most
m.h_packing_con = pyo.ConstraintList()


def h_pack_rule(model, body, timestep):
    return pyo.quicksum(model.x_kth[body, timestep, :])


for kt in kt_idxs:
    m.h_packing_con.add(h_pack_rule(m, *kt) <= 1)

# for each h, the time must be greater than the previous h
m.h_time_con = pyo.ConstraintList()

times = {}
for h in m.h:
    times[h] = pyo.quicksum(m.ts[kt] * m.x_kth[kt, h] for kt in kt_idxs)
    if h > 1:
        m.h_time_con.add(times[h] >= times[h - 1])

# Seasonal penalty
flybys = {}
for k in m.k:
    ri_k = [m.rfs[k, t] * m.x_kth[k, t, h] for t in range(discrete_rs[k]["t_k"]) for h in m.h]
