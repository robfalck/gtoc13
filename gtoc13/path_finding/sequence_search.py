"""
Description

Example
-------

"""

import pyomo.environ as pyo
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from gtoc13 import KMPDU, SPTU, YPTU, YEAR, MU_ALTAIRA, bodies_data, lambert_tof, lambert_v, Body


def orbital_period(body: Body):
    """
    Returns orbital period in seconds.
    """
    return 2 * jnp.pi * (body.elements.a) ** (3 / 2) / jnp.sqrt(MU_ALTAIRA)


## Generate position tables for just the bodies
discrete_rs = dict()
To = 0.5 * YEAR
Tf = 6 * YEAR  # years in seconds

for b_idx, body in bodies_data.items():
    if body.is_planet() or body.name == "Yandi":
        TP = orbital_period(body)
        num = int(jnp.ceil(Tf / (YEAR / 4))) if YEAR / 4 < TP / 3 else int(jnp.ceil(Tf / (TP)))
        ts = jnp.linspace(To, Tf, num)
        discrete_rs[b_idx] = dict(
            rf=jnp.array([body.get_state(ts[idx]).r / KMPDU for idx in range(num)]),
            ts=ts,
            t_k=num,
        )

## Generate decision variables
h_tot = 5  # sequence length
b_idxs = discrete_rs.keys()

print("...initialize model...")
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


m.ts = pyo.Param(kt_idxs, initialize=ts_rule)
m.rfs = pyo.Param(kt_idxs, initialize=rf_rule)

print("...create x and y binary variables...")
# X Binary Decision Variable: k-th body, t-th timestep, and h-th position in the sequence
m.x_kth = pyo.Var(kt_idxs * m.h, within=pyo.Binary, initialize=0)

# Y Binary Indicator Variable: k-th body, i-th timestep, j-th previous timestep
# toggle big-M constraints
m.y_kij0 = pyo.Var(kij_idxs, within=pyo.Binary, initialize=0)
m.y_kij1 = pyo.Var(kij_idxs, within=pyo.Binary, initialize=0)

print("...create y partition constraints...")
m.y_partition_con = pyo.ConstraintList()
for kij in kij_idxs:
    m.y_partition_con.add(m.y_kij0[kij] + m.y_kij1[kij] == 1)
print("...create y partition constraints...")
m.y_ind_con = pyo.ConstraintList()
for kij in kij_idxs:
    print(f"...{kij} y indicator constraints...")
    k, i, j = kij
    m.y_ind_con.add(
        5 * (1 - m.y_kij1[kij]) <= pyo.quicksum(m.x_kth[k, i, :]) + pyo.quicksum(m.x_kth[k, j, :])
    )
    m.y_ind_con.add(
        5 * (m.y_kij0[kij] + m.y_kij1[kij])
        >= pyo.quicksum(m.x_kth[k, i, :]) + pyo.quicksum(m.x_kth[k, j, :])
    )


## Constraints
print("...create x packing constraints...")
# X packing constraint: up to h_tot
m.x_packing_con = pyo.Constraint(rule=pyo.summation(m.x_kth) <= h_tot)

# TODO: h in order (1,2,3,4,... not 1, 2, 5, 6,...) ...doesn't matter.

print("...create h packing constraints...")
# for each (k,t), it can only exist in one position at most
m.h_packing_con = pyo.ConstraintList()


def h_pack_rule(model, body, timestep):
    return pyo.quicksum(model.x_kth[body, timestep, :])


for kt in kt_idxs:
    m.h_packing_con.add(h_pack_rule(m, *kt) <= 1)

print("...create h monotonic time constraints...")
# for each h, the time must be greater than the previous h
m.h_time_con = pyo.ConstraintList()

times = {}
for h in m.h:
    times[h] = pyo.quicksum(m.ts[kt] * m.x_kth[kt, h] for kt in kt_idxs)
    if h > 1:
        m.h_time_con.add(times[h] >= times[h - 1])

print("...create seasonal penalty terms constraints...")
# Seasonal penalty; add first term as 1.
flyby_terms = {k: {} for k in m.k}
old_k = 0
old_i = 0
for kij in kij_idxs:
    k, i, j = kij
    print(f"...{kij} seasonal terms...")
    if old_k != k or old_i != i:
        flyby_terms[k][i] = [0]
    dot_prods = jnp.clip(jnp.dot(m.rfs[k, i], m.rfs[k, j]), -1.0, 1.0)
    angles_deg = jnp.arccos(dot_prods) * 180.0 / jnp.pi
    exp_term = jnp.exp(-(angles_deg**2) / 50.0) * m.y_kij1[kij]

    flyby_terms[k][i].append(exp_term)
    old_k = k
    old_i = i

flybys = {}
for k in flyby_terms:
    for i in flyby_terms[k]:
        flybys[k, i] = (0.1 + 0.9 / (sum(flyby_terms[k][i]) * 10 + 1)) * pyo.quicksum(
            m.x_kth[k, i, :]
        )
