"""
Binary Integer Non-Linear Programming problem for finding the most valuable
sequences of a given length h for a starting and end year, Yo and Yf.

This file creates the Pyomo model to be paired with a MINLP/CIP/global solver
from APML.

NOTE: users need to update their environment according to the pyproject.toml
and run `conda install scip`.

Indices:
-----------
k :
t, i, j :
h :


Variables:
-----------
x[k,t,h] : primary binary variable to select body k at timestep t for position h
y[k,i,j] : binary variable for body k flyby at timestep i with a previous flyby at timestep j
z[k,t] : binary variable for FIRST body k flyby at timestep t
L[k,i,m,j] : binary variable indicating a trajectory arc between bodies k, m at timesteps i, j
planet_visited[k] : binary variable indicating if a planet k has been visited
Gp : binary variable indicating threshold of planets visited

"""

from binlp_utils import timer, IndexParams, DiscreteDict
import pyomo.environ as pyo
from tqdm import tqdm


@timer
def initialize_model(index_params: IndexParams, discrete_data: DiscreteDict) -> pyo.ConcreteModel:
    k_body = index_params.bodies_ID
    num = index_params.n_timesteps
    h_tot = index_params.seq_length
    bodies = discrete_data.bodies
    dv_table = discrete_data.dv_table

    print(">>>>> INSTANTIATE PYOMO CONCRETE MODEL >>>>>\n")
    seq_model = pyo.ConcreteModel()  # ConcreteModel instantiates during construction
    seq_model.name = "SequenceSearch"
    print("...create indices and parameters...")
    seq_model.K = pyo.Set(initialize=k_body)  # body index
    seq_model.T = pyo.RangeSet(num)  # timestep index, NOT SAME TIMESTEP
    seq_model.H = pyo.RangeSet(h_tot)  # sequence position index
    seq_model.w_k = pyo.Param(
        seq_model.K, initialize=lambda model, k: bodies[k].weight, within=pyo.PositiveReals
    )  # scoring weights

    # Cartesian product sets
    seq_model.KT = pyo.Set(initialize=[(k, t) for k in seq_model.K for t in seq_model.T])
    seq_model.KIJ = pyo.Set(
        initialize=[(k, i, j) for k in seq_model.K for i in seq_model.T for j in range(1, i)]
    )
    seq_model.KIMJ = pyo.Set(initialize=list(dv_table.keys()))

    # Parameters requiring the Cartesian product sets
    seq_model.tu_kt = pyo.Param(
        seq_model.KT,
        initialize=lambda model, k, t: bodies[k].t_tu[t - 1].tolist(),
        within=pyo.NonNegativeReals,
    )
    seq_model.rdu_kt = pyo.Param(
        seq_model.KT,
        initialize=lambda model, k, t: bodies[k].r_du[t - 1],
        within=pyo.Any,
    )

    # Delta-vs and tofs
    seq_model.tof_kimj = pyo.Param(
        seq_model.KIMJ, initialize=lambda model, k, i, m, j: dv_table[k, i, m, j]["tof"]
    )
    seq_model.dv1_kimj = pyo.Param(
        seq_model.KIMJ,
        initialize=lambda model, k, i, m, j: dv_table[k, i, m, j]["dv1"],
        within=pyo.Any,
    )
    seq_model.dv2_kimj = pyo.Param(
        seq_model.KIMJ,
        initialize=lambda model, k, i, m, j: dv_table[k, i, m, j]["dv1"],
        within=pyo.Any,
    )

    ### Print out important details ###
    return seq_model


@timer
def create_x_variables_and_constraints(seq_model: pyo.ConcreteModel):
    pass


if __name__ == "__main__":
    from gtoc13 import YEAR, SPTU, KMPDU, bodies_data
    import numpy as np
    from itertools import product
    from binlp_utils import DisBody

    np.set_printoptions(legacy="1.25")
    ############### CONFIG ###############
    Yo = 0
    Yf = 5
    perYear = 2
    dv_limit = 2000
    h_tot = 3  # sequence length
    gt_bodies = 3
    gt_smalls = 13
    Nk_lim = 3  # 13
    ############# END CONFIG #############
    print(">>>>> CREATE DISCRETIZED DATASET <<<<<\n")
    #############
    To = Yo * YEAR
    Tf = Yf * YEAR  # years in seconds
    num = int(np.ceil((Tf - To) / (YEAR / perYear)))
    dv_limit *= SPTU.tolist() / KMPDU  # km/s

    ## Generate position tables for just the bodies
    print("...discretizing body data...")
    dis_ephm = DiscreteDict()
    k_body = []
    for b_idx, body in tqdm(bodies_data.items()):
        if body.is_planet() or body.name == "Yandi":
            k_body.append(b_idx)
            timestep = np.linspace(To, Tf, num) / SPTU
            dis_ephm.bodies[b_idx] = DisBody(
                weight=body.weight,
                r_du=np.array([body.get_state(timestep[idx]).r / KMPDU for idx in range(num)]),
                v_dtu=np.array(
                    [body.get_state(timestep[idx]).v * SPTU / KMPDU for idx in range(num)]
                ),
                t_tu=timestep,
            )

    print("...making delta-v table...")
    dv_table = dict()
    for kimj in tqdm(
        [
            (k, i, m, j)
            for (k, m) in list(product(k_body, repeat=2))
            for i, __ in enumerate(dis_ephm.bodies[k].t_tu)
            for j, __ in enumerate(dis_ephm.bodies[m].t_tu)
        ]
    ):
        k, i, m, j = kimj
        tu_i = dis_ephm.bodies[k].t_tu[i]
        tu_j = dis_ephm.bodies[m].t_tu[j]
        tof = (tu_j - tu_i).tolist()
        dv_1, dv_2 = np.random.rand(2) * (dv_limit + 50.0)
        dv_table[(k, i + 1, m, j + 1)] = {"tof": tof, "dv1": dv_1, "dv2": dv_2}
    dis_ephm.dv_table = dv_table
    idx_params = IndexParams(bodies_ID=k_body, n_timesteps=num, seq_length=h_tot)
    print(">>>>> DISCRETIZED DATASET GENERATED <<<<<\n\n")
    sequence = initialize_model(index_params=idx_params, discrete_data=dis_ephm)
