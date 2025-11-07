"""
Binary Integer Non-Linear Programming problem for finding the most valuable
sequences of a given length h for a starting and end year, Yo and Yf.

This file creates the Pyomo model to be paired with a MINLP/CIP/global solver
from APML.

NOTE: users need to update their environment according to the pyproject.toml
and run `conda install scip`.

"""

from gtoc13 import DAY, SPTU
from b_utils import timer, IndexParams, DiscreteDict, lin_dots_penalty
import pyomo.environ as pyo
from math import factorial


@timer
def initialize_model(index_params: IndexParams, discrete_data: DiscreteDict) -> pyo.ConcreteModel:
    """
    Creates indices from the inputs for variables and constraints in the problem.
    Includes constant parameters to generate the matrix for the model.

    Inputs to the problem dictate the dimensionality of the generated model.

    k : k-th index for body
    t, i, j : t-th timestep index, with i and j as dummy indices for t. This is NOT the actual timestep!
    h : h-th position in the sequence

    Cartesian products of these indices are used for variable and constraint construction.

    """
    bodies = discrete_data.bodies
    print(">>>>> INSTANTIATE PYOMO CONCRETE MODEL >>>>>\n")
    seq_model = pyo.ConcreteModel()  # ConcreteModel instantiates during construction
    seq_model.name = "SequenceSearch"
    print("...create main indices and parameters...")
    seq_model.K = pyo.Set(initialize=index_params.bodies_ID)  # body index
    seq_model.T = pyo.RangeSet(index_params.n_timesteps)  # timestep index, NOT SAME TIMESTEP
    seq_model.H = pyo.RangeSet(index_params.seq_length)  # sequence position index
    seq_model.w_k = pyo.Param(
        seq_model.K, initialize=lambda model, k: bodies[k].weight, within=pyo.PositiveReals
    )  # scoring weights
    seq_model.Nk_limit = pyo.Param(initialize=index_params.flyby_limit)
    seq_model.gt_p = pyo.Param(initialize=index_params.gt_planets)
    seq_model.dt_tol = pyo.Param(initialize=(DAY / SPTU).tolist(), within=pyo.PositiveReals)

    # Cartesian product sets
    seq_model.KT = pyo.Set(initialize=[(k, t) for k in seq_model.K for t in seq_model.T])
    seq_model.KIJ = pyo.Set(
        initialize=[(k, i, j) for k in seq_model.K for i in seq_model.T for j in range(1, i)]
    )

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

    ### Print out important details ###
    return seq_model


@timer
def x_vars_and_constrs(seq_model: pyo.ConcreteModel):
    """
    x[k,t,h] : primary binary variable to select body k at timestep t for position h

    Constraints imposed:
    - no more than one selection per position h
    - no more than one state being selected per body k
    - must select up to the target sequence length
    - only score up to the flyby limit per k
    - time must be monotonically increasing per h
    - do not repeat bodies between h and h+1

    """
    print("...create x_kth binary variable...")
    # X Binary Decision Variable: k-th body, t-th timestep, and h-th position in the sequence
    seq_model.x_kth = pyo.Var(seq_model.K * seq_model.T * seq_model.H, within=pyo.Binary)

    print("...create x parition constraint...")
    # x partition constraint: selection of bodies must equal to h_tot
    seq_model.x_partition = pyo.Constraint(
        rule=pyo.summation(seq_model.x_kth) <= seq_model.H.at(-1)
    )

    print("...create x_kt* packing constraints...")
    # for each (k,t), it can only exist in one position at most
    seq_model.x_kt_packing = pyo.Constraint(
        seq_model.KT, rule=lambda model, k, t: pyo.quicksum(model.x_kth[k, t, :]) <= 1
    )

    print("...create x_**h partition constraints...")
    # for each h, there must only be one body at some time
    seq_model.x_h_packing = pyo.Constraint(
        seq_model.H,
        rule=lambda model, h: pyo.quicksum(model.x_kth[..., h]) <= 1,
    )

    print("...create x_k** max number of scientific flybys...")
    # for each k, there can only be a total of N_k flybys counted
    seq_model.flyby_limit = pyo.Constraint(
        seq_model.K,
        rule=lambda model, k: pyo.quicksum(model.x_kth[k, ...]) <= seq_model.Nk_limit,
    )

    print("...create x_*th monotonic time constraints...")
    # for each h, the time must be greater than the previous h

    def monotime_rule(model, h):
        if h > 1:
            term = (
                pyo.quicksum(
                    model.tu_kt[k, t] * model.x_kth[k, t, h] for k in model.K for t in model.T
                )
                - pyo.quicksum(
                    model.tu_kt[k, t] * model.x_kth[k, t, h - 1] for k in model.K for t in model.T
                )
                >= seq_model.dt_tol
            )
        else:
            term = pyo.Constraint.Skip
        return term

    seq_model.monotonic_time = pyo.Constraint(seq_model.H, rule=monotime_rule)

    print("...create x_k*h packing constraint...")
    # do not pick the same body k for sequential h positions

    def nodupes_rule(model, k, h):
        if h > 1:
            term = pyo.quicksum(model.x_kth[k, :, h]) + pyo.quicksum(model.x_kth[k, :, h - 1]) <= 1

        else:
            term = pyo.Constraint.Skip
        return term

    seq_model.no_dupes = pyo.Constraint(seq_model.K * seq_model.H, rule=nodupes_rule)


@timer
def y_vars_and_constrs(seq_model: pyo.ConcreteModel):
    """
    y[k,i,j] : binary variable for body k flyby at timestep i with a previous flyby at timestep j

    Constraints imposed:
    - total number of previous flybys possible cannot be greater than (nk_lim-1)!
    - if x_kt* is selected for k at t = i,j then y_kij can be toggled
    - if x_kt* is not selected for k at both t = i,j, then y_kij cannot be toggled

    """
    # y variables and constraints
    print("...create y_kij indicator variable of previous j flybys for i-th flyby...")
    # Y Binary Indicator Variable: k-th body, i-th timestep, j-th previous timestep
    seq_model.y_kij = pyo.Var(seq_model.KIJ, within=pyo.Binary)

    print("...create y_k** packing constraints...")
    # the amount of total previous flybys cannot be greater than (Nk_lim - 1)!
    seq_model.y_k_packing = pyo.Constraint(
        seq_model.K,
        rule=lambda model, k: pyo.quicksum(model.y_kij[k, ...])
        <= factorial(seq_model.Nk_limit - 1),
    )

    print("...create y_kij big-M constraints...")
    # if there is both x_ki* and x_kj*, then there must be a y_kij
    seq_model.y_bigm1_x = pyo.Constraint(
        seq_model.KIJ,
        rule=lambda model, k, i, j: pyo.quicksum(
            model.x_kth[k, i, h] + model.x_kth[k, j, h] for h in model.H
        )
        <= 10 * model.y_kij[k, i, j] + 1,
    )
    # if there isn't both x_ki* and x_kj*, then there cannot be a y_kij
    seq_model.y_bigm2_x = pyo.Constraint(
        seq_model.KIJ,
        rule=lambda model, k, i, j: pyo.quicksum(
            model.x_kth[k, i, h] + model.x_kth[k, j, h] for h in model.H
        )
        >= 2 - 10 * (1 - model.y_kij[k, i, j]),
    )


@timer
def z_vars_and_constrs(seq_model: pyo.ConcreteModel):
    """
    z[k,t] : binary variable indicating the FIRST body k flyby at timestep t

    Constraints imposed:
    - only one FIRST flyby per k, if there are flybys
    - if x_kt* is selected for k at t, then z_kt may be toggled
    - if x_kt** is selected for k, then there must be one first flyby z_k*
    - z_kt cannot be a first flyby if y_ki* for i = t indicates there are previous flybys

    """
    print("...create z_kt indicator variable of first flyby at t for body k...")
    # Z Binary Indicator Variable: k-th body, j-th first flyby timestep
    seq_model.z_kt = pyo.Var(seq_model.KT, within=pyo.Binary)

    print("...create z_k* packing constraints...")
    # at most only ONE first flyby for body k
    seq_model.z_k_packing = pyo.Constraint(
        seq_model.K, rule=lambda model, k: pyo.quicksum(model.z_kt[k, :]) <= 1
    )

    print("...create z_k* and z_kt implication constraints...")
    # if there is a flyby at that time, then (k, t) can be a first flyby
    seq_model.z_implies_x = pyo.Constraint(
        seq_model.KT,
        rule=lambda model, k, t: model.z_kt[k, t] <= pyo.quicksum(model.x_kth[k, t, :]),
    )
    # if there is a flyby for body k, then there must be a first flyby for k.
    seq_model.z_bigm_x = pyo.Constraint(
        seq_model.K,
        rule=lambda model, k: model.H.at(-1) * pyo.quicksum(model.z_kt[k, :])
        >= pyo.quicksum(model.x_kth[k, ...]),
    )
    # if there are previous flybys at time i, i cannot be a first flyby
    seq_model.z_implies_not_y = pyo.Constraint(
        seq_model.KT,
        rule=lambda model, k, t: (
            model.z_kt[k, t] <= 1 - pyo.quicksum(model.y_kij[k, t, :])
            if t > 1
            else pyo.Constraint.Feasible
        ),
    )


@timer
def grand_tour_vars_and_constrs(seq_model: pyo.ConcreteModel):
    """
    planet_visited[k] : binary variable indicating if a planet k has been visited
    Gp : binary variable indicating threshold of planets visited

    Constraints imposed:
    - if any x_k** for k is selected, toggle planet_visited[k]
    - do not allow planet_visited[k] to be toggled if x_k** is not active
    - if the sum of planet_visited is at least the target value, toggle the grand tour bonus
    - do not allow the grand tour bonus to be toggled unless the sum of planet_visited is over the threshold

    """
    print("...create grand tour bonus indicator variables Zp, Gp, Zc, Gc, and big-M constraints...")
    seq_model.planet_visited = pyo.Var(seq_model.K, within=pyo.Binary)  # planets and yandi
    seq_model.all_planets = pyo.Var(initialize=0, within=pyo.Binary)  # all planets indicator
    seq_model.count_p_bigm1 = pyo.Constraint(
        seq_model.K,
        rule=lambda model, k: pyo.quicksum(model.x_kth[k, ...])
        <= 2 * model.T.at(-1) * model.planet_visited[k],
    )
    seq_model.count_p_bigm2 = pyo.Constraint(
        seq_model.K,
        rule=lambda model, k: pyo.quicksum(model.x_kth[k, ...]) - 1
        >= 2 * model.T.at(-1) * (model.planet_visited[k] - 1),
    )
    seq_model.all_planets_bigm1 = pyo.Constraint(
        rule=pyo.summation(seq_model.planet_visited) - seq_model.gt_p <= seq_model.all_planets * 20
    )
    seq_model.all_planets_bigm2 = pyo.Constraint(
        rule=pyo.summation(seq_model.planet_visited) - seq_model.gt_p
        >= (seq_model.all_planets - 1) * 20
    )


@timer
def traj_arcs_vars_and_constrs(seq_model: pyo.ConcreteModel, dv_table: dict):
    """
    L[k,i,m,j] : binary variable indicating a trajectory arc between bodies k, m at timesteps i, j

    :param seq_model: Description
    :type seq_model: pyo.ConcreteModel
    :param dv_table: Description
    :type dv_table: dict
    """
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
    print("...create L_kimj arc indicator variables...")
    # bodies k and m, times i and j
    seq_model.L_kimj = pyo.Var(seq_model.KIMJ, within=pyo.Binary)

    print("...create L_ki** and L_**mj implication constraints...")
    # if x_kt* isn't 1, then there can't be an L_kt** or L_**kt
    seq_model.L_implies_x_nodes = pyo.Constraint(
        seq_model.KIMJ,
        rule=lambda model, k, i, m, j: model.L_kimj[k, i, m, j] * 2
        <= pyo.quicksum(model.x_kth[k, i, h] + model.x_kth[m, j, h] for h in model.H),
    )

    print("...create L_kt** and L_**kt packing constraints...")
    # each k, t node can only have at most one start and end.
    seq_model.L_single_start = pyo.Constraint(
        seq_model.KT, rule=lambda model, k, t: pyo.quicksum(model.L_kimj[k, t, ...]) <= 1
    )
    seq_model.L_single_end = pyo.Constraint(
        seq_model.KT, rule=lambda model, k, t: 1 >= pyo.quicksum(model.L_kimj[..., k, t])
    )

    print("...create L_kimj positive dt constraints...")
    # if L_kimj is selected, it must have a positive delta-t
    seq_model.positive_dt = pyo.Constraint(
        seq_model.KIMJ,
        rule=lambda model, k, i, m, j: model.L_kimj[k, i, m, j]
        * (model.tu_kt[m, j] - model.tu_kt[k, i])
        >= 0,
    )

    print("...create L_kimj partition constraints...")
    # must have lambert checks up to h_tot - 1
    seq_model.L_partition = pyo.Constraint(
        rule=pyo.summation(seq_model.L_kimj) == seq_model.H.at(-2)
    )

    print("...create L_kimj implication constraints for x_ki(h) and x_mj(h+1)...")

    # if kimj aren't connected by h and h+1, it is not a valid lambert arc
    def h_arcs_expr(model, k, i, m, j, h):
        if h > 1:
            term = model.x_kth[k, i, h - 1] + model.x_kth[m, j, h] <= 1 + model.L_kimj[k, i, m, j]
        else:
            term = pyo.Constraint.Skip
        return term

    seq_model.L_arcs = pyo.Constraint(seq_model.KIMJ * seq_model.H, rule=h_arcs_expr)

    print("...create L_kimj delta-v limit constraints...")
    # if the lambert arc is used, it must not exceed the dv limits.
    seq_model.dv_limit = pyo.Constraint(
        seq_model.KIMJ,
        rule=lambda model, k, i, m, j: model.L_kimj[k, i, m, j]
        * (model.dv1_kimj[k, i, m, j] + model.dv2_kimj[k, i, m, j])
        <= seq_model.dv_limit,
    )


@timer
def objective_fnc(seq_model: pyo.ConcreteModel):
    def obj_rule(model):
        ##### Objective Function and Scoring #####
        print("...create (z_kt, y_kij, k_kth) -> S(r_kij) seasonal penalty terms...")
        flyby_kt = {kt: model.z_kt[kt] for kt in model.KT}

        # subsequent flybys
        lin_term = 0
        for kij in model.KIJ:
            k, i, j = kij
            lin_term += lin_dots_penalty(model.rdu_kt[k, i], model.rdu_kt[k, j]) * model.y_kij[kij]
            if j == i - 1:
                flyby_kt[k, i] = lin_term
                lin_term = 0

        GT_bonus = 1.0 + 0.3 * model.all_planets
        return GT_bonus * pyo.quicksum(model.w_k[k] * flyby_kt[kt] for kt in model.KT)

    seq_model.maximize_score = pyo.Objective(
        rule=obj_rule,
        sense=pyo.maximize,
    )


@timer
def first_arcs_constrs(
    seq_model: pyo.ConcreteModel, body_list: list[int | tuple[int, tuple[int, int]] | None]
):
    for h, body in enumerate(body_list):
        if isinstance(body, tuple):
            seq_model.first_arcs.add(
                pyo.quicksum(
                    seq_model.x_kth[body[0], t, h + 1] for t in range(body[1][0], body[1][1] + 1)
                )
                == 1
            )
        elif body is None:
            continue
        else:
            seq_model.first_arcs.add(pyo.quicksum(seq_model.x_kth[body, :, h + 1]) == 1)
