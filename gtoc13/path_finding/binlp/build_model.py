"""
Binary Integer Non-Linear Programming problem for finding the most valuable
sequences of a given length h for a starting and end year, Yo and Yf.

This file creates the Pyomo model to be paired with a MINLP/CIP/global solver
from APML.

NOTE: users need to update their environment according to the pyproject.toml
and run `conda install scip`.

"""

from gtoc13 import DAY, SPTU, KMPDU
from gtoc13.path_finding.binlp.b_utils import timer, IndexParams, DVTable, lin_dots_penalty
import pyomo.environ as pyo
from math import factorial
from typing import Optional
from numpy import float64


@timer
def initialize_model(
    index_params: IndexParams, discrete_data: dict, flyby_history: Optional[dict] = None
) -> pyo.ConcreteModel:
    """
    Creates indices from the inputs for variables and constraints in the problem.
    Includes constant parameters to generate the matrix for the model.

    Inputs to the problem dictate the dimensionality of the generated model.

    k : k-th index for body
    i, j : i-th timestep index, with j as dummy index for i. This is NOT the actual timestep!
    h : h-th position in the sequence

    Cartesian products of these indices are used for variable and constraint construction.

    """
    bodies = discrete_data
    print(">>>>> INSTANTIATE PYOMO CONCRETE MODEL >>>>>\n")
    seq_model = pyo.ConcreteModel()  # ConcreteModel instantiates during construction
    seq_model.name = "SequenceSearch"
    print("...create main indices and parameters...")
    seq_model.K = pyo.Set(initialize=index_params.bodies_ID)  # body index
    seq_model.I = pyo.RangeSet(index_params.n_timesteps)  # timestep index, NOT SAME TIMESTEP
    seq_model.H = pyo.RangeSet(index_params.seq_length)  # sequence position index
    seq_model.w_k = pyo.Param(
        seq_model.K, initialize=lambda model, k: bodies[k].weight, within=pyo.PositiveReals
    )  # scoring weights
    seq_model.name_k = pyo.Param(seq_model.K, initialize=lambda model, k: bodies[k].name, within=pyo.Any)
    seq_model.period_k = pyo.Param(seq_model.K, initialize=lambda model, k: bodies[k].tp_tu, within=pyo.PositiveReals)
    seq_model.dtu_limit = pyo.Param(initialize=index_params.dv_limit * float64(SPTU / KMPDU), within=pyo.PositiveReals)
    seq_model.dtu_tol = pyo.Param(
        initialize=index_params.dv_match_tol * float64(SPTU / KMPDU), within=pyo.PositiveReals
    )

    seq_model.Nk_limit = pyo.Param(initialize=index_params.flyby_limit, within=pyo.PositiveIntegers)
    seq_model.gt_p = pyo.Param(initialize=index_params.gt_planets)
    seq_model.dt_tol = pyo.Param(initialize=float(DAY * 7 / SPTU), within=pyo.PositiveReals)
    if flyby_history:
        seq_model.prev_encounter = pyo.Param(
            seq_model.K,
            initialize=lambda model, k: 1 if k in flyby_history.keys() else 0,
            within=pyo.Binary,
        )
        seq_model.prev_fb_number = pyo.Param(
            seq_model.K,
            initialize=lambda model, k: len(flyby_history[k]) if k in flyby_history.keys() else 0,
        )
        seq_model.flyby_history = pyo.Param(seq_model.K, initialize=flyby_history, within=pyo.Any)

    # Cartesian product sets
    seq_model.KI = pyo.Set(initialize=[(k, i) for k in seq_model.K for i in seq_model.I])
    seq_model.KIJ = pyo.Set(initialize=[(k, i, j) for k in seq_model.K for i in seq_model.I for j in range(1, i)])

    # Parameters requiring the Cartesian product sets
    seq_model.tu_ki = pyo.Param(
        seq_model.KI,
        initialize=lambda model, k, i: bodies[k].t_tu[i - 1].tolist(),
        within=pyo.NonNegativeReals,
    )
    seq_model.rdu_ki = pyo.Param(
        seq_model.KI,
        initialize=lambda model, k, i: bodies[k].r_du[i - 1],
        within=pyo.Any,
    )
    return seq_model


@timer
def x_vars_and_constrs(seq_model: pyo.ConcreteModel):
    """
    x[k,i,h] : primary binary variable to select body k at timestep i for position h

    Constraints imposed:
    - no more than one selection per position h
    - no more than one state being selected per body k
    - must select up to the target sequence length
    - only score up to the flyby limit per k
    - time must be monotonically increasing per h
    - do not repeat bodies between h and h+1

    """
    print("...create x_kih binary variable...")
    # X Binary Decision Variable: k-th body, t-th timestep, and h-th position in the sequence
    seq_model.x_kih = pyo.Var(seq_model.K * seq_model.I * seq_model.H, within=pyo.Binary)

    print("...create x parition constraint...")
    # x partition constraint: selection of bodies must equal to h_tot
    seq_model.x_partition = pyo.Constraint(rule=pyo.summation(seq_model.x_kih) == seq_model.H.at(-1))

    print("...create x_**h partition constraints...")
    # for each h, there must only be one body at some time
    seq_model.x_h_partition = pyo.Constraint(
        seq_model.H,
        rule=lambda model, h: pyo.quicksum(model.x_kih[..., h]) == 1,
    )

    print("...create x_ki* packing constraints...")
    # for each (k,i), it can only exist in one position at most
    seq_model.x_ki_packing = pyo.Constraint(
        seq_model.KI, rule=lambda model, k, i: pyo.quicksum(model.x_kih[k, i, :]) <= 1
    )

    print("...create x_k** max number of scientific flybys...")

    # def flyby_limit_rule(model, k):
    #     if model.find_component("flyby_history"):
    #         prev_flyby = len(model.flyby_history[k]) - pyo.quicksum(model.x_kih[k, :, 1])
    #     else:
    #         prev_flyby = 0
    #     return pyo.quicksum(model.x_kih[k, ...]) + prev_flyby <= seq_model.Nk_limit

    # for each k, there can only be a total of N_k flybys counted
    # seq_model.flyby_limit = pyo.Constraint(
    #     seq_model.K,
    #     rule=flyby_limit_rule,
    # )
    # for each k, there can only be a total of N_k flybys counted
    seq_model.flyby_limit = pyo.Constraint(
        seq_model.K,
        rule=lambda model, k: pyo.quicksum(model.x_kih[k, ...]) <= seq_model.Nk_limit,
    )

    print("...create x_*ih monotonic time constraints...")
    # for each h, the time must be greater than the previous h

    def monotime_rule(model, h):
        if h > 1:
            term = (
                pyo.quicksum(model.tu_ki[k, i] * model.x_kih[k, i, h] for k in model.K for i in model.I)
                - pyo.quicksum(model.tu_ki[k, i] * model.x_kih[k, i, h - 1] for k in model.K for i in model.I)
                >= seq_model.dt_tol
            )
        else:
            term = pyo.Constraint.Skip
        return term

    seq_model.monotonic_time = pyo.Constraint(seq_model.H, rule=monotime_rule)

    print("...create x_k*h packing constraint...")
    # do not pick the same body k for sequential h positions unless the timestep is larger than 1/3 of their period.

    def nodupes_rule(model, k, h):
        if h > 1:
            term = pyo.quicksum(model.x_kih[k, :, h]) + pyo.quicksum(model.x_kih[k, :, h - 1]) <= 1

        else:
            term = pyo.Constraint.Skip
        return term

    seq_model.no_dupes = pyo.Constraint(seq_model.K * seq_model.H, rule=nodupes_rule)
    # def successive_flyby(model, k, i, h):
    #     if h > 1 and i > 1:
    #         term = (
    #             model.x_kih[k, i, h] * model.tu_ki[k, i] - model.x_kih[k, i - 1, h - 1] * model.tu_ki[k, i - 1]
    #         ) >= model.period_k[k] / 3


@timer
def y_vars_and_constrs(seq_model: pyo.ConcreteModel):
    """
    y[k,i,j] : binary variable for body k flyby at timestep i with a previous flyby at timestep j

    Constraints imposed:
    - total number of previous flybys possible cannot be greater than (nk_lim-1)!
    - if x_k_* is selected for k at i,j then y_kij can be toggled
    - if x_k_* is not selected for k at both i,j, then y_kij cannot be toggled

    """
    # y variables and constraints
    print("...create y_kij indicator variable of previous j flybys for i-th flyby...")
    # Y Binary Indicator Variable: k-th body, i-th timestep, j-th previous timestep
    seq_model.y_kij = pyo.Var(seq_model.KIJ, within=pyo.Binary)

    if seq_model.find_component("flyby_history"):
        pass

    print("...create y_k** packing constraints...")
    # the amount of total previous flybys cannot be greater than (Nk_lim - 1)!
    seq_model.y_k_packing = pyo.Constraint(
        seq_model.K,
        rule=lambda model, k: pyo.quicksum(model.y_kij[k, ...]) <= factorial(seq_model.Nk_limit - 1),
    )

    print("...create y_kij big-M constraints...")
    # if there is both x_ki* and x_kj*, then there must be a y_kij
    seq_model.y_bigm1_x = pyo.Constraint(
        seq_model.KIJ,
        rule=lambda model, k, i, j: pyo.quicksum(model.x_kih[k, i, h] + model.x_kih[k, j, h] for h in model.H)
        <= 10 * model.y_kij[k, i, j] + 1,
    )
    # if there isn't both x_ki* and x_kj*, then there cannot be a y_kij
    seq_model.y_bigm2_x = pyo.Constraint(
        seq_model.KIJ,
        rule=lambda model, k, i, j: pyo.quicksum(model.x_kih[k, i, h] + model.x_kih[k, j, h] for h in model.H)
        >= 2 - 10 * (1 - model.y_kij[k, i, j]),
    )


@timer
def z_vars_and_constrs(seq_model: pyo.ConcreteModel):
    """
    z[k,i] : binary variable indicating the FIRST body k flyby at timestep i

    Constraints imposed:
    - only one FIRST flyby per k, if there are flybys
    - if x_ki* is selected for k at i, then z_ki may be toggled
    - if x_k** is selected for k, then there must be one first flyby z_k*
    - z_ki cannot be a first flyby if y_ki* for timestep i indicates there are previous flybys

    """
    print("...create z_ki indicator variable of first flyby at i for body k...")
    # Z Binary Indicator Variable: k-th body, i-th first flyby timestep
    seq_model.z_ki = pyo.Var(seq_model.KI, within=pyo.Binary)

    print("...create z_k* packing constraints...")
    # at most only ONE first flyby for body k
    seq_model.z_k_packing = pyo.Constraint(seq_model.K, rule=lambda model, k: pyo.quicksum(model.z_ki[k, :]) <= 1)

    print("...create z_k* and z_ki implication constraints...")
    # if there is a flyby at that time, then (k, i) can be a first flyby
    seq_model.z_implies_x = pyo.Constraint(
        seq_model.KI,
        rule=lambda model, k, i: model.z_ki[k, i] <= pyo.quicksum(model.x_kih[k, i, :]),
    )

    # if there is a flyby for body k, then there must be a first flyby for k, unless it's already in the history.
    # def z_bigm_x_history_rule(model, k):
    #     if model.find_component("flyby_history"):
    #         history_term = len(model.flyby_history[k])
    #     else:
    #         history_term = 0
    #     return model.H.at(-1) * (pyo.quicksum(model.z_ki[k, :]) + history_term) >= pyo.quicksum(
    #         model.x_kih[k, ...]
    #     )

    # seq_model.z_bigm_x = pyo.Constraint(
    #     seq_model.K,
    #     rule=z_bigm_x_history_rule,
    # )
    seq_model.z_bigm_x = pyo.Constraint(
        seq_model.K,
        rule=lambda model, k: model.H.at(-1) * pyo.quicksum(model.z_ki[k, :]) >= pyo.quicksum(model.x_kih[k, ...]),
    )
    # if there are previous flybys at time i, i cannot be a first flyby
    seq_model.z_implies_not_y = pyo.Constraint(
        seq_model.KI,
        rule=lambda model, k, i: (
            model.z_ki[k, i] <= 1 - pyo.quicksum(model.y_kij[k, i, :]) if i > 1 else pyo.Constraint.Feasible
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
        rule=lambda model, k: pyo.quicksum(model.x_kih[k, ...]) <= 2 * model.I.at(-1) * model.planet_visited[k],
    )
    seq_model.count_p_bigm2 = pyo.Constraint(
        seq_model.K,
        rule=lambda model, k: pyo.quicksum(model.x_kih[k, ...]) - 1
        >= 2 * model.I.at(-1) * (model.planet_visited[k] - 1),
    )
    seq_model.all_planets_bigm1 = pyo.Constraint(
        rule=pyo.summation(seq_model.planet_visited) - seq_model.gt_p <= seq_model.all_planets * 20
    )
    seq_model.all_planets_bigm2 = pyo.Constraint(
        rule=pyo.summation(seq_model.planet_visited) - seq_model.gt_p >= (seq_model.all_planets - 1) * 20
    )


@timer
def traj_arcs_vars_and_constrs(seq_model: pyo.ConcreteModel, dv_table: DVTable):
    """
    L[k,i,m,j] : binary variable indicating a trajectory arc between bodies k, m at timesteps i, j

    :param seq_model: Description
    :type seq_model: pyo.ConcreteModel
    :param dv_table: Description
    :type dv_table: dict
    """
    # KIMJ indices:
    seq_model.KIMJ = pyo.Set(initialize=dv_table.dv_in.keys())

    # Delta-vs and tofs
    seq_model.dvout_kimj = pyo.Param(
        seq_model.KIMJ,
        initialize=lambda model, k, i, m, j: dv_table.dv_out[k, i, m, j],
        within=pyo.PositiveReals,
    )
    seq_model.dvin_kimj = pyo.Param(
        seq_model.KIMJ,
        initialize=lambda model, k, i, m, j: dv_table.dv_in[k, i, m, j],
        within=pyo.PositiveReals,
    )

    print("...create L_kimj arc indicator variables...")
    # bodies k and m, times i and j
    seq_model.L_kimj = pyo.Var(seq_model.KIMJ, within=pyo.Binary)

    print("...create L_kimj partition constraints...")
    # must have lambert checks up to h_tot - 1
    seq_model.L_partition = pyo.Constraint(rule=pyo.summation(seq_model.L_kimj) == seq_model.H.at(-2))

    print("...create L_kimj implication constraints for x_ki(h) and x_mj(h+1)...")

    # if kimj aren't connected by h and h+1, it is not a valid lambert arc
    def h_arcs_expr(model, k, i, m, j, h):
        if h > 1:
            term = model.x_kih[k, i, h - 1] + model.x_kih[m, j, h] <= 1 + model.L_kimj[k, i, m, j]
        else:
            term = pyo.Constraint.Skip
        return term

    seq_model.L_arcs = pyo.Constraint(seq_model.KIMJ * seq_model.H, rule=h_arcs_expr)

    print("...create L_kimj delta-v limit constraints...")
    # if the lambert arc is used, it must not exceed the dv limits.
    seq_model.dv_limit = pyo.Constraint(
        seq_model.KIMJ,
        rule=lambda model, k, i, m, j: model.L_kimj[k, i, m, j]
        * (model.dvin_kimj[k, i, m, j] + model.dvout_kimj[k, i, m, j])
        <= model.dtu_limit,
    )

    print("...create L_**ki and L_ki** dv_in and dv_out match constraints...")
    # if x_ki* is an internal body, dv_in should match dv_out
    dv_in = {}
    dv_out = {}
    for k, i, m, j in seq_model.KIMJ:
        dv_in[m, j] = seq_model.dvin_kimj[k, i, m, j] * seq_model.L_kimj[k, i, m, j]
        dv_out[k, i] = seq_model.dvout_kimj[k, i, m, j] * seq_model.L_kimj[k, i, m, j]
    seq_model.dv_match = pyo.ConstraintList()
    for k, i in seq_model.KI:
        if (k, i) in dv_in and (k, i) in dv_out:
            diff_term = dv_in[k, i] - dv_out[k, i]
            seq_model.dv_match.add(-seq_model.dtu_tol <= diff_term)
            seq_model.dv_match.add(diff_term <= seq_model.dtu_tol)


@timer
def objective_fnc(seq_model: pyo.ConcreteModel):
    def obj_rule(model):
        ##### Objective Function and Scoring #####
        print("...create (z_ki, y_kij, k_kih) -> S(r_kij) seasonal penalty terms...")
        flyby_ki = {ki: model.z_ki[ki] for ki in model.KI}

        # subsequent flybys
        lin_term = 0
        for k, i, j in model.KIJ:
            lin_term += lin_dots_penalty(model.rdu_ki[k, i], model.rdu_ki[k, j]) * model.y_kij[k, i, j]
            if j == i - 1:
                flyby_ki[k, i] = lin_term
                lin_term = 0

        GT_bonus = 1.0 + 0.3 * model.all_planets
        # if seq_model.find_component("L_kimj"):
        #     dv_penalty = -pyo.summation(model.L_kimj, model.dv_kimj, index=model.KIMJ) / 10
        # else:
        #     dv_penalty = 0
        return GT_bonus * pyo.quicksum(model.w_k[k] * flyby_ki[ki] for ki in model.KI)  # + dv_penalty

    seq_model.maximize_score = pyo.Objective(
        rule=obj_rule,
        sense=pyo.maximize,
    )


@timer
def first_arcs_constrs(seq_model: pyo.ConcreteModel, body_list: list[int | tuple[int, tuple[int, int]] | tuple | None]):
    if not seq_model.find_component("first_arcs"):
        seq_model.first_arcs = pyo.ConstraintList()

    for h, body in enumerate(body_list):
        if not isinstance(body, int):
            if isinstance(body[1], tuple):
                seq_model.first_arcs.add(
                    pyo.quicksum(seq_model.x_kih[body[0], i, h + 1] for i in range(body[1][0], body[1][1] + 1)) == 1
                )
            elif isinstance(body, tuple):
                seq_model.first_arcs.add(
                    pyo.quicksum(seq_model.x_kih[k, i, h + 1] for k in body for i in seq_model.I) == 1
                )
            elif body is None:
                continue
        else:
            seq_model.first_arcs.add(pyo.quicksum(seq_model.x_kih[body, :, h + 1]) == 1)


@timer
def nogood_cuts_constrs(seq_model: pyo.ConcreteModel):
    if not seq_model.find_component("ng_cuts"):
        seq_model.ng_cuts = pyo.ConstraintList()
    expr = 0
    for kih, v in seq_model.x_kih:
        if pyo.value(v) < 0.5:
            expr += seq_model.x_kih[kih]
        else:
            expr += 1 - seq_model.x_kih[kih]
    seq_model.ng_cuts.add(expr >= 1)


@timer
def disallow_constrs(
    seq_model: pyo.ConcreteModel,
    disallowed: list[tuple[int, str]],
):
    if not seq_model.find_component("disallow"):
        seq_model.disallow = pyo.ConstraintList()
    for body in disallowed:
        match body[-1]:
            case "all":
                seq_model.disallow.add(pyo.quicksum(seq_model.x_kih[body[0], ...]) == 0)
            case "end":
                seq_model.disallow.add(pyo.quicksum(seq_model.x_kih[body[0], :, seq_model.H.at(-1)]) == 0)
            case __:
                raise Exception("type 'all' or 'end' after the body ID.")
