"""
Main run script.

"""

import pyomo.environ as pyo
import logging
from gtoc13 import bodies_data
from binlp_utils import Timer, create_discrete_dataset, IndexParams, SolverParams
from gtoc13.path_finding.binlp.problem_builder import (
    initialize_model,
    x_vars_and_constrs,
    y_vars_and_constrs,
    z_vars_and_constrs,
    grand_tour_vars_and_constrs,
    objective_fnc,
    first_arcs_constrs,
)
from solver_outputs import generate_solutions

############### CONFIG ###############
debug = False
input_dict = dict(Yo=0, Yf=10, perYear=1, bodies_data=bodies_data)
discrete_data, k_body, num = create_discrete_dataset(**input_dict)
# prob_dict = dict(gt_smalls=13, dv_limit=2000)
pidxs_params = IndexParams(
    bodies_ID=k_body,
    n_timesteps=num,
    seq_length=6,
    flyby_limit=3,
    gt_planets=5,
    first_arcs=[(10, (1, 3)), 9, 8],
)
solv_params = SolverParams(
    solver_name="scip",  # AMPL-format solvers
)
############# END CONFIG #############

if debug:
    logging.getLogger("pyomo").setLevel(logging.DEBUG)

with Timer():
    print(">>>>> WRITE PYOMO MODEL >>>>>\n")
    with Timer():
        S = initialize_model(index_params=pidxs_params, discrete_data=discrete_data)
        x_vars_and_constrs(S)
        y_vars_and_constrs(S)
        z_vars_and_constrs(S)
        grand_tour_vars_and_constrs(S)
        objective_fnc(S)
        print("...total model setup time...")

    if pidxs_params.first_arcs:
        print(">>>>> SET UP ASSUMED INITIAL ARCS CONSTRAINTS >>>>>\n")
        S.first_arcs = pyo.ConstraintList()
        first_arcs_constrs(S, pidxs_params.first_arcs)
        print("<<<<< ASSUMED INITIAL ARCS ADDED <<<<<\n")
    print("<<<<< MODEL SETUP COMPLETE <<<<<")

    solns = generate_solutions(model=S, solver_params=solv_params, bodies_data=bodies_data)
    print("...total time...")
