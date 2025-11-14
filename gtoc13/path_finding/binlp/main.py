"""
Main run script to create the Pyomo BINLP model and solve it for
a prescribed number of solutions (problem can be degenerate based
on parameters).

Only edit the entries between the ###### lines. Returns a list of
solution sequences based on number of iterations.

"""

import logging
from gtoc13 import bodies_data
from gtoc13.path_finding.binlp.b_utils import (
    create_discrete_dataset,
    build_arc_table,
    IndexParams,
    SolverParams,
)
from gtoc13.path_finding.binlp.problems import run_basic_problem, run_trajectory_problem

############### EDIT CONFIG ###############
debug = False
input_dict = dict(Yo=3, Yf=68, perYear=0.25, bodies_data=bodies_data)
discrete_data, k_body, num, timesteps = create_discrete_dataset(**input_dict)
arc_table = build_arc_table(k_body, timesteps)
pidxs_params = IndexParams(
    bodies_ID=k_body,
    n_timesteps=num,
    seq_length=5,
    flyby_limit=1,
    gt_planets=5,
    dv_limit=200.0,
    dE_tol=30.0,
    # first_arcs=[(10, 9, 8, 7)],
)
solv_params = SolverParams(
    solver_name="scip",  # AMPL-format solvers
)
###########################################

if debug:
    logging.getLogger("pyomo").setLevel(logging.DEBUG)

# seg, m = run_basic_problem(pidxs_params, discrete_data, solv_params)
tseg, tm = run_trajectory_problem(pidxs_params, discrete_data, solv_params, arc_table)

import numpy as np
import pyomo.environ as pyo
