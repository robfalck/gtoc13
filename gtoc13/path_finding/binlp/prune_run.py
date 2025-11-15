"""
Prune attempt.

"""

import logging
from gtoc13 import bodies_data
from gtoc13.path_finding.binlp.b_utils import (
    create_discrete_dataset,
    IndexParams,
    SolverParams,
)
from gtoc13.path_finding.binlp.problems import run_basic_problem

############### EDIT CONFIG ###############
debug = False
input_dict = dict(Yo=3, Yf=18, perYear=5, bodies_data=bodies_data)
discrete_data, k_body, num, timesteps = create_discrete_dataset(**input_dict)
pidxs_params = IndexParams(
    bodies_ID=k_body,
    n_timesteps=num,
    seq_length=6,
    flyby_limit=3,
    gt_planets=5,
    gt_smalls=3,
    # dv_limit=175.0,
    # dE_tol=30.0,
    # first_arcs=[(10, 9, 8, 7)],
)
solv_params = SolverParams(
    solver_name="scip",  # AMPL-format solvers
)
###########################################

if debug:
    logging.getLogger("pyomo").setLevel(logging.DEBUG)

seg, m = run_basic_problem(pidxs_params, discrete_data, solv_params)
