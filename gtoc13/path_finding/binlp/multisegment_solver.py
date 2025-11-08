"""
Connect multiple BINLP seqments together.
"""

from gtoc13 import bodies_data
from b_utils import create_discrete_dataset, build_dv_table, IndexParams, SolverParams
from problems import run_basic_problem, run_trajseg_problem

############### CONFIG 1 ###############
input_dict = dict(Yo=3, Yf=10, perYear=2, bodies_data=bodies_data)
discrete_data, k_body, num, timesteps = create_discrete_dataset(**input_dict)
dv_table = build_dv_table(k_body, timesteps)
# prob_dict = dict(gt_smalls=13, dv_limit=2000)
pidxs_params = IndexParams(
    bodies_ID=k_body,
    n_timesteps=num,
    seq_length=6,
    flyby_limit=3,
    gt_planets=5,
    dv_limit=10.0,  # km/s
    first_arcs=[(10, 9, 8)],
)
solv_params = SolverParams(
    solver_name="scip",  # AMPL-format solvers
)
###########################################

# segment = run_basic_problem(pidxs_params, discrete_data, solv_params)
segment = run_trajseg_problem(pidxs_params, discrete_data, dv_table, solv_params)

############### CONFIG 2 ###############
# input_dict = dict(Yo=3, Yf=10, perYear=2, bodies_data=bodies_data)
# discrete_data, k_body, num, timesteps = create_discrete_dataset(**input_dict)
# dv_table = build_dv_table(k_body, timesteps)
# # prob_dict = dict(gt_smalls=13, dv_limit=2000)
# pidxs_params = IndexParams(
#     bodies_ID=k_body,
#     n_timesteps=num,
#     seq_length=6,
#     flyby_limit=3,
#     gt_planets=5,
#     dv_limit=10.0,  # km/s
#     first_arcs=[(10, 9, 8)],
# )
###########################################
