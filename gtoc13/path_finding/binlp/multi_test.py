"""
Connect multiple BINLP seqments together.
"""

# import cloudpickle
from gtoc13 import bodies_data
from b_utils import create_discrete_dataset, build_dv_table, IndexParams, SolverParams
from problems import run_segment_problem

############### CONFIG 1 ###############
shift = 0
input_dict_1 = dict(Yo=3 + shift, Yf=10 + shift, perYear=5, bodies_data=bodies_data)
discrete_data_1, k_body_1, num_1, timesteps_1 = create_discrete_dataset(**input_dict_1)
dv_table_1 = build_dv_table(k_body_1, timesteps_1)
pidxs_params_1 = IndexParams(
    bodies_ID=k_body_1,
    n_timesteps=num_1,
    seq_length=5,
    flyby_limit=1,
    gt_planets=5,
    dv_limit=30.0,  # km/s
    first_arcs=[(10, 9, 8, 7)],
)
solv_params = SolverParams(
    solver_name="scip",  # AMPL-format solvers
    soln_gap=0.10,
)
###########################################

segment_1, flyby_history, seg_model = run_segment_problem(
    pidxs_params_1, discrete_data_1, solv_params, dv_table_1
)
