"""
Connect multiple BINLP seqments together.
"""

# import cloudpickle
from gtoc13 import bodies_data
from gtoc13.path_finding.binlp.b_utils import (
    create_discrete_dataset,
    build_arc_table,
    IndexParams,
    SolverParams,
)
from gtoc13.path_finding.binlp.problems import run_segment_problem
import numpy as np

# ############### CONFIG 1 ###############
input_dict_1 = dict(Yo=100, Yf=140, perYear=0.5, bodies_data=bodies_data)
discrete_data_1, k_body_1, num_1, timesteps_1 = create_discrete_dataset(**input_dict_1)
arc_table_1 = build_arc_table(k_body_1, timesteps_1)
pidxs_params_1 = IndexParams(
    bodies_ID=k_body_1,
    n_timesteps=num_1,
    seq_length=3,
    flyby_limit=1,
    gt_planets=3,
    gt_smalls=3,
    dv_limit=175.0,  # km/s
    dE_tol=25.0,
    # first_arcs=[(10, 9, 8, 7)],
    # disallowed=[(1000, "all")],
)
solv_params_1 = SolverParams(
    solver_name="scip",  # AMPL-format solvers
)
###########################################

segment_1, flyby_history, seg_model_1 = run_segment_problem(
    pidxs_params_1, discrete_data_1, solv_params_1, arc_table_1
)

############### CONFIG 1.5 ###############
# input_dict_1 = dict(Yo=80, Yf=120, perYear=0.75, bodies_data=bodies_data)
# discrete_data_1, k_body_1, num_1, timesteps_1 = create_discrete_dataset(**input_dict_1)
# dv_table_1 = build_dv_table(k_body_1, timesteps_1)
# pidxs_params_1 = IndexParams(
#     bodies_ID=k_body_1,
#     n_timesteps=num_1,
#     seq_length=6,
#     flyby_limit=1,
#     gt_planets=6,
#     dv_limit=20.0,  # km/s
#     first_arcs=[(3, (1, 4))],
#     disallowed=[(1000, "all")],
# )
# solv_params_1 = SolverParams(
#     solver_name="scip",  # AMPL-format solvers
# )
# ###########################################

# segment_1, flyby_history, seg_model_1 = run_segment_problem(
#     pidxs_params_1, discrete_data_1, solv_params_1, dv_table_1
# )

# ############### CONFIG 2 ###############
# window = 40
# input_dict_2 = dict(
#     Yo=np.round(segment_1[-1].year),
#     Yf=np.round(segment_1[-1].year) + window,
#     perYear=0.5,
#     bodies_data=bodies_data,
# )
# discrete_data_2, k_body_2, num_2, timesteps_2 = create_discrete_dataset(**input_dict_2)
# dv_table_2 = build_dv_table(k_body_2, timesteps_2)
# pidxs_params_2 = IndexParams(
#     bodies_ID=k_body_2,
#     n_timesteps=num_2,
#     seq_length=6,
#     flyby_limit=2,
#     gt_planets=6,
#     dv_limit=20.0,  # km/s
#     first_arcs=[(segment_1[-1].body_id, (1, 4))],
#     # disallowed=
# )
# solv_params_2 = SolverParams(
#     solver_name="scip",  # AMPL-format solvers
#     soln_gap=0.15,
# )
# ###########################################

# segment_2, flyby_history, seg_model_2 = run_segment_problem(
#     pidxs_params_2, discrete_data_2, solv_params_2, dv_table_2, segment_1, flyby_history
# )
