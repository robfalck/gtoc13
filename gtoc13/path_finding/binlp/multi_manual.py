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
input_dict_1 = dict(Yo=30, Yf=60, perYear=0.5, bodies_data=bodies_data)
discrete_data_1, k_body_1, num_1, timesteps_1 = create_discrete_dataset(**input_dict_1)
arc_table_1 = build_arc_table(k_body_1, timesteps_1)
pidxs_params_1 = IndexParams(
    bodies_ID=k_body_1,
    n_timesteps=num_1,
    seq_length=4,
    flyby_limit=1,
    gt_planets=4,
    dv_limit=100.0,  # km/s
    dE_tol=15.0,
    first_arcs=[(10, 9, 8, 7)],
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

# ############### CONFIG 3 ###############
# shift = 2
# input_dict_3 = dict(Yo=3 + 10 * shift, Yf=13 + 10 * shift, perYear=5, bodies_data=bodies_data)
# discrete_data_3, k_body_3, num_3, timesteps_3 = create_discrete_dataset(**input_dict_3)
# dv_table_3 = build_dv_table(k_body_3, timesteps_3)
# pidxs_params_3 = IndexParams(
#     bodies_ID=k_body_3,
#     n_timesteps=num_3,
#     seq_length=5,
#     flyby_limit=4,
#     gt_planets=11,
#     dv_limit=150.0,  # km/s
#     first_arcs=[segment_2[-1].body_id],
# )
# solv_params_3 = SolverParams(
#     solver_name="scip",  # AMPL-format solvers
#     soln_gap=0.5,
# )
# ###########################################

# segment_3, flyby_history, seg_model_3 = run_segment_problem(
#     pidxs_params_3, discrete_data_3, solv_params_3, dv_table_3, flyby_history
# )

# ############### CONFIG 3 ###############
# shift = 3
# input_dict_4 = dict(Yo=3 + 10 * shift, Yf=13 + 10 * shift, perYear=5, bodies_data=bodies_data)
# discrete_data_4, k_body_4, num_4, timesteps_4 = create_discrete_dataset(**input_dict_4)
# dv_table_4 = build_dv_table(k_body_4, timesteps_4)
# pidxs_params_4 = IndexParams(
#     bodies_ID=k_body_4,
#     n_timesteps=num_4,
#     seq_length=5,
#     flyby_limit=1,
#     gt_planets=5,
#     dv_limit=150.0,  # km/s
#     first_arcs=[segment_3[-1].body_id],
# )
# solv_params_4 = SolverParams(
#     solver_name="scip",  # AMPL-format solvers
#     soln_gap=0.5,
# )
# ###########################################

# segment_4, flyby_history, seg_model_4 = run_segment_problem(
#     pidxs_params_4, discrete_data_4, solv_params_4, dv_table_4, flyby_history
# )
