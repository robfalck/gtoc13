"""
Connect multiple BINLP seqments together.
"""

# import cloudpickle
from gtoc13 import bodies_data
from b_utils import create_discrete_dataset, build_dv_table, IndexParams, SolverParams
from problems import run_trajectory_problem

############### CONFIG 1 ###############
input_dict1 = dict(Yo=3, Yf=13, perYear=2, bodies_data=bodies_data)
discrete_data1, k_body1, num1, timesteps1 = create_discrete_dataset(**input_dict1)
dv_table1 = build_dv_table(k_body1, timesteps1)
# prob_dict = dict(gt_smalls=13, dv_limit=2000)
pidxs_params1 = IndexParams(
    bodies_ID=k_body1,
    n_timesteps=num1,
    seq_length=7,
    flyby_limit=1,
    gt_planets=7,
    dv_limit=10.0,  # km/s
    first_arcs=[(10, 9, 8, 7), (10, 9, 8, 7)],
)
solv_params = SolverParams(
    solver_name="scip",  # AMPL-format solvers
    write_log=True,
)
###########################################

segment_1 = run_trajectory_problem(pidxs_params1, discrete_data1, dv_table1, solv_params)
nextstart_1 = segment_1[0][-1]

############## CONFIG 2 ###############
input_dict2 = dict(
    Yo=nextstart_1.year - 0.5, Yf=nextstart_1.year + 10, perYear=2, bodies_data=bodies_data
)
discrete_data2, k_body2, num2, timesteps2 = create_discrete_dataset(**input_dict2)
dv_table2 = build_dv_table(k_body2, timesteps2)
# prob_dict = dict(gt_smalls=13, dv_limit=2000)
pidxs_params2 = IndexParams(
    bodies_ID=k_body2,
    n_timesteps=num2,
    seq_length=4,
    flyby_limit=3,
    gt_planets=4,
    dv_limit=10.0,  # km/s
    first_arcs=[(nextstart_1.body_state[1][0], (1, 4))],
)
##########################################

segment_2 = run_trajectory_problem(pidxs_params2, discrete_data2, dv_table2, solv_params)
