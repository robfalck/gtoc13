from b_utils import timer, IndexParams, SolverParams, DVTable
from build_model import (
    initialize_model,
    x_vars_and_constrs,
    y_vars_and_constrs,
    z_vars_and_constrs,
    traj_arcs_vars_and_constrs,
    grand_tour_vars_and_constrs,
    objective_fnc,
    first_arcs_constrs,
)
from solver_outputs import generate_iterative_solutions


@timer
def run_basic_problem(index_params: IndexParams, discrete_data: dict, solver_params: SolverParams):
    print(">>>>> WRITE PYOMO MODEL >>>>>\n")
    segment = initialize_model(index_params=index_params, discrete_data=discrete_data)
    x_vars_and_constrs(segment)
    y_vars_and_constrs(segment)
    z_vars_and_constrs(segment)
    grand_tour_vars_and_constrs(segment)
    objective_fnc(segment)
    print("...total segment setup time...")

    if index_params.first_arcs:
        print(">>>>> SET UP ASSUMED INITIAL ARCS CONSTRAINTS >>>>>\n")
        first_arcs_constrs(segment, index_params.first_arcs)
        print("<<<<< ASSUMED INITIAL ARCS ADDED <<<<<\n")
    print("<<<<< MODEL SETUP COMPLETE <<<<<")

    segments = generate_iterative_solutions(model=segment, solver_params=solver_params)
    print("...total time...")

    return segments


@timer
def run_trajseg_problem(
    index_params: IndexParams, discrete_data: dict, dv_table: DVTable, solver_params: SolverParams
):
    print(">>>>> WRITE PYOMO MODEL >>>>>\n")
    segment = initialize_model(index_params=index_params, discrete_data=discrete_data)
    x_vars_and_constrs(segment)
    y_vars_and_constrs(segment)
    z_vars_and_constrs(segment)
    traj_arcs_vars_and_constrs(segment, dv_table)
    grand_tour_vars_and_constrs(segment)
    objective_fnc(segment)
    print("...total segment setup time...")

    if index_params.first_arcs:
        print(">>>>> SET UP ASSUMED INITIAL ARCS CONSTRAINTS >>>>>\n")
        first_arcs_constrs(segment, index_params.first_arcs)
        print("<<<<< ASSUMED INITIAL ARCS ADDED <<<<<\n")
    print("<<<<< MODEL SETUP COMPLETE <<<<<")

    segments = generate_iterative_solutions(model=segment, solver_params=solver_params)
    print("...total time...")

    return segments
