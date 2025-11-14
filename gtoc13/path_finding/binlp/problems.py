from numpy import ndarray
from gtoc13.path_finding.binlp.b_utils import (
    timer,
    IndexParams,
    SolverParams,
    ArcTable,
    SequenceTarget,
)
from gtoc13.path_finding.binlp.build_model import (
    initialize_model,
    x_vars_and_constrs,
    y_vars_and_constrs,
    z_vars_and_constrs,
    traj_arcs_vars_and_constrs,
    grand_tour_vars_and_constrs,
    objective_fnc,
    first_arcs_constrs,
    disallow_constrs,
)
from gtoc13.path_finding.binlp.solver_outputs import generate_iterative_solutions, generate_segment


@timer
def run_basic_problem(index_params: IndexParams, discrete_data: dict, solver_params: SolverParams):
    print(">>>>> WRITE PYOMO MODEL >>>>>\n")
    segment_model = initialize_model(index_params=index_params, discrete_data=discrete_data)
    x_vars_and_constrs(segment_model)
    y_vars_and_constrs(segment_model)
    z_vars_and_constrs(segment_model)
    grand_tour_vars_and_constrs(segment_model)
    objective_fnc(segment_model)
    print("...total segment setup time...")

    if index_params.first_arcs:
        print(">>>>> SET UP ASSUMED INITIAL ARCS CONSTRAINTS >>>>>\n")
        first_arcs_constrs(segment_model, index_params.first_arcs)
        print("<<<<< ASSUMED INITIAL ARCS ADDED <<<<<\n")
    print("<<<<< MODEL SETUP COMPLETE <<<<<")

    soln_segments = generate_iterative_solutions(model=segment_model, solver_params=solver_params)
    print("...total time...")

    return soln_segments, segment_model


@timer
def run_trajectory_problem(
    index_params: IndexParams,
    discrete_data: dict,
    solver_params: SolverParams,
    arc_table: ArcTable,
):
    print(">>>>> WRITE PYOMO MODEL >>>>>\n")
    segment_model = initialize_model(index_params=index_params, discrete_data=discrete_data)
    x_vars_and_constrs(segment_model)
    y_vars_and_constrs(segment_model)
    z_vars_and_constrs(segment_model)
    vinfs = traj_arcs_vars_and_constrs(segment_model, arc_table)
    grand_tour_vars_and_constrs(segment_model)
    objective_fnc(segment_model, vinfs)
    print("...total segment setup time...")

    if index_params.first_arcs:
        print(">>>>> SET UP ASSUMED INITIAL ARCS CONSTRAINTS >>>>>\n")
        first_arcs_constrs(segment_model, index_params.first_arcs)
        print("<<<<< ASSUMED INITIAL ARCS ADDED <<<<<\n")
    print("<<<<< MODEL SETUP COMPLETE <<<<<")

    soln_segments = generate_iterative_solutions(model=segment_model, solver_params=solver_params)
    print("...total time...")

    return soln_segments, segment_model


@timer
def run_segment_problem(
    index_params: IndexParams,
    discrete_data: dict,
    solver_params: SolverParams,
    arc_table: ArcTable | None = None,
    sequence: list[SequenceTarget] | None = None,
    flyby_history: dict[int : list[ndarray]] | None = None,
):
    """
    - initialize the model with trajectory problem + extra?
    - set up first arcs
    - set up disallowed bodies/place
    - run
    - process, save the flybys, last item and timestep (not index), unique planets visited
    """
    print(">>>>> WRITE PYOMO MODEL >>>>>\n")
    segment_model = initialize_model(
        index_params=index_params, discrete_data=discrete_data, flyby_history=flyby_history
    )
    x_vars_and_constrs(segment_model)
    y_vars_and_constrs(segment_model)
    z_vars_and_constrs(segment_model)
    vinfs = traj_arcs_vars_and_constrs(segment_model, arc_table)
    grand_tour_vars_and_constrs(segment_model)
    objective_fnc(segment_model, vinfs)
    print("...total segment setup time...")

    if index_params.first_arcs:
        print(">>>>> SET UP ASSUMED INITIAL ARCS CONSTRAINTS >>>>>\n")
        first_arcs_constrs(segment_model, index_params.first_arcs)
        print("<<<<< ASSUMED INITIAL ARCS ADDED <<<<<\n")

    if index_params.disallowed:
        print(">>>>> SET UP DISALLOWED END TARGETS CONSTRAINTS >>>>>\n")
        disallow_constrs(segment_model, index_params.disallowed)
        print("<<<<< DISALLOWED END TARGETS ADDED <<<<<\n")
    print("<<<<< MODEL SETUP COMPLETE <<<<<")

    segment, flyby_history, model = generate_segment(segment_model, solver_params, flyby_history)
    print("...total time...")
    if sequence:
        sequence.extend(segment)
        return sequence, flyby_history, model
    else:
        return segment, flyby_history, model
