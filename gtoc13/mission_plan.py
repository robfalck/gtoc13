from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np

import dymos as dm

from gtoc13 import bodies_data
from gtoc13.dymos_solver.dymos_solver import get_dymos_vectorized_solver_problem


class GeneralConstraint(BaseModel):
    """
    A general constraint specification for OpenMDAO.

    Attributes
    ----------
    lower : Optional[float]
        Lower bound for the constraint
    upper : Optional[float]
        Upper bound for the constraint
    equals : Optional[float]
        Equality constraint value
    ref : Optional[float]
        Reference value for scaling
    ref0 : Optional[float]
        Zero-reference value for scaling
    indices : Optional[List[int]]
        Indices to constrain (for array variables)
    units : Optional[str]
        Units for the constraint
    """
    lower: Optional[float] = Field(default=None, description="Lower bound")
    upper: Optional[float] = Field(default=None, description="Upper bound")
    equals: Optional[float] = Field(default=None, description="Equality constraint value")
    ref: Optional[float] = Field(default=None, description="Reference value for scaling")
    ref0: Optional[float] = Field(default=None, description="Zero-reference value for scaling")
    indices: Optional[List[int]] = Field(default=None, description="Indices to constrain")
    units: Optional[str] = Field(default=None, description="Units for the constraint")


class PathConstraint(BaseModel):
    """
    A path constraint specification for Dymos phases.

    Attributes
    ----------
    lower : Optional[float]
        Lower bound for the constraint along the path
    upper : Optional[float]
        Upper bound for the constraint along the path
    equals : Optional[float]
        Equality constraint value along the path
    ref : Optional[float]
        Reference value for scaling
    ref0 : Optional[float]
        Zero-reference value for scaling
    indices : Optional[List[int]]
        Indices to constrain (for array variables)
    units : Optional[str]
        Units for the constraint
    """
    lower: Optional[float] = Field(default=None, description="Lower bound")
    upper: Optional[float] = Field(default=None, description="Upper bound")
    equals: Optional[float] = Field(default=None, description="Equality constraint value")
    ref: Optional[float] = Field(default=None, description="Reference value for scaling")
    ref0: Optional[float] = Field(default=None, description="Zero-reference value for scaling")
    indices: Optional[List[int]] = Field(default=None, description="Indices to constrain")
    units: Optional[str] = Field(default=None, description="Units for the constraint")


class BoundaryConstraint(BaseModel):
    """
    A boundary constraint specification for Dymos phases.

    Attributes
    ----------
    loc : str
        Location of the boundary constraint ('initial' or 'final')
    lower : Optional[float]
        Lower bound for the constraint
    upper : Optional[float]
        Upper bound for the constraint
    equals : Optional[float]
        Equality constraint value
    ref : Optional[float]
        Reference value for scaling
    ref0 : Optional[float]
        Zero-reference value for scaling
    indices : Optional[List[int]]
        Indices to constrain (for array variables)
    units : Optional[str]
        Units for the constraint
    """
    loc: str = Field(..., description="Location: 'initial' or 'final'")
    lower: Optional[float] = Field(default=None, description="Lower bound")
    upper: Optional[float] = Field(default=None, description="Upper bound")
    equals: Optional[float] = Field(default=None, description="Equality constraint value")
    ref: Optional[float] = Field(default=None, description="Reference value for scaling")
    ref0: Optional[float] = Field(default=None, description="Zero-reference value for scaling")
    indices: Optional[List[int]] = Field(default=None, description="Indices to constrain")
    units: Optional[str] = Field(default=None, description="Units for the constraint")


class DesignVariable(BaseModel):
    """
    A design variable specification for OpenMDAO.

    Attributes
    ----------
    lower : Optional[float]
        Lower bound for the design variable
    upper : Optional[float]
        Upper bound for the design variable
    ref : Optional[float]
        Reference value for scaling
    ref0 : Optional[float]
        Zero-reference value for scaling
    scaler : Optional[float]
        Scaler value for the design variable
    adder : Optional[float]
        Adder value for the design variable
    indices : Optional[List[int]]
        Indices to use as design variables (for array variables)
    units : Optional[str]
        Units for the design variable
    """
    lower: Optional[float] = Field(default=None, description="Lower bound")
    upper: Optional[float] = Field(default=None, description="Upper bound")
    ref: Optional[float] = Field(default=None, description="Reference value for scaling")
    ref0: Optional[float] = Field(default=None, description="Zero-reference value for scaling")
    scaler: Optional[float] = Field(default=None, description="Scaler value")
    adder: Optional[float] = Field(default=None, description="Adder value")
    indices: Optional[List[int]] = Field(default=None, description="Indices to use as design variables")
    units: Optional[str] = Field(default=None, description="Units for the design variable")


class Objective(BaseModel):
    """
    An objective specification for OpenMDAO.

    Attributes
    ----------
    ref : Optional[float]
        Reference value for scaling
    ref0 : Optional[float]
        Zero-reference value for scaling
    scaler : Optional[float]
        Scaler value for the objective
    adder : Optional[float]
        Adder value for the objective
    index : Optional[int]
        Index to use as objective (for array variables)
    units : Optional[str]
        Units for the objective
    """
    ref: Optional[float] = Field(default=None, description="Reference value for scaling")
    ref0: Optional[float] = Field(default=None, description="Zero-reference value for scaling")
    scaler: Optional[float] = Field(default=None, description="Scaler value")
    adder: Optional[float] = Field(default=None, description="Adder value")
    index: Optional[int] = Field(default=None, description="Index to use as objective")
    units: Optional[str] = Field(default=None, description="Units for the objective")


class MissionPlan(BaseModel):
    """
    A mission plan for a GTOC13 trajectory.

    Attributes
    ----------
    bodies : List[int]
        List of integer body indices to visit in order
    flyby_times : List[float]
        Guess of the flyby time (years) for each body
    t0 : float
        Initial start time in years (default: 0.0)
    general_constraints : Dict[str, GeneralConstraint]
        General constraints to apply to the OpenMDAO problem (e.g., model-level constraints)
        Keys are OpenMDAO variable names (e.g., 'times', 'flyby_comp.v_inf_mag_defect')
    path_constraints : Dict[str, PathConstraint]
        Path constraints to apply to the phase (constraints along the trajectory)
        Keys are phase variable names (e.g., 'u_n_norm', 'cos_alpha')
    boundary_constraints : Dict[str, BoundaryConstraint]
        Boundary constraints to apply to the phase (constraints at initial/final points)
        Keys are phase variable names (e.g., 'v', 'r')
    design_variables : Dict[str, DesignVariable]
        Design variables for the optimization problem
        Keys are OpenMDAO variable names (e.g., 't0', 'dt', 'y0', 'z0', 'v_end')
    objectives : Dict[str, Objective]
        Objectives for the optimization problem
        Keys are OpenMDAO variable names (e.g., 'E_end', 'times')
    guess_solution : Optional[Any]
        Solution object loaded from a .txt file for use as initial guess
        Not saved to the .pln file (excluded from serialization)
    """
    bodies: List[int] = Field(..., description="List of integer body indices to visit")
    flyby_times: List[float] = Field(..., description="Guess of flyby time (years) for each body")
    t0: float = Field(default=0.0, description="Initial start time in years")
    general_constraints: Dict[str, GeneralConstraint] = Field(
        default_factory=dict,
        description="General constraints to apply to the OpenMDAO problem"
    )
    path_constraints: Dict[str, PathConstraint] = Field(
        default_factory=dict,
        description="Path constraints to apply to the phase"
    )
    boundary_constraints: Dict[str, BoundaryConstraint] = Field(
        default_factory=dict,
        description="Boundary constraints to apply to the phase"
    )
    design_variables: Dict[str, DesignVariable] = Field(
        default_factory=dict,
        description="Design variables for the optimization problem"
    )
    objectives: Dict[str, Objective] = Field(
        default_factory=dict,
        description="Objectives for the optimization problem"
    )
    guess_solution: Optional[Any] = Field(
        default=None,
        exclude=True,
        description="Solution object for initial guess (not saved to file)"
    )

    @field_validator('t0')
    @classmethod
    def validate_t0(cls, v: float) -> float:
        """Validate that t0 is between 0.0 and 200.0 years."""
        if not (0.0 <= v <= 200.0):
            raise ValueError(f"t0 must be between 0.0 and 200.0 years, got {v}")
        return v

    @field_validator('flyby_times')
    @classmethod
    def validate_flyby_times(cls, v: List[float]) -> List[float]:
        """Validate that flyby times are monotonically increasing."""
        if len(v) == 0:
            raise ValueError("flyby_times must contain at least one element")

        for i in range(1, len(v)):
            if v[i] <= v[i-1]:
                raise ValueError(
                    f"flyby_times must be monotonically increasing. "
                    f"Found {v[i]} <= {v[i-1]} at indices {i} and {i-1}"
                )
        return v

    @field_validator('bodies')
    @classmethod
    def validate_bodies_length(cls, v: List[int], info) -> List[int]:
        """Validate that bodies and flyby_times have the same length."""
        # Note: This validator runs before flyby_times, so we can't check the length here
        # We'll check it in a model validator instead
        if len(v) == 0:
            raise ValueError("bodies must contain at least one element")
        return v

    @model_validator(mode='after')
    def validate_mission_plan(self):
        """Validate the entire mission plan after all fields are set."""
        # Check that bodies and flyby_times have the same length
        if len(self.bodies) != len(self.flyby_times):
            raise ValueError(
                f"bodies and flyby_times must have the same length. "
                f"Got {len(self.bodies)} bodies and {len(self.flyby_times)} flyby_times"
            )

        # Check that all flyby_times are greater than t0
        if len(self.flyby_times) > 0 and self.flyby_times[0] <= self.t0:
            raise ValueError(
                f"First flyby time ({self.flyby_times[0]}) must be greater than t0 ({self.t0})"
            )

        return self

    def save(self, filepath: str | Path) -> None:
        """
        Save the mission plan to a .pln file in JSON format.

        Parameters
        ----------
        filepath : str | Path
            Path to the file where the mission plan will be saved.
            Should end with .pln suffix.
        """
        filepath = Path(filepath)
        if filepath.suffix != '.pln':
            filepath = filepath.with_suffix('.pln')

        with open(filepath, 'w') as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, filepath: str | Path) -> 'MissionPlan':
        """
        Load a mission plan from a .pln file.

        Also attempts to load a solution file with the same name (but .txt extension)
        for use as an initial guess. If found, validates that the body sequence matches
        and uses the flyby times from the solution.

        Parameters
        ----------
        filepath : str | Path
            Path to the .pln file to load.

        Returns
        -------
        MissionPlan
            The loaded mission plan instance.
        """
        from gtoc13.solution import GTOC13Solution, FlybyArc
        from gtoc13.constants import YEAR

        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            plan = cls.model_validate_json(f.read())

        # Try to load a solution file with the same name
        solution_path = filepath.with_suffix('.txt')
        if solution_path.exists():
            try:
                print(f"Found solution file: {solution_path}")
                solution = GTOC13Solution.from_file(solution_path)

                # Extract flyby arcs to get bodies and times
                flyby_arcs = [arc for arc in solution.arcs if isinstance(arc, FlybyArc)]

                if len(flyby_arcs) > 0:
                    solution_bodies = [arc.body_id for arc in flyby_arcs]

                    # Check if solution bodies match the first N bodies in the plan
                    n_solution_bodies = len(solution_bodies)
                    n_plan_bodies = len(plan.bodies)

                    if n_solution_bodies <= n_plan_bodies:
                        # Check if first N bodies match
                        if plan.bodies[:n_solution_bodies] == solution_bodies:
                            # Partial or full match - update flyby times for matched arcs
                            solution_flyby_times = [arc.epoch / YEAR for arc in flyby_arcs]

                            # Update the flyby times for the matched bodies
                            for i in range(n_solution_bodies):
                                plan.flyby_times[i] = solution_flyby_times[i]

                            print(f"Loaded solution with {n_solution_bodies} bodies matching first "
                                  f"{n_solution_bodies} of {n_plan_bodies} plan bodies")
                            print(f"Updated flyby times for bodies {solution_bodies}: {solution_flyby_times}")

                            if n_solution_bodies < n_plan_bodies:
                                print(f"Remaining {n_plan_bodies - n_solution_bodies} bodies will use default guess")

                            # Store the solution for use as initial guess
                            plan.guess_solution = solution
                        else:
                            print(f"Warning: Solution body sequence {solution_bodies} does not match "
                                  f"first {n_solution_bodies} bodies of plan {plan.bodies[:n_solution_bodies]}. "
                                  f"Ignoring solution.")
                    else:
                        print(f"Warning: Solution has more bodies ({n_solution_bodies}) than plan ({n_plan_bodies}). "
                              f"Ignoring solution.")
            except Exception as e:
                print(f"Warning: Could not load solution file {solution_path}: {e}")

        return plan

    @classmethod
    def get_default_design_variables(cls) -> Dict[str, DesignVariable]:
        """
        Get the default design variables used by dymos_solver.

        Returns
        -------
        Dict[str, DesignVariable]
            Default design variables
        """
        return {
            't0': DesignVariable(lower=0.0, units='gtoc_year'),
            'dt': DesignVariable(lower=0.0, upper=200, units='gtoc_year'),
            'y0': DesignVariable(units='DU'),
            'z0': DesignVariable(units='DU'),
            'v_end': DesignVariable(units='DU/TU'),
        }

    @classmethod
    def get_default_general_constraints(cls) -> Dict[str, GeneralConstraint]:
        """
        Get the default general constraints used by dymos_solver.

        Returns
        -------
        Dict[str, GeneralConstraint]
            Default general constraints
        """
        return {
            'flyby_comp.v_inf_mag_defect': GeneralConstraint(equals=0.0, units='km/s'),
            'flyby_comp.h_p_defect': GeneralConstraint(upper=0.0, ref=1000.0),
            'times': GeneralConstraint(indices=[-1], upper=199.999, units='gtoc_year'),
        }

    @classmethod
    def get_default_boundary_constraints(cls) -> Dict[str, BoundaryConstraint]:
        """
        Get the default boundary constraints used by dymos_solver.

        Returns
        -------
        Dict[str, BoundaryConstraint]
            Default boundary constraints
        """
        return {
            'v': BoundaryConstraint(loc='initial', indices=[1, 2], equals=0.0),
        }

    @classmethod
    def get_default_objectives(cls) -> Dict[str, Objective]:
        """
        Get the default objectives used by dymos_solver.

        Returns
        -------
        Dict[str, Objective]
            Default objectives
        """
        return {
            'obj': Objective(),
        }

    def get_design_variables_with_defaults(self) -> Dict[str, DesignVariable]:
        """
        Get design variables, filling in defaults for any that are not specified.

        If a design variable is explicitly set to None, it will be removed from the defaults.

        Returns
        -------
        Dict[str, DesignVariable]
            Design variables with defaults applied
        """
        defaults = self.get_default_design_variables()
        defaults.update(self.design_variables)
        # Remove entries that are explicitly set to None
        return {k: v for k, v in defaults.items() if v is not None}

    def get_general_constraints_with_defaults(self) -> Dict[str, GeneralConstraint]:
        """
        Get general constraints, filling in defaults for any that are not specified.

        If a constraint is explicitly set to None, it will be removed from the defaults.

        Returns
        -------
        Dict[str, GeneralConstraint]
            General constraints with defaults applied
        """
        defaults = self.get_default_general_constraints()
        defaults.update(self.general_constraints)
        # Remove entries that are explicitly set to None
        return {k: v for k, v in defaults.items() if v is not None}

    def get_boundary_constraints_with_defaults(self) -> Dict[str, BoundaryConstraint]:
        """
        Get boundary constraints, filling in defaults for any that are not specified.

        If a constraint is explicitly set to None, it will be removed from the defaults.

        Returns
        -------
        Dict[str, BoundaryConstraint]
            Boundary constraints with defaults applied
        """
        defaults = self.get_default_boundary_constraints()
        defaults.update(self.boundary_constraints)
        # Remove entries that are explicitly set to None
        return {k: v for k, v in defaults.items() if v is not None}

    def get_path_constraints_with_defaults(self) -> Dict[str, PathConstraint]:
        """
        Get path constraints, filling in defaults for any that are not specified.

        If a constraint is explicitly set to None, it will be removed from the defaults.

        Returns
        -------
        Dict[str, PathConstraint]
            Path constraints with defaults applied (currently no defaults)
        """
        # No default path constraints currently, but filter out None values
        return {k: v for k, v in self.path_constraints.items() if v is not None}

    def get_objectives_with_defaults(self) -> Dict[str, Objective]:
        """
        Get objectives, filling in defaults for any that are not specified.

        If an objective is explicitly set to None, it will be removed from the defaults.

        Returns
        -------
        Dict[str, Objective]
            Objectives with defaults applied
        """
        defaults = self.get_default_objectives()
        defaults.update(self.objectives)
        # Remove entries that are explicitly set to None
        return {k: v for k, v in defaults.items() if v is not None}

    def solve(self, num_nodes=20, run_driver=True):
        """
        Solve the trajectory optimization problem defined by this mission plan.

        Parameters
        ----------
        num_nodes : int
            Number of collocation nodes per arc (default: 20)
        run_driver : int
            If True, run the optimization driver. Otherwise just propagate inputs
            through the model.

        Returns
        -------
        prob : om.Problem
            The solved OpenMDAO problem
        """
        # Enable warm-start if we have a solution guess for better convergence
        warm_start = self.guess_solution is not None
        if warm_start:
            print("Enabling IPOPT warm-start settings for solution guess")

        prob, phase = get_dymos_vectorized_solver_problem(self.bodies, num_nodes=num_nodes, warm_start=warm_start)

        # Add design variables with defaults
        for var_name, dv in self.get_design_variables_with_defaults().items():
            kwargs = {}
            if dv.lower is not None:
                kwargs['lower'] = dv.lower
            if dv.upper is not None:
                kwargs['upper'] = dv.upper
            if dv.ref is not None:
                kwargs['ref'] = dv.ref
            if dv.ref0 is not None:
                kwargs['ref0'] = dv.ref0
            if dv.scaler is not None:
                kwargs['scaler'] = dv.scaler
            if dv.adder is not None:
                kwargs['adder'] = dv.adder
            if dv.indices is not None:
                kwargs['indices'] = dv.indices
            if dv.units is not None:
                kwargs['units'] = dv.units
            prob.model.add_design_var(var_name, **kwargs)

        # Add general constraints with defaults
        for con_name, con in self.get_general_constraints_with_defaults().items():
            kwargs = {}
            if con.lower is not None:
                kwargs['lower'] = con.lower
            if con.upper is not None:
                kwargs['upper'] = con.upper
            if con.equals is not None:
                kwargs['equals'] = con.equals
            if con.ref is not None:
                kwargs['ref'] = con.ref
            if con.ref0 is not None:
                kwargs['ref0'] = con.ref0
            if con.indices is not None:
                kwargs['indices'] = con.indices
            if con.units is not None:
                kwargs['units'] = con.units
            prob.model.add_constraint(con_name, **kwargs)

        # Add boundary constraints with defaults
        for con_name, con in self.get_boundary_constraints_with_defaults().items():
            kwargs = {'loc': con.loc}
            if con.lower is not None:
                kwargs['lower'] = con.lower
            if con.upper is not None:
                kwargs['upper'] = con.upper
            if con.equals is not None:
                kwargs['equals'] = con.equals
            if con.ref is not None:
                kwargs['ref'] = con.ref
            if con.ref0 is not None:
                kwargs['ref0'] = con.ref0
            if con.indices is not None:
                kwargs['indices'] = con.indices
            if con.units is not None:
                kwargs['units'] = con.units
            phase.add_boundary_constraint(con_name, **kwargs)

        # Add path constraints with defaults
        for con_name, con in self.get_path_constraints_with_defaults().items():
            kwargs = {}
            if con.lower is not None:
                kwargs['lower'] = con.lower
            if con.upper is not None:
                kwargs['upper'] = con.upper
            if con.equals is not None:
                kwargs['equals'] = con.equals
            if con.ref is not None:
                kwargs['ref'] = con.ref
            if con.ref0 is not None:
                kwargs['ref0'] = con.ref0
            if con.indices is not None:
                kwargs['indices'] = con.indices
            if con.units is not None:
                kwargs['units'] = con.units
            phase.add_path_constraint(con_name, **kwargs)

        # Add objectives with defaults
        for obj_name, obj in self.get_objectives_with_defaults().items():
            kwargs = {}
            if obj.ref is not None:
                kwargs['ref'] = obj.ref
            if obj.ref0 is not None:
                kwargs['ref0'] = obj.ref0
            if obj.scaler is not None:
                kwargs['scaler'] = obj.scaler
            if obj.adder is not None:
                kwargs['adder'] = obj.adder
            if obj.index is not None:
                kwargs['index'] = obj.index
            if obj.units is not None:
                kwargs['units'] = obj.units
            prob.model.add_objective(obj_name, **kwargs)

        # Setup the problem
        prob.setup()

        # Set initial guess values
        N = len(self.bodies)

        # Compute dt values from flyby_times
        dt = [self.flyby_times[0] - self.t0]
        for i in range(1, len(self.flyby_times)):
            dt.append(self.flyby_times[i] - self.flyby_times[i-1])

        # Set t0 and dt
        prob.set_val('t0', self.t0, units='gtoc_year')
        prob.set_val('dt', dt, units='gtoc_year')

        # Get body positions and velocities at flyby times
        # Convert flyby times to seconds for get_state
        from gtoc13.constants import YEAR
        flyby_times_s = [t * YEAR for t in self.flyby_times]

        # Set initial guess for positions and velocities for each arc
        if self.guess_solution is not None:
            # Use the guess solution to extract states and controls
            from gtoc13.solution import PropagatedArc, FlybyArc, ConicArc
            print("Using guess_solution for initial guess")

            # Extract propagated arcs (the ones between flybys)
            propagated_arcs = [arc for arc in self.guess_solution.arcs if isinstance(arc, PropagatedArc)]
            n_guess_arcs = len(propagated_arcs)

            if n_guess_arcs > 0 and n_guess_arcs <= N:
                # We have some guess data - use it for the first n_guess_arcs
                print(f"Using solution guess for first {n_guess_arcs} of {N} arcs")

                r_initial = []
                r_final = []
                v_initial = []
                v_final = []
                u_n_vals = []

                # Extract all state points from solution for proper interpolation
                # We need to build arrays of shape (num_time_points, N, 3)
                # where num_time_points is the number of state points in each arc (should be same for all)

                # First, check if all arcs have the same number of state points
                num_pts_per_arc = [len(arc.state_points) for arc in propagated_arcs[:n_guess_arcs]]
                if len(set(num_pts_per_arc)) == 1:
                    # All arcs have same number of points - we can use the full solution
                    num_time_pts = num_pts_per_arc[0]

                    # Build arrays: (num_time_pts, N, 3) for positions and velocities
                    # and corresponding time array
                    r_guess = np.zeros((num_time_pts, N, 3))
                    v_guess = np.zeros((num_time_pts, N, 3))
                    u_n_guess = np.zeros((num_time_pts, N, 3))
                    t_guess = np.zeros((num_time_pts, N))

                    # Fill in data from solution for first n_guess_arcs
                    for i in range(n_guess_arcs):
                        arc = propagated_arcs[i]
                        for j, pt in enumerate(arc.state_points):
                            r_guess[j, i, :] = pt.position
                            v_guess[j, i, :] = pt.velocity
                            u_n_guess[j, i, :] = pt.control
                            t_guess[j, i] = pt.epoch

                    # Fill in interpolated data for remaining arcs
                    for i in range(n_guess_arcs, N):
                        body_id = self.bodies[i]
                        t_flyby_s = flyby_times_s[i]
                        r_body, v_body = bodies_data[body_id].get_state(t_flyby_s)

                        # Just use same value at all time points (will be linearly interpolated by dymos)
                        for j in range(num_time_pts):
                            r_guess[j, i, :] = r_body
                            v_guess[j, i, :] = v_body
                            u_n_guess[j, i, :] = [0.0, 0.0, 0.0]
                            t_guess[j, i] = t_flyby_s

                    use_detailed_guess = True
                else:
                    # Arc lengths don't match - fall back to simple endpoint guess
                    print(f"Warning: Arcs have different numbers of state points: {num_pts_per_arc}")
                    print(f"Falling back to endpoint guess")
                    use_detailed_guess = False

                    # Build simple endpoint guess
                    for i in range(n_guess_arcs):
                        arc = propagated_arcs[i]
                        r_initial.append(list(arc.state_points[0].position))
                        r_final.append(list(arc.state_points[-1].position))
                        v_initial.append(list(arc.state_points[0].velocity))
                        v_final.append(list(arc.state_points[-1].velocity))
                        u_n_vals.append(list(arc.state_points[0].control))

                # Set state values for the phase
                if use_detailed_guess:
                    # Use the detailed guess with all time points
                    # Average times across arcs (they should be similar)
                    t_avg = np.mean(t_guess, axis=1)

                    phase.set_state_val('r', vals=r_guess, time_vals=t_avg, units='km')
                    phase.set_state_val('v', vals=v_guess, time_vals=t_avg, units='km/s')
                    phase.set_control_val('u_n', vals=u_n_guess, time_vals=t_avg, units='unitless')

                    print(f"Set detailed initial guess with {num_time_pts} time points per arc")
                else:
                    # Use endpoint guess - need to finish building it for remaining arcs
                    if n_guess_arcs < N and not use_detailed_guess:
                        print(f"Using default guess for remaining {N - n_guess_arcs} arcs")
                        for i in range(n_guess_arcs, N):
                            body_id = self.bodies[i]
                            t_flyby_s = flyby_times_s[i]
                            r_body, v_body = bodies_data[body_id].get_state(t_flyby_s)

                            r_initial.append(r_body)
                            r_final.append(r_body)
                            v_initial.append(v_body)
                            v_final.append(v_body)
                            u_n_vals.append([0.0, 0.0, 0.0])

                    # Set with endpoint interpolation
                    phase.set_state_val('r', vals=[r_initial, r_final], units='km')
                    phase.set_state_val('v', vals=[v_initial, v_final], units='km/s')

                    u_n = np.array(u_n_vals)
                    if phase.control_options['u_n']['opt']:
                        for i in range(n_guess_arcs, N):
                            u_n[i, 0] = 1.0
                    phase.set_control_val('u_n', [u_n, u_n])
            else:
                print(f"Warning: Number of propagated arcs ({n_guess_arcs}) is incompatible with "
                      f"number of bodies ({N}). Using default guess.")
                # Fall back to default guess
                r_initial = []
                r_final = []
                v_initial = []
                v_final = []

                for i, (body_id, t_flyby_s, dt_i) in enumerate(zip(self.bodies, flyby_times_s, dt)):
                    # Get body state at flyby time
                    r_body, v_body = bodies_data[body_id].get_state(t_flyby_s)

                    r_initial.append(r_body)
                    r_final.append(r_body)
                    v_initial.append(v_body)
                    v_final.append(v_body)

                # Set state values for the phase
                phase.set_state_val('r', vals=[r_initial, r_final], units='km')
                phase.set_state_val('v', vals=[v_initial, v_final], units='km/s')

                # Set control values (ballistic trajectory by default)
                u_n = np.zeros((N, 3))
                if phase.control_options['u_n']['opt']:
                    u_n[:, 0] = 1.0
                phase.set_control_val('u_n', [u_n, u_n])
        else:
            # No guess solution - use default guess
            r_initial = []
            r_final = []
            v_initial = []
            v_final = []

            for i, (body_id, t_flyby_s, dt_i) in enumerate(zip(self.bodies, flyby_times_s, dt)):
                # Get body state at flyby time
                r_body, v_body = bodies_data[body_id].get_state(t_flyby_s)

                r_initial.append(r_body)
                r_final.append(r_body)
                v_initial.append(v_body)
                v_final.append(v_body)

            # Set state values for the phase
            phase.set_state_val('r', vals=[r_initial, r_final], units='km')
            phase.set_state_val('v', vals=[v_initial, v_final], units='km/s')

            # Set control values (ballistic trajectory by default)
            u_n = np.zeros((N, 3))
            if phase.control_options['u_n']['opt']:
                u_n[:, 0] = 1.0
            phase.set_control_val('u_n', [u_n, u_n])

        # Set time values (nondimensional time from -1 to 1)
        phase.set_time_val(initial=-1.0, duration=2.0, units='unitless')

        # Set parameter values (dt_dtau = dt/2)
        phase.set_parameter_val('dt_dtau', np.asarray(dt) / 2., units='gtoc_year')

        dm.run_problem(prob, run_driver=run_driver, simulate=False)

        return prob
