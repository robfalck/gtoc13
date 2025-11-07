"""Tests for MissionPlan validation and functionality."""
import unittest
from pathlib import Path
import tempfile
import json

from pydantic import ValidationError

from gtoc13.dymos_solver.mission_plan import (
    MissionPlan,
    DesignVariable,
    GeneralConstraint,
    BoundaryConstraint,
    PathConstraint,
    Objective,
)


class TestMissionPlanValidation(unittest.TestCase):
    """Test validation logic for MissionPlan."""

    def test_valid_mission_plan(self):
        """Test that a valid mission plan can be created."""
        plan = MissionPlan(bodies=[10], flyby_times=[20.0], t0=0.0)
        self.assertEqual(plan.bodies, [10])
        self.assertEqual(plan.flyby_times, [20.0])
        self.assertEqual(plan.t0, 0.0)

    def test_valid_multiple_bodies(self):
        """Test valid mission plan with multiple bodies."""
        plan = MissionPlan(
            bodies=[10, 9, 8],
            flyby_times=[20.0, 40.0, 60.0],
            t0=0.0
        )
        self.assertEqual(len(plan.bodies), 3)
        self.assertEqual(len(plan.flyby_times), 3)

    def test_t0_below_minimum(self):
        """Test that t0 < 0.0 raises ValueError."""
        with self.assertRaises(ValidationError) as cm:
            MissionPlan(bodies=[10], flyby_times=[20.0], t0=-1.0)

        error_msg = str(cm.exception)
        self.assertIn("t0 must be between 0.0 and 200.0", error_msg)

    def test_t0_above_maximum(self):
        """Test that t0 > 200.0 raises ValueError."""
        with self.assertRaises(ValidationError) as cm:
            MissionPlan(bodies=[10], flyby_times=[220.0], t0=201.0)

        error_msg = str(cm.exception)
        self.assertIn("t0 must be between 0.0 and 200.0", error_msg)

    def test_t0_at_boundaries(self):
        """Test that t0 at 0.0 and 200.0 are valid."""
        plan1 = MissionPlan(bodies=[10], flyby_times=[0.5], t0=0.0)
        self.assertEqual(plan1.t0, 0.0)

        plan2 = MissionPlan(bodies=[10], flyby_times=[199.5], t0=199.0)
        self.assertEqual(plan2.t0, 199.0)

    def test_non_monotonic_flyby_times(self):
        """Test that non-monotonic flyby_times raises ValueError."""
        with self.assertRaises(ValidationError) as cm:
            MissionPlan(bodies=[10, 9], flyby_times=[20.0, 15.0], t0=0.0)

        error_msg = str(cm.exception)
        self.assertIn("flyby_times must be monotonically increasing", error_msg)
        self.assertIn("15.0 <= 20.0", error_msg)

    def test_equal_flyby_times(self):
        """Test that equal consecutive flyby_times raises ValueError."""
        with self.assertRaises(ValidationError) as cm:
            MissionPlan(bodies=[10, 9], flyby_times=[20.0, 20.0], t0=0.0)

        error_msg = str(cm.exception)
        self.assertIn("flyby_times must be monotonically increasing", error_msg)

    def test_flyby_time_not_greater_than_t0(self):
        """Test that first flyby_time <= t0 raises ValueError."""
        with self.assertRaises(ValidationError) as cm:
            MissionPlan(bodies=[10], flyby_times=[5.0], t0=10.0)

        error_msg = str(cm.exception)
        self.assertIn("First flyby time", error_msg)
        self.assertIn("must be greater than t0", error_msg)

    def test_flyby_time_equal_to_t0(self):
        """Test that first flyby_time == t0 raises ValueError."""
        with self.assertRaises(ValidationError) as cm:
            MissionPlan(bodies=[10], flyby_times=[10.0], t0=10.0)

        error_msg = str(cm.exception)
        self.assertIn("must be greater than t0", error_msg)

    def test_bodies_flyby_times_length_mismatch(self):
        """Test that mismatched lengths raise ValueError."""
        with self.assertRaises(ValidationError) as cm:
            MissionPlan(bodies=[10, 9], flyby_times=[20.0], t0=0.0)

        error_msg = str(cm.exception)
        self.assertIn("bodies and flyby_times must have the same length", error_msg)
        self.assertIn("2 bodies", error_msg)
        self.assertIn("1 flyby_times", error_msg)

    def test_empty_bodies(self):
        """Test that empty bodies list raises ValueError."""
        with self.assertRaises(ValidationError) as cm:
            MissionPlan(bodies=[], flyby_times=[], t0=0.0)

        error_msg = str(cm.exception)
        self.assertIn("bodies must contain at least one element", error_msg)

    def test_empty_flyby_times(self):
        """Test that empty flyby_times list raises ValueError."""
        with self.assertRaises(ValidationError) as cm:
            MissionPlan(bodies=[10], flyby_times=[], t0=0.0)

        error_msg = str(cm.exception)
        # This will fail due to length mismatch
        self.assertIn("must have the same length", error_msg)


class TestMissionPlanSaveLoad(unittest.TestCase):
    """Test save/load functionality for MissionPlan."""

    def test_save_and_load(self):
        """Test that a mission plan can be saved and loaded."""
        plan = MissionPlan(
            bodies=[10, 9],
            flyby_times=[20.0, 40.0],
            t0=5.0
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_plan.pln'
            plan.save(filepath)

            # Verify file exists
            self.assertTrue(filepath.exists())

            # Load and verify
            loaded_plan = MissionPlan.load(filepath)
            self.assertEqual(loaded_plan.bodies, plan.bodies)
            self.assertEqual(loaded_plan.flyby_times, plan.flyby_times)
            self.assertEqual(loaded_plan.t0, plan.t0)

    def test_save_adds_pln_extension(self):
        """Test that save adds .pln extension if not present."""
        plan = MissionPlan(bodies=[10], flyby_times=[20.0], t0=0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_plan'  # No extension
            plan.save(filepath)

            # Verify .pln extension was added
            expected_path = Path(tmpdir) / 'test_plan.pln'
            self.assertTrue(expected_path.exists())

    def test_save_with_constraints_and_design_vars(self):
        """Test saving/loading with constraints and design variables."""
        plan = MissionPlan(
            bodies=[10],
            flyby_times=[20.0],
            t0=0.0,
            design_variables={
                'dt': DesignVariable(lower=5.0, upper=50.0, units='gtoc_year')
            },
            general_constraints={
                'times': GeneralConstraint(indices=[-1], upper=100.0, units='gtoc_year')
            },
            boundary_constraints={
                'v': BoundaryConstraint(loc='initial', indices=[0], equals=0.0)
            },
            path_constraints={
                'u_n_norm': PathConstraint(equals=1.0)
            },
            objectives={
                'E_end': Objective(scaler=-1.0)
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_plan.pln'
            plan.save(filepath)

            loaded_plan = MissionPlan.load(filepath)

            # Verify design variables
            self.assertIn('dt', loaded_plan.design_variables)
            self.assertEqual(loaded_plan.design_variables['dt'].lower, 5.0)
            self.assertEqual(loaded_plan.design_variables['dt'].upper, 50.0)

            # Verify constraints
            self.assertIn('times', loaded_plan.general_constraints)
            self.assertEqual(loaded_plan.general_constraints['times'].upper, 100.0)

            # Verify boundary constraints
            self.assertIn('v', loaded_plan.boundary_constraints)
            self.assertEqual(loaded_plan.boundary_constraints['v'].loc, 'initial')

            # Verify path constraints
            self.assertIn('u_n_norm', loaded_plan.path_constraints)
            self.assertEqual(loaded_plan.path_constraints['u_n_norm'].equals, 1.0)

            # Verify objectives
            self.assertIn('E_end', loaded_plan.objectives)
            self.assertEqual(loaded_plan.objectives['E_end'].scaler, -1.0)


class TestMissionPlanDefaults(unittest.TestCase):
    """Test default values for constraints and design variables."""

    def test_default_design_variables(self):
        """Test that default design variables are correct."""
        defaults = MissionPlan.get_default_design_variables()

        self.assertIn('t0', defaults)
        self.assertIn('dt', defaults)
        self.assertIn('y0', defaults)
        self.assertIn('z0', defaults)
        self.assertIn('v_end', defaults)

        self.assertEqual(defaults['t0'].lower, 0.0)
        self.assertEqual(defaults['dt'].lower, 0.0)
        self.assertEqual(defaults['dt'].upper, 200)

    def test_default_general_constraints(self):
        """Test that default general constraints are correct."""
        defaults = MissionPlan.get_default_general_constraints()

        self.assertIn('flyby_comp.v_inf_mag_defect', defaults)
        self.assertIn('flyby_comp.h_p_defect', defaults)
        self.assertIn('times', defaults)

        self.assertEqual(defaults['flyby_comp.v_inf_mag_defect'].equals, 0.0)
        self.assertEqual(defaults['flyby_comp.h_p_defect'].upper, 0.0)
        self.assertEqual(defaults['times'].upper, 199.999)

    def test_default_boundary_constraints(self):
        """Test that default boundary constraints are correct."""
        defaults = MissionPlan.get_default_boundary_constraints()

        self.assertIn('v', defaults)
        self.assertEqual(defaults['v'].loc, 'initial')
        self.assertEqual(defaults['v'].indices, [1, 2])
        self.assertEqual(defaults['v'].equals, 0.0)

    def test_default_objectives(self):
        """Test that default objectives are correct."""
        defaults = MissionPlan.get_default_objectives()

        self.assertIn('E_end', defaults)

    def test_get_design_variables_with_defaults(self):
        """Test merging custom design variables with defaults."""
        plan = MissionPlan(
            bodies=[10],
            flyby_times=[20.0],
            t0=0.0,
            design_variables={
                'dt': DesignVariable(lower=10.0, upper=100.0, units='gtoc_year')
            }
        )

        dvs = plan.get_design_variables_with_defaults()

        # Custom dt should override default
        self.assertEqual(dvs['dt'].lower, 10.0)
        self.assertEqual(dvs['dt'].upper, 100.0)

        # Other defaults should remain
        self.assertIn('t0', dvs)
        self.assertIn('y0', dvs)
        self.assertIn('z0', dvs)
        self.assertIn('v_end', dvs)

    def test_remove_default_with_none(self):
        """Test that setting a design variable to None removes it."""
        plan = MissionPlan(
            bodies=[10],
            flyby_times=[20.0],
            t0=0.0,
            design_variables={
                'dt': None  # Remove dt from defaults
            }
        )

        dvs = plan.get_design_variables_with_defaults()

        # dt should be removed
        self.assertNotIn('dt', dvs)

        # Other defaults should remain
        self.assertIn('t0', dvs)
        self.assertIn('y0', dvs)


class TestMissionPlanEdgeCases(unittest.TestCase):
    """Test edge cases for MissionPlan."""

    def test_single_body_mission(self):
        """Test mission plan with single body."""
        plan = MissionPlan(bodies=[10], flyby_times=[20.0], t0=0.0)
        self.assertEqual(len(plan.bodies), 1)

    def test_many_bodies_mission(self):
        """Test mission plan with many bodies."""
        bodies = list(range(10, 20))  # 10 bodies
        flyby_times = [float(i * 10) for i in range(1, 11)]  # 10, 20, ..., 100

        plan = MissionPlan(bodies=bodies, flyby_times=flyby_times, t0=0.0)
        self.assertEqual(len(plan.bodies), 10)

    def test_near_boundary_times(self):
        """Test mission plan with times near boundaries."""
        plan = MissionPlan(
            bodies=[10],
            flyby_times=[199.99],
            t0=199.0
        )
        self.assertEqual(plan.t0, 199.0)
        self.assertEqual(plan.flyby_times[0], 199.99)

    def test_small_time_differences(self):
        """Test mission plan with small time differences between flybys."""
        plan = MissionPlan(
            bodies=[10, 9, 8],
            flyby_times=[20.0, 20.1, 20.2],
            t0=0.0
        )
        self.assertEqual(plan.flyby_times, [20.0, 20.1, 20.2])


if __name__ == '__main__':
    unittest.main()
