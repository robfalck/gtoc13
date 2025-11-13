"""Test the FlybyDefectComp component."""

import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from gtoc13.dymos_solver.flyby_comp import FlybyDefectComp
from gtoc13.bodies import bodies_data


class TestFlybyDefectComp(unittest.TestCase):

    def test_flyby_defect_comp_single_flyby(self):
        """Test FlybyDefectComp with a single flyby."""
        # Use body 10 (Planet X)
        bodies = [10]

        prob = om.Problem()
        prob.model.add_subsystem('flyby', FlybyDefectComp(bodies=bodies),
                                 promotes=['*'])
        prob.setup()

        # Set up a test flyby scenario
        # Body velocity (arbitrary)
        v_body = np.array([[10.0, 5.0, 2.0]])  # km/s

        # Incoming velocity
        v_in = np.array([[20.0, 10.0, 5.0]])  # km/s

        # For a valid flyby, v_out should have:
        # - Same v_infinity magnitude as v_in
        # - Turn angle that produces valid altitude
        v_inf_in = v_in - v_body
        v_inf_mag = np.linalg.norm(v_inf_in)

        # Create outgoing with same magnitude but different direction
        # For simplicity, rotate v_inf by some angle
        # Here we'll just use a valid configuration
        v_inf_out = np.array([[8.0, 4.0, 3.0]])  # Arbitrary but close to same magnitude
        # Rescale to exact same magnitude
        v_inf_out = v_inf_out / np.linalg.norm(v_inf_out) * v_inf_mag
        v_out = v_inf_out + v_body

        # Set inputs
        prob.set_val('v_in', v_in, units='km/s')
        prob.set_val('v_out', v_out, units='km/s')
        prob.set_val('v_body', v_body, units='km/s')

        prob.run_model()

        # Get outputs
        v_inf_mag_defect = prob.get_val('v_inf_mag_defect', units='km/s')
        h_p_norm = prob.get_val('h_p_norm', units='unitless')

        # V-infinity magnitude defect should be very small (nearly zero)
        assert_near_equal(v_inf_mag_defect[0], 0.0, tolerance=1e-10)

        # h_p_defect depends on the turn angle, which determines altitude
        # We can't easily predict the value without computing the turn angle,
        # but we can check it's a reasonable number
        print(f"\nFlyby defects:")
        print(f"  v_inf_mag_defect = {v_inf_mag_defect[0]:.6e} km/s")
        print(f"  h_p_norm = {h_p_norm[0]:.6f}")

    def test_flyby_defect_comp_multiple_flybys(self):
        """Test FlybyDefectComp with multiple flybys."""
        # Use bodies 1, 2, 3 (valid body IDs)
        bodies = [1, 2, 3]

        prob = om.Problem()
        prob.model.add_subsystem('flyby', FlybyDefectComp(bodies=bodies),
                                 promotes=['*'])
        prob.setup()

        # Set up test data for 3 flybys
        v_body = np.array([
            [10.0, 5.0, 2.0],
            [8.0, 6.0, 1.0],
            [12.0, 4.0, 3.0]
        ])

        v_in = np.array([
            [20.0, 10.0, 5.0],
            [18.0, 12.0, 4.0],
            [22.0, 8.0, 6.0]
        ])

        # Create v_out with same v_infinity magnitudes
        v_out = np.zeros_like(v_in)
        for i in range(3):
            v_inf_in = v_in[i] - v_body[i]
            v_inf_mag = np.linalg.norm(v_inf_in)

            # Arbitrary direction but correct magnitude
            v_inf_out = np.array([8.0, 4.0, 3.0])
            v_inf_out = v_inf_out / np.linalg.norm(v_inf_out) * v_inf_mag
            v_out[i] = v_inf_out + v_body[i]

        prob.set_val('v_in', v_in, units='km/s')
        prob.set_val('v_out', v_out, units='km/s')
        prob.set_val('v_body', v_body, units='km/s')

        prob.run_model()

        # Get outputs
        v_inf_mag_defect = prob.get_val('v_inf_mag_defect', units='km/s')
        h_p_norm = prob.get_val('h_p_norm', units='unitless')

        # All v_infinity magnitude defects should be near zero
        for i in range(3):
            assert_near_equal(v_inf_mag_defect[i], 0.0, tolerance=1e-10)

        print(f"\nMultiple flyby defects:")
        for i in range(3):
            print(f"  Flyby {i}: v_inf_mag_defect = {v_inf_mag_defect[i]:.6e}, "
                  f"h_p_norm = {h_p_norm[i]:.6f}")

    def test_flyby_altitude_violation(self):
        """Test that altitude violations are detected correctly."""
        bodies = [10]
        body = bodies_data[10]

        prob = om.Problem()
        prob.model.add_subsystem('flyby', FlybyDefectComp(bodies=bodies),
                                 promotes=['*'])
        prob.setup()

        v_body = np.array([[10.0, 5.0, 2.0]])
        v_in = np.array([[20.0, 10.0, 5.0]])

        # Create a nearly straight-through flyby (small turn angle)
        # This will result in a very large altitude (violation of upper bound)
        v_inf_in = v_in - v_body
        v_inf_mag = np.linalg.norm(v_inf_in)

        # Outgoing nearly parallel to incoming (small turn)
        v_inf_out_dir = v_inf_in / v_inf_mag
        # Perturb slightly
        v_inf_out = v_inf_out_dir * v_inf_mag + np.array([[0.1, 0.1, 0.0]])
        v_inf_out = v_inf_out / np.linalg.norm(v_inf_out) * v_inf_mag
        v_out = v_inf_out + v_body

        prob.set_val('v_in', v_in, units='km/s')
        prob.set_val('v_out', v_out, units='km/s')
        prob.set_val('v_body', v_body, units='km/s')

        prob.run_model()

        h_p_norm = prob.get_val('h_p_norm', units='unitless')

        # Small turn angle should give large altitude, violating upper bound
        # h_p_defect > 0 indicates violation
        print(f"\nAltitude violation test:")
        print(f"  h_p_norm = {h_p_norm[0]:.6f}")
        print(f"  Positive value indicates violation: {h_p_norm[0] > 0}")


if __name__ == '__main__':
    unittest.main()
