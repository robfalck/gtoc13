import unittest

import numpy as np
import jax.numpy as jnp

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from gtoc13.bodies import bodies_data
from gtoc13.dymos_solver.ephem_comp import EphemComp


class TestEpehemComp(unittest.TestCase):

    def test_ephem_single(self):

        prob = om.Problem()
        prob.model.add_subsystem('ephem', EphemComp(bodies=[10]))

        prob.setup()

        prob.set_val('ephem.dt', [0.0])

        prob.run_model()

        # prob.model.list_vars(print_arrays=True)

        r = prob.get_val('ephem.event_pos')
        v = prob.get_val('ephem.body_vel')  # Changed from event_vel to body_vel
        times_s = prob.get_val('ephem.times', units='s')  # Get times in seconds for get_state

        r_check, v_check = bodies_data[10].get_state(times_s[1])  # Default time_units is seconds

        assert_near_equal(r[1, :], r_check, tolerance=1e-12)
        assert_near_equal(v[0, :], v_check, tolerance=1e-12)

    def test_ephem_vec(self):
        body_sequence = [10, 9]
        dt = [10, 10]

        prob = om.Problem()
        prob.model.add_subsystem('ephem', EphemComp(bodies=body_sequence))

        prob.setup()

        prob.set_val('ephem.dt', dt, units='gtoc_year')

        prob.run_model()

        # prob.model.list_vars(print_arrays=True)

        r = prob.get_val('ephem.event_pos')
        v = prob.get_val('ephem.body_vel')
        times_s = prob.get_val('ephem.times', units='s')  # Get times in seconds for get_state

        for i, body_id in enumerate(body_sequence):
            r_check, v_check = bodies_data[body_id].get_state(times_s[i+1])  # Default time_units is seconds
            assert_near_equal(r[i+1, :], r_check, tolerance=1e-12)
            assert_near_equal(v[i, :], v_check, tolerance=1e-12)  # Changed from v[i+1, :] to v[i, :]
        

if __name__ == '__main__':
    unittest.main()
