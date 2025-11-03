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
        prob.model.add_subsystem('ephem', EphemComp(units='km/s', bodies=[10]))

        prob.setup()

        prob.set_val('ephem.times', [0.0, 0.0])

        prob.run_model()

        # prob.model.list_vars(print_arrays=True)

        r = prob.get_val('ephem.body_pos')
        v = prob.get_val('ephem.body_vel')

        r_check, v_check = bodies_data[10].get_state(0.0)
        r_check = np.reshape(r_check, (1, 3))
        v_check = np.reshape(v_check, (1, 3))

        assert_near_equal(r, r_check)
        assert_near_equal(v, v_check)

    def test_ephem_vec(self):
        body_sequence = [10, 9]
        times = [0.0, 10.0, 20.0]

        prob = om.Problem()
        prob.model.add_subsystem('ephem', EphemComp(units='km/s', bodies=body_sequence))

        prob.setup()

        prob.set_val('ephem.times', times, units='year')

        prob.run_model()

        # prob.model.list_vars(print_arrays=True)

        r = prob.get_val('ephem.body_pos')
        v = prob.get_val('ephem.body_vel')
        times = prob.get_val('ephem.times')

        for i, body_id in enumerate(body_sequence):
            r_check, v_check = bodies_data[body_id].get_state(times[i+1])
            assert_near_equal(r[i, :], r_check)
            assert_near_equal(v[i, :], v_check)
        

if __name__ == '__main__':
    unittest.main()
