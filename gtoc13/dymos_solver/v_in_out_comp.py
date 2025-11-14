import jax.numpy as jnp

import openmdao.api as om


class VInOutComp(om.ExplicitComponent):
    """
    Component that concatenates all of the intial and final arc velocities for the
    flyby component.
    """

    def initialize(self):
        self.options.declare('N', types=(int,), desc='Number of bodies in the solution')

    def setup(self):
        N = self.options['N']

        # The final inertial velocity after the last flyby
        self.add_input('v_end', shape=(1, 3), units='km/s')

        # The initial velocity of all but the first arc. These are the inertial
        # outgoing velocities for each flyby calc, except the final one.
        for i in range(N):
            if i > 0:
                self.add_input(f'arc_{i}_v_initial',
                               shape=(1, 3),
                               units='km/s')
            self.add_input(f'arc_{i}_v_final',
                           shape=(1, 3),
                           units='km/s')

        self.add_output('flyby_v_in', shape=(N, 3), units='km/s')
        self.add_output('flyby_v_out', shape=(N, 3), units='km/s')

    def setup_partials(self):
        N = self.options['N']

        # Partials for flyby_v_in (shape: N x 3)
        # flyby_v_in[i, :] = arc_i_v_final[0, :] for all i
        for i in range(N):
            # Each row of flyby_v_in comes from arc_i_v_final
            # arc_i_v_final has shape (1, 3), so flattened indices are 0, 1, 2
            rows = [i * 3, i * 3 + 1, i * 3 + 2]  # Three elements for x, y, z of flyby_v_in[i]
            cols = [0, 1, 2]  # All three elements from arc_i_v_final[0, :]
            self.declare_partials(of='flyby_v_in', wrt=f'arc_{i}_v_final',
                                  rows=rows, cols=cols, val=1.0)

        # Partials for flyby_v_out (shape: N x 3)
        # flyby_v_out[i-1, :] = arc_i_v_initial[0, :] for i > 0
        for i in range(1, N):
            # arc_i_v_initial has shape (1, 3), so flattened indices are 0, 1, 2
            rows = [(i - 1) * 3, (i - 1) * 3 + 1, (i - 1) * 3 + 2]
            cols = [0, 1, 2]
            self.declare_partials(of='flyby_v_out', wrt=f'arc_{i}_v_initial',
                                  rows=rows, cols=cols, val=1.0)

        # flyby_v_out[-1, :] = v_end[0, :]
        # v_end has shape (1, 3), so flattened indices are 0, 1, 2
        rows = [(N - 1) * 3, (N - 1) * 3 + 1, (N - 1) * 3 + 2]
        cols = [0, 1, 2]
        self.declare_partials(of='flyby_v_out', wrt='v_end',
                              rows=rows, cols=cols, val=1.0)

    def compute(self, inputs, outputs):
        N = self.options['N']
        for i in range(N):
            if i > 0:
                outputs['flyby_v_out'][i - 1, ...] = inputs[f'arc_{i}_v_initial']
            outputs['flyby_v_in'][i, ...] = inputs[f'arc_{i}_v_final']
        outputs['flyby_v_out'][-1, ...] = inputs['v_end']



class SingleArcVInOutComp(om.ExplicitComponent):
    """
    Component that concatenates all of the intial and final arc velocities for the
    flyby component in a single arc problem.
    """
    def setup(self):
        # The initial inertial velocity before the first flyby
        self.add_input('v_in_prev_flyby', shape=(1, 3), units='km/s')

        # The final inertial velocity after the last flyby
        self.add_input('v_end', shape=(1, 3), units='km/s')

        self.add_input('arc_0_v_initial', shape=(1, 3), units='km/s')
        self.add_input('arc_0_v_final', shape=(1, 3), units='km/s')

        self.add_output('flyby_v_in', shape=(2, 3), units='km/s')
        self.add_output('flyby_v_out', shape=(2, 3), units='km/s')

    def declare_partials(self):
        # flyby_v_in[0, :] = v_in_prev_flyby[0, :]
        # Maps indices 0-2 of output to indices 0-2 of input
        self.declare_partials('flyby_v_in', 'v_in_prev_flyby',
                              rows=[0, 1, 2],
                              cols=[0, 1, 2],
                              val=1.0)

        # flyby_v_in[1, :] = arc_0_v_final[0, :]
        # Maps indices 3-5 of output to indices 0-2 of input
        self.declare_partials('flyby_v_in', 'arc_0_v_final',
                              rows=[3, 4, 5],
                              cols=[0, 1, 2],
                              val=1.0)

        # flyby_v_out[0, :] = arc_0_v_initial[0, :]
        # Maps indices 0-2 of output to indices 0-2 of input
        self.declare_partials('flyby_v_out', 'arc_0_v_initial',
                              rows=[0, 1, 2],
                              cols=[0, 1, 2],
                              val=1.0)

        # flyby_v_out[1, :] = v_end[0, :]
        # Maps indices 3-5 of output to indices 0-2 of input
        self.declare_partials('flyby_v_out', 'v_end',
                              rows=[3, 4, 5],
                              cols=[0, 1, 2],
                              val=1.0)

    def compute(self, inputs, outputs):
        outputs['flyby_v_in'][0, ...] = inputs['v_in_prev_flyby']
        outputs['flyby_v_in'][1, ...] = inputs['arc_0_v_final']

        outputs['flyby_v_out'][0, ...] = inputs['arc_0_v_initial']
        outputs['flyby_v_out'][1, ...] = inputs['v_end']