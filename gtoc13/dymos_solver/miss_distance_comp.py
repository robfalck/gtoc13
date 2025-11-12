import openmdao.api as om

class MissDisanceComp(om.ExplicitComponent):

    def initialize(self):
        """Declare options."""
        self.options.declare('N', types=int,
                             desc='Number of trajectory arcs')

    def setup(self):
        """Set up inputs and outputs."""
        N = self.options['N']

        # Inputs: velocities at each flyby
        self.add_input('event_pos', shape=(N + 1, 3), units='DU',
                       desc='Endpoint locations of each arc.')

        for i in range(N):
            self.add_input(f'arc_{i}_r_initial', shape=(1, 3), units='DU')
            self.add_input(f'arc_{i}_r_final', shape=(1, 3), units='DU')

        # Outputs: defects for each flyby
        self.add_output('r_error', shape=(N, 2, 3), units='DU',
                        desc='Initial and final miss distance for each arc')

    def setup_partials(self):
        """
        Declare partial derivatives.

        All partials are linear (just +1 or -1), so we can declare them sparsely
        using row/column indices.

        For each arc i:
            r_error[i, 0, :] = arc_i_r_initial - event_pos[i, :]
            r_error[i, 1, :] = arc_i_r_final - event_pos[i+1, :]
        """
        N = self.options['N']

        # Collect all rows/cols for event_pos partials (declare once)
        event_pos_rows = []
        event_pos_cols = []

        # For each arc
        for i in range(N):
            # Partials of r_error[i, 0, :] w.r.t. arc_i_r_initial
            # r_error[i, 0, :] has flat indices: [i*6 + 0, i*6 + 1, i*6 + 2]
            # arc_i_r_initial has shape (1, 3), flat indices: [0, 1, 2]
            rows_initial = [i * 6 + 0, i * 6 + 1, i * 6 + 2]
            cols_initial = [0, 1, 2]
            self.declare_partials(f'r_error', f'arc_{i}_r_initial',
                                  rows=rows_initial, cols=cols_initial,
                                  val=1.0)

            # Partials of r_error[i, 0, :] w.r.t. event_pos[i, :]
            # event_pos has shape (N+1, 3), event_pos[i, :] has flat indices starting at i*3
            event_pos_rows.extend([i * 6 + 0, i * 6 + 1, i * 6 + 2])
            event_pos_cols.extend([i * 3 + 0, i * 3 + 1, i * 3 + 2])

            # Partials of r_error[i, 1, :] w.r.t. arc_i_r_final
            # r_error[i, 1, :] has flat indices: [i*6 + 3, i*6 + 4, i*6 + 5]
            # arc_i_r_final has shape (1, 3), flat indices: [0, 1, 2]
            rows_final = [i * 6 + 3, i * 6 + 4, i * 6 + 5]
            cols_final = [0, 1, 2]
            self.declare_partials(f'r_error', f'arc_{i}_r_final',
                                  rows=rows_final, cols=cols_final,
                                  val=1.0)

            # Partials of r_error[i, 1, :] w.r.t. event_pos[i+1, :]
            event_pos_rows.extend([i * 6 + 3, i * 6 + 4, i * 6 + 5])
            event_pos_cols.extend([(i + 1) * 3 + 0, (i + 1) * 3 + 1, (i + 1) * 3 + 2])

        # Declare all event_pos partials at once
        self.declare_partials('r_error', 'event_pos',
                              rows=event_pos_rows, cols=event_pos_cols,
                              val=-1.0)

    def compute(self, inputs, outputs):
        """Compute miss distance errors."""
        N = self.options['N']
        event_pos = inputs['event_pos']
        r_error = outputs['r_error']
        for i in range(N):
            r_initial_i = inputs[f'arc_{i}_r_initial']
            r_final_i = inputs[f'arc_{i}_r_final']
            r_error[i, 0, :] = r_initial_i - event_pos[i, :]
            r_error[i, 1, :] = r_final_i - event_pos[i+1, :]