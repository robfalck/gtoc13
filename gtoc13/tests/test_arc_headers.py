"""Test arc headers in solution output"""
import unittest
from io import StringIO

from gtoc13 import GTOC13Solution, FlybyArc, ConicArc, PropagatedArc, YEAR


class TestArcHeaders(unittest.TestCase):

    def test_conic_arc_header(self):
        """Test that conic arc headers are generated correctly"""
        conic = ConicArc.create(
            epoch_start=0.0,
            epoch_end=50.0 * YEAR,
            position_start=(1.5e8, 0.0, 0.0),
            position_end=(1.6e8, 1.0e7, 0.0),
            velocity_start=(0.0, 30.0, 0.0),
            velocity_end=(0.0, 29.0, 0.0)
        )
        solution = GTOC13Solution(arcs=[conic])

        output = StringIO()
        solution.write(stream=output, precision=6)
        result = output.getvalue()

        # Check for conic arc header
        self.assertIn("# Conic Arc: Body 0 (heliocentric)", result)
        self.assertIn("from t=0.000000 years to t=50.000000 years", result)
        self.assertIn("# body_id", result)
        self.assertIn("epoch (s)", result)

    def test_flyby_arc_header(self):
        """Test that flyby arc headers include body name"""
        # Need a heliocentric arc first
        conic = ConicArc.create(
            epoch_start=0.0,
            epoch_end=50.0 * YEAR,
            position_start=(1.5e8, 0.0, 0.0),
            position_end=(1.6e8, 1.0e7, 0.0),
            velocity_start=(0.0, 30.0, 0.0),
            velocity_end=(0.0, 29.0, 0.0)
        )

        flyby = FlybyArc.create(
            body_id=10,  # Should show planet name
            epoch=50.0 * YEAR,
            position=(1.6e8, 1.0e7, 0.0),
            velocity_in=(0.0, 29.0, 0.0),
            velocity_out=(5.0, 28.0, 1.0),
            v_inf_in=(-5.0, 2.0, 0.0),
            v_inf_out=(5.0, 1.0, 1.0),
            is_science=True
        )
        solution = GTOC13Solution(arcs=[conic, flyby])

        output = StringIO()
        solution.write(stream=output, precision=6)
        result = output.getvalue()

        # Check for flyby header with body name
        self.assertIn("# Flyby of Body 10", result)
        self.assertIn("at t=50.000000 years", result)
        self.assertIn("v_inf_x", result)  # Flyby-specific column labels

    def test_propagated_arc_header(self):
        """Test that propagated arc headers are generated correctly"""
        prop = PropagatedArc.create(
            epochs=[50.0 * YEAR, 50.0 * YEAR + 1000, 50.0 * YEAR + 2000],
            positions=[(1.6e8, 1.0e7, 0.0), (1.61e8, 1.05e7, 1e5), (1.62e8, 1.1e7, 2e5)],
            velocities=[(5.0, 28.0, 1.0), (5.0, 27.9, 1.05), (5.0, 27.8, 1.1)],
            controls=[(0.707, 0.707, 0.0), (0.707, 0.707, 0.0), (0.707, 0.707, 0.0)]
        )
        solution = GTOC13Solution(arcs=[prop])

        output = StringIO()
        solution.write(stream=output, precision=6)
        result = output.getvalue()

        # Check for propagated arc header
        self.assertIn("# Propagated Arc: Body 0 (heliocentric)", result)
        self.assertIn("from t=50.000000 years to", result)
        self.assertIn("# body_id", result)
        self.assertIn("cx", result)  # Control column labels

    def test_multiple_arcs_headers(self):
        """Test that multiple arcs each get their own headers"""
        conic = ConicArc.create(
            epoch_start=0.0,
            epoch_end=50.0 * YEAR,
            position_start=(1.5e8, 0.0, 0.0),
            position_end=(1.6e8, 1.0e7, 0.0),
            velocity_start=(0.0, 30.0, 0.0),
            velocity_end=(0.0, 29.0, 0.0)
        )

        flyby = FlybyArc.create(
            body_id=2,
            epoch=50.0 * YEAR,
            position=(1.6e8, 1.0e7, 0.0),
            velocity_in=(0.0, 29.0, 0.0),
            velocity_out=(5.0, 28.0, 1.0),
            v_inf_in=(-5.0, 2.0, 0.0),
            v_inf_out=(5.0, 1.0, 1.0)
        )

        solution = GTOC13Solution(arcs=[conic, flyby])

        output = StringIO()
        solution.write(stream=output, precision=6)
        result = output.getvalue()

        # Check that both arc headers are present
        self.assertIn("# Conic Arc:", result)
        self.assertIn("# Flyby of Body 2", result)

        # Count the number of arc headers (should be at least 2)
        conic_headers = result.count("# Conic Arc:")
        flyby_headers = result.count("# Flyby of Body")
        self.assertEqual(conic_headers, 1)
        self.assertEqual(flyby_headers, 1)


if __name__ == '__main__':
    unittest.main()
