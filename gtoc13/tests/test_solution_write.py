"""Test solution write methods"""
import unittest
from io import StringIO
import tempfile
import os
from pathlib import Path

from gtoc13 import GTOC13Solution, ConicArc, YEAR


class TestSolutionWrite(unittest.TestCase):

    def setUp(self):
        """Create a simple solution for testing"""
        conic = ConicArc.create(
            epoch_start=0.0,
            epoch_end=100.0 * YEAR,
            position_start=(1.5e8, 0.0, 0.0),
            position_end=(1.6e8, 1.0e7, 0.0),
            velocity_start=(0.0, 30.0, 0.0),
            velocity_end=(0.0, 29.0, 0.0)
        )
        self.solution = GTOC13Solution(arcs=[conic], comments=["Test solution"])

    def test_write_to_stream(self):
        """Test writing to a stream (StringIO)"""
        output = StringIO()
        self.solution.write(stream=output, precision=8)
        result = output.getvalue()

        # Check that output contains expected elements
        self.assertIn("# GTOC13 Solution File", result)
        self.assertIn("# Test solution", result)

        # Verify scientific notation is used
        self.assertTrue('e+' in result or 'e-' in result,
                       "Scientific notation should be used")

        # Check that we have data lines (not just comments)
        lines = [l for l in result.split('\n') if l and not l.startswith('#')]
        self.assertGreater(len(lines), 0, "Should have data lines")

    def test_write_to_file(self):
        """Test writing to a file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
            tmp_path = tmp.name

        try:
            # Write to file
            self.solution.write_to_file(tmp_path, precision=15)

            # Read it back
            with open(tmp_path, 'r') as f:
                content = f.read()

            # Verify content
            self.assertIn("# GTOC13 Solution File", content)
            self.assertTrue('e+' in content or 'e-' in content,
                          "Scientific notation should be used")

            # Verify we can read it back
            loaded = GTOC13Solution.load(tmp_path)
            self.assertEqual(len(loaded.arcs), 1)

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_scientific_notation_format(self):
        """Test that scientific notation format is used"""
        output = StringIO()
        self.solution.write(stream=output, precision=6)
        result = output.getvalue()

        # Parse a data line
        lines = [l for l in result.split('\n') if l and not l.startswith('#')]
        first_line = lines[0]
        parts = first_line.split()

        # Check that numeric values use scientific notation
        # Skip first two (body_id and flag which are integers)
        for i in range(2, len(parts)):
            # Each should be in format like 0.000000e+00 or 1.500000e+08
            self.assertRegex(parts[i], r'[+-]?\d\.\d+e[+-]\d+',
                           f"Value '{parts[i]}' should be in scientific notation")


if __name__ == '__main__':
    unittest.main()
