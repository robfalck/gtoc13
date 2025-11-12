"""Test dymos_solver execution"""
import unittest
import subprocess
from pathlib import Path

import numpy as np

from gtoc13 import GTOC13Solution

from openmdao.utils.assert_utils import assert_near_equal


class TestDymosSolver(unittest.TestCase):

    def test_dymos_solver_execution(self):
        """Test that dymos_solver executes successfully and produces expected output files"""
        # Define the solution name
        solution_name = 'ten_nine'
        solutions_dir = Path(__file__).parent.parent.parent.parent / 'solutions'
        txt_file = solutions_dir / f'{solution_name}.txt'
        png_file = solutions_dir / f'{solution_name}.png'

        try:
            # Execute dymos_solver
            cmd = [
                'python',
                '-m', 'gtoc13.dymos_solver',
                '--bodies', '10', 
                '--flyby-times', '50',
                '--controls', 'r',
                '--max-time', '200',
                '--name', solution_name
            ]

            print(' '.join(cmd))

            # Run with timeout to prevent hanging (10 minutes should be enough)
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent.parent,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )

            # Check that the process completed successfully
            self.assertEqual(result.returncode, 0,
                           f"dymos_solver failed with return code {result.returncode}\n"
                           f"stdout: {result.stdout}\n"
                           f"stderr: {result.stderr}")

            # Verify that output files were created
            self.assertTrue(txt_file.exists(),
                          f"Expected solution file {txt_file} was not created")
            self.assertTrue(png_file.exists(),
                          f"Expected plot file {png_file} was not created")

            # Verify that files are not empty
            self.assertGreater(txt_file.stat().st_size, 0,
                             "Solution file is empty")
            self.assertGreater(png_file.stat().st_size, 0,
                             "Plot file is empty")
            
            solution_name2 = 'ten_nine2'
            txt_file2 = solutions_dir / f'{solution_name2}.txt'
            png_file2 = solutions_dir / f'{solution_name2}.png'

            # Run again, using the first solution as a guess
            cmd = [
                'python',
                '-m', 'gtoc13.dymos_solver',
                '--bodies', '10', 
                '--flyby-times', '50',
                '--controls', '1',  # Optimal control, load radial solution from previous
                '--max-time', '50',
                '--name', solution_name2,
                '--load', str(txt_file),
                '--mode', 'run'
            ]

            print(' '.join(cmd))

            # Run with timeout to prevent hanging (10 minutes should be enough)
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent.parent,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )

            # Check that the process completed successfully
            self.assertEqual(result.returncode, 0,
                           f"dymos_solver failed with return code {result.returncode}\n"
                           f"stdout: {result.stdout}\n"
                           f"stderr: {result.stderr}")

            # Verify that output files were created
            self.assertTrue(txt_file2.exists(),
                          f"Expected solution file {txt_file2} was not created")
            self.assertTrue(png_file2.exists(),
                          f"Expected plot file {png_file2} was not created")

            # Verify that files are not empty
            self.assertGreater(txt_file2.stat().st_size, 0,
                             "Solution file is empty")
            self.assertGreater(png_file2.stat().st_size, 0,
                             "Plot file is empty")
            
            sol1 = GTOC13Solution.load(txt_file)
            sol2 = GTOC13Solution.load(txt_file)

            prop_arc_1 = sol1.arcs[0]
            flyby_arc_1 = sol1.arcs[1]

            prop_arc_2 = sol2.arcs[0]
            flyby_arc_2 = sol2.arcs[1]

            sol1_t = np.array([state.epoch for state in prop_arc_1.state_points])
            sol1_r = np.array([state.position for state in prop_arc_1.state_points])
            sol1_v = np.array([state.velocity for state in prop_arc_1.state_points])
            sol1_u = np.array([state.control for state in prop_arc_1.state_points])

            prop_arc_2 = sol2.arcs[0]
            flyby_arc_2 = sol2.arcs[1]

            sol2_t = np.array([state.epoch for state in prop_arc_2.state_points])
            sol2_r = np.array([state.position for state in prop_arc_2.state_points])
            sol2_v = np.array([state.velocity for state in prop_arc_2.state_points])
            sol2_u = np.array([state.control for state in prop_arc_2.state_points])

            self.assertEqual(prop_arc_1.bodies, (-1, 10))
            self.assertEqual(prop_arc_2.bodies, (-1, 10))

            assert_near_equal(sol1_r, sol2_r)
            assert_near_equal(sol1_v, sol2_v)
            assert_near_equal(sol1_u, sol2_u)
            assert_near_equal(sol1_t, sol2_t)

            assert_near_equal(flyby_arc_1.position,
                              flyby_arc_2.position)
            
            assert_near_equal(flyby_arc_1.velocity_in,
                              flyby_arc_2.velocity_in)
            
            assert_near_equal(flyby_arc_1.velocity_out,
                              flyby_arc_2.velocity_out)       

        except subprocess.TimeoutExpired:
            self.fail("dymos_solver execution timed out after 10 minutes")

        finally:
            # Clean up output files
            if txt_file.exists():
                txt_file.unlink()
            if png_file.exists():
                png_file.unlink()
            if txt_file2.exists():
                txt_file2.unlink()
            if png_file2.exists():
                png_file2.unlink()

    def test_multi_body_load(self):
        """Test that dymos_solver executes successfully and produces expected output files"""
        # Define the solution name
        solution_name = 'multi_body_load_test'
        solutions_dir = Path(__file__).parent.parent.parent.parent / 'solutions'
        txt_file = solutions_dir / f'{solution_name}.txt'
        sol_file = solutions_dir / 'x_bespin_hoth_beyonce.txt'
        png_file = solutions_dir / f'{solution_name}.png'

        try:
            # Execute dymos_solver
            cmd = [
                'python',
                '-m', 'gtoc13.dymos_solver',
                '--bodies', '10', '6', '4', '5', 
                '--flyby-times', '20', '50', '70', '80',
                '--controls', 'r', 'r', '0', '0',
                '--max-time', '200',
                '--name', solution_name,
                '--load', str(sol_file),
                '--mode', 'run'
            ]

            print(' '.join(cmd))

            # Run with timeout to prevent hanging (10 minutes should be enough)
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent.parent,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )

            # Check that the process completed successfully
            self.assertEqual(result.returncode, 0,
                        f"dymos_solver failed with return code {result.returncode}\n"
                        f"stdout: {result.stdout}\n"
                        f"stderr: {result.stderr}")

            # Verify that output files were created
            self.assertTrue(txt_file.exists(),
                        f"Expected solution file {txt_file} was not created")
            self.assertTrue(png_file.exists(),
                        f"Expected plot file {png_file} was not created")

            # Verify that files are not empty
            self.assertGreater(txt_file.stat().st_size, 0,
                            "Solution file is empty")
            self.assertGreater(png_file.stat().st_size, 0,
                            "Plot file is empty")
            
            sol1 = GTOC13Solution.load(sol_file)
            sol2 = GTOC13Solution.load(txt_file)

            for i in range(4):

                prop_arc_1 = sol1.arcs[2*i]
                flyby_arc_1 = sol1.arcs[2*i+1]

                prop_arc_2 = sol2.arcs[2*i]
                flyby_arc_2 = sol2.arcs[2*i+1]

                sol1_t = np.array([state.epoch for state in prop_arc_1.state_points])
                sol1_r = np.array([state.position for state in prop_arc_1.state_points])
                sol1_v = np.array([state.velocity for state in prop_arc_1.state_points])
                sol1_u = np.array([state.control for state in prop_arc_1.state_points])

                sol2_t = np.array([state.epoch for state in prop_arc_2.state_points])
                sol2_r = np.array([state.position for state in prop_arc_2.state_points])
                sol2_v = np.array([state.velocity for state in prop_arc_2.state_points])
                sol2_u = np.array([state.control for state in prop_arc_2.state_points])

                self.assertEqual(prop_arc_1.bodies, prop_arc_2.bodies)

                assert_near_equal(sol1_r, sol2_r)
                assert_near_equal(sol1_v, sol2_v)
                assert_near_equal(sol1_u, sol2_u)
                assert_near_equal(sol1_t, sol2_t)

                assert_near_equal(flyby_arc_1.position,
                                flyby_arc_2.position)
                
                assert_near_equal(flyby_arc_1.velocity_in,
                                flyby_arc_2.velocity_in)
                
                assert_near_equal(flyby_arc_1.velocity_out,
                                flyby_arc_2.velocity_out)       

        except subprocess.TimeoutExpired:
            self.fail("dymos_solver execution timed out after 10 minutes")

        finally:
            # Clean up output files
            if txt_file.exists():
                txt_file.unlink()
            if png_file.exists():
                png_file.unlink()


if __name__ == '__main__':
    unittest.main()
