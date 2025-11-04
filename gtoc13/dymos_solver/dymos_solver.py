from typing import Sequence

import openmdao.api as om
import dymos as dm

from gtoc13.solution import GTOC13Solution
from gtoc13.dymos_solver.ephem_comp import EphemComp


def solve(bodies: Sequence[int], times: Sequence[float]) -> GTOC13Solution:
    """
    Parameters
    ----------
    bodies : Sequence[int]
        The bodies that make up the solution, in order of visit.
    times : Sequence[float]
        The approximate encounter year of each event. times has 
        one more element than bodies because the initial time
        of the trajectory at the starting plane is included.

    Returns
    -------
    solution : Solution
        The GTOC solution instance for the posed problem.

    Raises
    ------
    ValueError
        If dymos is unable to find a solution.
    """
    
    prob = om.Problem()

    prob.model.add_subsystem('ephem', EphemComp(units='km/s', bodies=bodies))

    prob.setup()

    prob.set_val('ephem.times', times, units='year')

    prob.run_model()


if __name__ == '__main__':
    solve(bodies=[10, 9], times=[0.0, 30.0, 40.0])