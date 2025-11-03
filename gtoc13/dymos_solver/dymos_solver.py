from typing import Sequence

import openmdao.api as om
import dymos as dm

from gtoc13.solution import GTOC13Solution


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