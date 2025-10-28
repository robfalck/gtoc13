from dataclasses import dataclass, replace
from typing import Iterable, Optional, Tuple, Hashable
import math
from gtoc13.bodies import bodies_data
from gtoc13.lambert import lambert

# --------------------- Data shapes ---------------------
"""
script to use the beam search with lambert for the dynamics
"""

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Encounter:
    """State element at an encounter epoch."""
    body: int
    t: float  # encounter epoch (days)
    r: Vec3  # position at epoch
    vinf_in: Optional[float] = None
    vinf_out: Optional[float] = None
    J_total: float = 0.0  # cumulative score up to/including this encounter


@dataclass(frozen=True, slots=True)
class Proposal:
    """Cheap next-step proposal from expand_fn."""
    body: int  # candidate next target
    tof: float  # proposed time-of-flight (days)


State = tuple[Encounter, ...]  # full path (immutable)


# --------------------- Cheap expansion ---------------------

def expand_fn(path: State) -> Iterable[Proposal]:
    """
    Enumerate cheap proposals only (no heavy physics).
    Replace (targets, TOFs) with your sampler.
    """
    last_body = path[-1].body if path else 2  # e.g., 2=Earth as default start
    for tgt in (0, 1, 3):  # e.g., Mercury, Venus, Mars
        if tgt == last_body:
            continue
        for tof in (120.0, 180.0):  # small grid around a nominal
            yield Proposal(body=tgt, tof=tof)


# --------------------- Heavy scoring/resolution ---------------------

def score_fn(path: State, prop: Proposal):
    """
    Heavy step: run Lambert/feasibility, retro-resolve parent.vinf_out,
    build child Encounter (with t, r, vinf_in), and return (delta, new_path).
    """
    # Parent encounter (or synthetic launch if empty)
    if not path:
        t0 = 0.0
        r0 = ephemeris_fake(2, t0)  # Earth at t0 (example)
        parent = Encounter(body=2, t=t0, r=r0, J_total=0.0)
        parent_path = (parent,)
    else:
        parent = path[-1]
        parent_path = path

    # Child epoch and Lambert solve
    t1 = parent.t + prop.tof
    feasible, vinf_out, vinf_in, r1 = lambert(parent, prop.body, t1)
    if not feasible:
        return float("-inf")  # prune this candidate

    # Retro-resolve parent's vinf_out (branch-specific, immutable copy)
    parent_resolved = replace(parent, vinf_out=vinf_out)
    prefix = parent_path[:-1] + (parent_resolved,)

    # Child encounter (vinf_out is unknown until *its* child is chosen)
    child = Encounter(body=prop.body, t=t1, r=r1, vinf_in=vinf_in, vinf_out=None)

    # Leg contribution depends on full context as needed (but not on vinf_out(parent) in your case)
    child_contrib = leg_score(parent_resolved, child)

    # Update cumulative and return (delta, resolved_state)
    child = replace(child, J_total=parent_resolved.J_total + child_contrib)
    return child_contrib, prefix + (child,)


# --------------------- Dedup key (coarse) ---------------------

def key_fn(state: State) -> Hashable:
    """
    Coarsely bucket by latest encounter to collapse near-duplicates.
    Tune bin widths to control pruning strength.
    """
    if not state:
        return ("root",)
    last = state[-1]
    t_bin = int(round(last.t / 10.0))  # 10-day bins
    v_in = -1 if last.vinf_in is None else last.vinf_in
    v_bin = int(round(v_in * 10.0))  # 0.1-unit bins
    return (last.body, t_bin, v_bin)


# --------------------- Stubs you will replace ---------------------

def ephemeris_fake(body: int, t: float) -> Vec3:
    """Return position r(body, t). Replace with real ephemeris."""
    return (1.0, 0.0, 0.0)


def lambert_fake(parent: Encounter, child_body: int, t1: float) -> Tuple[bool, float, float, Vec3]:
    """
    Solve Lambert from (parent.r at parent.t) to r(child_body, t1).
    Returns: (feasible, vinf_out_at_parent, vinf_in_at_child, r_child_at_t1).
    Replace with your real solver + feasibility checks.
    """
    r1 = ephemeris_fake(child_body, t1)
    feasible = True
    vinf_out, vinf_in = 2.8, 3.1
    return feasible, vinf_out, vinf_in, r1


def leg_score(parent: Encounter, child: Encounter) -> float:
    """
    Incremental score for the new leg. Replace with your science/mission term.
    Assumption (per your note): parent’s contribution depends only on vinf_in,
    so retro-setting vinf_out doesn’t change J_total(parent).
    """
    # Example: prefer smaller arrival vinf; add your seasonal/weighting logic here.
    return 1.0 / (1.0 + (child.vinf_in or math.inf))


if __name__ == '__main__':
    print(bodies_data.keys())

    x = lambert(1, 2, 100 * 86400, 200 * 864000, bodies_data=bodies_data)
    print(x)
