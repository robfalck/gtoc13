import unittest
import math

import numpy as np
from numpy.testing import assert_allclose
from gtoc13.path_finding import BeamSearch
from gtoc13.path_finding.beam.config import (
    BodyRegistry,
    BASE_BODY_WEIGHTS,
    BASE_SEMI_MAJOR_AXES,
    make_lambert_config,
)
from gtoc13.path_finding.beam.dv_limits import max_transfer_dv_solar_sail
from gtoc13.path_finding.beam.lambert import Encounter, ephemeris_position
from gtoc13.path_finding.beam.pipeline import make_expand_fn, make_score_fn, key_fn


class TestBeamSearch(unittest.TestCase):

    def test_basic(self):
        # Ultra-brief smoke test using integers
        def expand_fn(s: int):              # cheap proposals
            return (s + d for d in (1, 2, 3))

        def score_fn(p: int, c: int):       # heavy scoring (mock)
            target = 12
            return -abs(target - c)         # higher is better

        bs = BeamSearch(
            expand_fn,
            score_fn,
            beam_width=4,
            max_depth=5,
            key_fn=lambda s: s,
            parallel_backend="thread",
        )
        final_beam = bs.run(0)
        best = max(final_beam, key=lambda n: n.cum_score)

        assert_allclose(best.cum_score, -19.0, rtol=1.0E-3, atol=1.0E-3)  # use allclose for floating point testing
        self.assertEqual(best.state, 13)
        self.assertEqual(bs.reconstruct_path(best), [0, 3, 6, 9, 12, 13])


class TestDynamicDvHelper(unittest.TestCase):

    def test_zero_factor_returns_zero(self):
        r = (1.0e5, 0.0, 0.0)
        cap = max_transfer_dv_solar_sail(r, r, tof_days=1.0, factor=0.0)
        self.assertEqual(cap, 0.0)

    def test_expected_nominal_value(self):
        # Simple geometry: rotate arrival position 60 degrees in-plane.
        au_km = 149_597_870.7
        theta = math.radians(60.0)
        r_depart = (au_km, 0.0, 0.0)
        r_arrive = (au_km * math.cos(theta), au_km * math.sin(theta), 0.0)
        factor = 0.25
        tof_days = 100.0

        cap = max_transfer_dv_solar_sail(r_depart, r_arrive, tof_days=tof_days, factor=factor)

        # Manual evaluation of the published bound for comparison.
        r1_au = np.linalg.norm(r_depart) / au_km
        r2_au = np.linalg.norm(r_arrive) / au_km
        avg_au = 0.5 * (r1_au + r2_au)
        a1au = 3.24156e-4
        seconds = tof_days * 86_400.0
        expected = (a1au / (avg_au**2)) * seconds * factor * 1e-3

        assert_allclose(cap, expected, rtol=1e-12, atol=0.0)


if __name__ == '__main__':
    unittest.main()
