import unittest

from numpy.testing import assert_allclose
from gtoc13.path_finding import BeamSearch
from gtoc13.path_finding.beam.config import (
    BodyRegistry,
    BASE_BODY_WEIGHTS,
    BASE_SEMI_MAJOR_AXES,
    make_lambert_config,
)
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


if __name__ == '__main__':
    unittest.main()
