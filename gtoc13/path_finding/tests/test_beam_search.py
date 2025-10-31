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

    def test_bs_lambert_smoke(self):
        config = make_lambert_config(2.0, 30.0, None)
        registry = BodyRegistry(
            body_ids=(1,),
            semi_major_axes={1: BASE_SEMI_MAJOR_AXES[1]},
            weights={1: BASE_BODY_WEIGHTS[1]},
            tof_sample_count=6,
        )
        expand_fn = make_expand_fn(
            config,
            registry,
            allow_repeat=True,
            same_body_samples=registry.tof_sample_count,
        )
        score_fn = make_score_fn(config, registry, "simple")

        start_r = ephemeris_position(1, 0.0)
        root_state = (
            Encounter(
                body=1,
                t=0.0,
                r=start_r,
                vinf_in=None,
                vinf_in_vec=None,
                vinf_out=None,
                vinf_out_vec=None,
                flyby_valid=None,
                flyby_altitude=None,
                dv_periapsis=None,
                dv_periapsis_vec=None,
                J_total=0.0,
            ),
        )

        beam = BeamSearch(
            expand_fn=expand_fn,
            score_fn=score_fn,
            beam_width=1,
            max_depth=1,
            key_fn=key_fn,
            max_workers=2,
            score_chunksize=4,
            parallel_backend="process",
        )

        final_nodes = beam.run(root_state)
        self.assertEqual(len(final_nodes), 1)
        best = final_nodes[0]

        assert_allclose(best.cum_score, 2529.5333370314443, rtol=1e-10, atol=1e-10)
        self.assertEqual(len(best.state), 2)
        child = best.state[-1]
        self.assertEqual(child.body, 1)
        assert_allclose(child.t, 11.199992501712394, rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
