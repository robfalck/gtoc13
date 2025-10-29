import unittest

from gtoc13.path_finding import BeamSearch


class TestBeamSearch(unittest.TestCase):

    def test_basic(self):
        # Ultra-brief smoke test using integers
        def expand_fn(s: int):              # cheap proposals
            return (s + d for d in (1, 2, 3))

        def score_fn(p: int, c: int):       # heavy scoring (mock)
            target = 12
            return -abs(target - c)         # higher is better

        bs = BeamSearch(expand_fn, score_fn, beam_width=4, max_depth=5, key_fn=lambda s: s)
        final_beam = bs.run(0)
        best = max(final_beam, key=lambda n: n.cum_score)
        print("Best:", best.cum_score, "state:", best.state, "path:", bs.reconstruct_path(best))


if __name__ == '__main__':
    unittest.main()
