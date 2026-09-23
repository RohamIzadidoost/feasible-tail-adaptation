"""CPU regression tests for tied-score EER and threshold diagnostics."""
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from metrics import compute_eer, threshold_diagnostics


class MetricTests(unittest.TestCase):
    def test_all_tied_and_order_invariance(self):
        y = np.array([0, 0, 1, 1, 1])
        s = np.ones(len(y))
        self.assertAlmostEqual(compute_eer(y, s)[0], 0.5)
        self.assertAlmostEqual(compute_eer(y[::-1], s)[0], 0.5)
        self.assertAlmostEqual(roc_auc_score(y, s), 0.5)
        self.assertAlmostEqual(threshold_diagnostics(y, s)["oracle_accuracy"], .6)

    def test_interpolated_crossing(self):
        # ROC jumps from (0, 1/3) to (1/2, 1): crossing is EER=2/7.
        y = np.array([1, 0, 1, 1, 0])
        s = np.array([.9, .5, .5, .5, .1])
        self.assertAlmostEqual(compute_eer(y, s)[0], 2 / 7)

    def test_perfect_and_reversed(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(compute_eer(y, y)[0], 0)
        self.assertEqual(compute_eer(y, 1 - y)[0], 1)

    def test_oracle_includes_constant_decisions(self):
        y = np.array([0, 0, 0, 1])
        s = np.array([.8, .8, .8, .2])
        d = threshold_diagnostics(y, s)
        self.assertAlmostEqual(d["oracle_accuracy"], .75)
        self.assertAlmostEqual(d["threshold_gap"], .75)

    def test_oracle_matches_exhaustive_thresholds(self):
        rng = np.random.default_rng(12)
        for _ in range(20):
            y = np.r_[0, 1, rng.integers(0, 2, 48)]
            s = rng.integers(0, 5, len(y)) / 4
            best = max(np.mean((s >= t) == y) for t in np.r_[np.inf, np.unique(s)])
            self.assertAlmostEqual(threshold_diagnostics(y, s)["oracle_accuracy"], best)

    def test_invalid_inputs(self):
        for y, s in [([0, 0], [.1, .2]), ([0, 1], [np.nan, .2]),
                     ([0, 1], [.2]), ([], [])]:
            with self.assertRaises(ValueError):
                compute_eer(y, s)


if __name__ == "__main__":
    unittest.main()
