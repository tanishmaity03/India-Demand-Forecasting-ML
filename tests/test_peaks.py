"""Tests for src/peaks.py (run: python -m unittest discover -s tests -t .)."""
import unittest

import numpy as np

from src.peaks import episode_ids, warning_metrics, sweep


class PeakTests(unittest.TestCase):
    def test_episode_merging(self):
        flag = [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
        ids = episode_ids(flag, max_gap=3)
        self.assertEqual(ids.max(), 1)                  # hours 1-5 merge (gap of 3); hour 13 is separate
        self.assertTrue((ids[1:6] == 0).all())
        self.assertEqual(ids[13], 1)

    def test_perfect_warning(self):
        true_high = np.array([0, 1, 1, 0, 0, 1], bool)
        m = warning_metrics(true_high, true_high)
        self.assertEqual((m["precision"], m["recall"], m["F1"]), (1.0, 1.0, 1.0))
        self.assertEqual(m["episode_recall"], 1.0)

    def test_missed_and_false_alarm(self):
        true_high = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
        warn = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], bool)
        m = warning_metrics(true_high, warn)
        self.assertEqual((m["TP"], m["FP"], m["FN"]), (0, 2, 2))
        self.assertEqual(m["episode_recall"], 0.0)
        self.assertEqual(m["episode_precision"], 0.0)

    def test_margin_increases_recall(self):
        actual = np.array([100, 98, 90, 101, 80], float); thr = np.full(5, 99.0)
        pred = np.array([100, 97, 90, 100, 80], float)
        sw = sweep(actual, pred, thr, margins=[0.0, 0.03])
        self.assertGreaterEqual(sw.loc[1, "recall"], sw.loc[0, "recall"])


if __name__ == "__main__":
    unittest.main()