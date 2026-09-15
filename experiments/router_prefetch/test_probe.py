"""Small checks for the analysis decisions that can invalidate a result."""

import unittest

import numpy as np

from probe import calibrate, cache_presence, summarize, summarize_prompts


class ProbeTests(unittest.TestCase):
    def test_confidence_ties_cannot_hide_an_error(self):
        # Taking just one of the .8 candidates would manufacture 100% precision.
        c = np.array([.9, .8, .8, .7])
        y = np.array([True, True, False, False])
        self.assertIsNone(calibrate(c, y, 1, 2))
        self.assertEqual(calibrate(c, y, 1, 1), .9)

    def test_test_errors_do_not_change_the_calibration_cutoff(self):
        cutoff = calibrate(np.array([.9, .8, .7]), np.array([True, True, False]), 1, 2)
        result = summarize(np.array([.95, .85, .75]), np.array([False, False, True]), cutoff)
        self.assertEqual(cutoff, .8)
        self.assertEqual(result["issued"], 2)
        self.assertEqual(result["precision"], 0)

    def test_no_qualifying_predictions_is_not_perfect_accuracy(self):
        result = summarize(np.array([.1]), np.array([True]), .5)
        self.assertEqual(result["issued"], 0)
        self.assertIsNone(result["precision"])
        self.assertIsNone(calibrate(np.array([]), np.array([], dtype=bool), .99, 200))

    def test_prompt_summary_includes_all_layers(self):
        events = [("math", np.array([.9, .8]), np.array([True, False])),
                  ("math", np.array([.95]), np.array([True])),
                  ("code", np.array([.99]), np.array([False]))]
        results = summarize_prompts(events, .5)
        self.assertEqual(results["math"]["issued"], 3)
        self.assertEqual(results["math"]["correct"], 2)
        self.assertEqual(results["code"]["issued"], 1)

    def test_cache_keys_include_layer_and_snapshot_precedes_current_loads(self):
        selected = np.tile(np.arange(6), (1, 43, 1))
        present = cache_presence(selected, 6)
        self.assertFalse(present[0, 0].any())
        self.assertTrue(present[0, 1, 0, :6].all())
        self.assertFalse(present[0, 1, 1].any())
        self.assertFalse(present[0, 2, 0].any())
        self.assertTrue(present[0, 2, 1, :6].all())


if __name__ == "__main__":
    unittest.main()
