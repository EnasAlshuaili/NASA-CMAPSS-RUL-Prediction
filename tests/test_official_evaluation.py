import json
import unittest

import pandas as pd

from src.evaluate_official_test import METRICS_FILE, PREDICTIONS_FILE, evaluate


class OfficialTestEvaluationTests(unittest.TestCase):
    """Recompute the official FD001 test evaluation and compare with the saved results."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.result, cls.predictions = evaluate()
        cls.saved = json.loads(METRICS_FILE.read_text(encoding="utf-8"))

    def test_saved_metrics_reproduce(self) -> None:
        for key in ("model", "data", "official_last_cycle", "all_test_rows", "validation_reference"):
            with self.subTest(section=key):
                self.assertEqual(self.result[key], self.saved[key])

    def test_saved_predictions_reproduce(self) -> None:
        saved = pd.read_csv(PREDICTIONS_FILE)
        pd.testing.assert_frame_equal(saved, self.predictions.reset_index(drop=True), check_dtype=False)

    def test_engine_and_label_alignment(self) -> None:
        self.assertEqual(self.result["data"]["test_engines"], 100)
        self.assertEqual(self.result["data"]["rul_labels"], 100)
        self.assertEqual(self.result["data"]["test_rows"], 13096)
        self.assertEqual(int(self.predictions["is_last_cycle"].sum()), 100)


if __name__ == "__main__":
    unittest.main()
