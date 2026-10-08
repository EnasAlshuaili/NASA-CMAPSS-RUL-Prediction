from pathlib import Path
import unittest

from src.data_loader import (
    COLUMN_NAMES,
    add_training_rul,
    load_rul_labels,
    load_test_data,
    load_train_data,
    validate_cmapss_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw"


class DataLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_data = load_train_data(RAW_DATA / "train_FD001.txt")

    def test_loads_fd001_measurements_and_labels(self) -> None:
        test_data = load_test_data(RAW_DATA / "test_FD001.txt")
        labels = load_rul_labels(RAW_DATA / "RUL_FD001.txt")

        self.assertEqual(self.train_data.shape, (20631, 26))
        self.assertEqual(list(self.train_data.columns), COLUMN_NAMES)
        self.assertEqual(test_data["engine_id"].nunique(), len(labels))
        self.assertEqual(labels.name, "RUL")
        self.assertGreaterEqual(labels.min(), 0)

    def test_add_training_rul_does_not_mutate_input(self) -> None:
        labeled_data = add_training_rul(self.train_data)

        self.assertNotIn("RUL", self.train_data.columns)
        self.assertTrue(labeled_data.groupby("engine_id")["RUL"].min().eq(0).all())
        self.assertEqual(labeled_data.iloc[0]["RUL"], 191)

    def test_rejects_non_contiguous_cycles(self) -> None:
        malformed = self.train_data.iloc[:2].copy()
        malformed.loc[malformed.index[1], "cycle"] = 3

        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_cmapss_data(malformed)

    def test_rejects_out_of_order_engines(self) -> None:
        malformed = self.train_data.iloc[[0, 192]].copy()
        malformed.iloc[0, malformed.columns.get_loc("engine_id")] = 2
        malformed.iloc[1, malformed.columns.get_loc("engine_id")] = 1

        with self.assertRaisesRegex(ValueError, "ordered"):
            validate_cmapss_data(malformed)


if __name__ == "__main__":
    unittest.main()
