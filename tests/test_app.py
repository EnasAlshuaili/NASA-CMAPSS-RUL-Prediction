"""End-to-end tests that drive the real app.py with Streamlit's AppTest."""

import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.data_loader import add_training_rul, load_train_data
from src.model_service import FEATURE_NAMES, load_model_bundle, sensor_ranges


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(PROJECT_ROOT / "app.py")
BUNDLE = load_model_bundle(PROJECT_ROOT / "models")
DATA = add_training_rul(load_train_data(PROJECT_ROOT / "data" / "raw" / "train_FD001.txt"))
TRAIN_RANGES = sensor_ranges(DATA[DATA["engine_id"].isin(BUNDLE.train_engines)])


def row_for(engine: int, cycle: int) -> pd.Series:
    return DATA[(DATA["engine_id"] == engine) & (DATA["cycle"] == cycle)].iloc[0]


def direct_prediction(engine: int, cycle: int) -> float:
    """Prediction from the saved model on the exact dataset row, bypassing the app."""
    frame = row_for(engine, cycle)[FEATURE_NAMES].to_frame().T.astype(float)
    return max(0.0, float(BUNDLE.model.predict(frame)[0]))


def html_text(at: AppTest) -> str:
    return "\n".join(element.proto.body for element in at.get("html"))


class AppTestCase(unittest.TestCase):
    def new_app(self, page: str = "Prediction") -> AppTest:
        at = AppTest.from_file(APP_PATH, default_timeout=120).run()
        self.assert_clean(at)
        if page == "Prediction":
            at.button(key="cta_prediction").click().run()
            self.assert_clean(at)
        return at

    def assert_clean(self, at: AppTest) -> None:
        self.assertFalse(at.exception, [e.value for e in at.exception])

    def select(self, at: AppTest, engine: int, cycle: int) -> None:
        at.selectbox(key="selected_engine").set_value(engine).run()
        at.selectbox(key="selected_cycle").set_value(cycle).run()
        self.assert_clean(at)

    def load_and_predict(self, at: AppTest, engine: int, cycle: int) -> dict:
        self.select(at, engine, cycle)
        at.button(key="load_btn").click().run()
        at.button(key="predict_btn").click().run()
        self.assert_clean(at)
        return at.session_state["predictions"]["sample"]


class SampleEnginePredictionTests(AppTestCase):
    def test_engine_5_predictions_match_direct_model(self) -> None:
        at = self.new_app()
        for cycle in (1, 50, 100, 150, 200, 250):
            with self.subTest(cycle=cycle):
                prediction = self.load_and_predict(at, 5, cycle)
                row = row_for(5, cycle)
                self.assertEqual((prediction["engine"], prediction["cycle"]), (5, cycle))
                self.assertEqual(prediction["values"], {f: float(row[f]) for f in FEATURE_NAMES})
                self.assertAlmostEqual(prediction["rul"], direct_prediction(5, cycle), places=6)
                self.assertEqual(prediction["actual_rul"], int(row["RUL"]))

                page = html_text(at)
                self.assertIn(f"Telemetry loaded: Engine 5 • Cycle {cycle}", page)
                self.assertIn(f'{prediction["rul"]:.1f}<span> cycles</span>', page)
                self.assertIn(f'<div class="fact-label">Cycle</div><div class="fact-value">{cycle}</div>', page)

    def test_engine_5_cycle_250_uses_cycle_250_telemetry(self) -> None:
        prediction = self.load_and_predict(self.new_app(), 5, 250)
        self.assertEqual(prediction["actual_rul"], 19)
        self.assertAlmostEqual(prediction["rul"], direct_prediction(5, 250), places=6)
        self.assertNotAlmostEqual(prediction["rul"], direct_prediction(5, 205), places=1)

    def test_other_engines_match_direct_model(self) -> None:
        at = self.new_app()
        for engine, cycle in ((1, 96), (31, 164), (91, 50), (2, 201), (100, 1)):
            with self.subTest(engine=engine, cycle=cycle):
                prediction = self.load_and_predict(at, engine, cycle)
                self.assertAlmostEqual(prediction["rul"], direct_prediction(engine, cycle), places=6)


class StaleTelemetryTests(AppTestCase):
    def test_changing_cycle_without_loading_blocks_prediction(self) -> None:
        at = self.new_app()
        old = self.load_and_predict(at, 5, 205)

        at.selectbox(key="selected_cycle").set_value(250).run()
        self.assert_clean(at)
        self.assertTrue(at.button(key="predict_btn").disabled)
        self.assertTrue(any("Engine 5 • Cycle 250" in w.value for w in at.warning))
        page = html_text(at)
        self.assertIn("Telemetry loaded: Engine 5 • Cycle 205", page)
        self.assertIn("Selection not loaded", page)
        self.assertNotIn(f'{old["rul"]:.1f}<span> cycles</span>', page)
        self.assertNotIn('<div class="fact-label">Cycle</div><div class="fact-value">250</div>', page)
        self.assertEqual(at.session_state["predictions"]["sample"]["cycle"], 205)

        at.button(key="load_btn").click().run()
        at.button(key="predict_btn").click().run()
        prediction = at.session_state["predictions"]["sample"]
        self.assertEqual(prediction["cycle"], 250)
        self.assertAlmostEqual(prediction["rul"], direct_prediction(5, 250), places=6)

    def test_changing_engine_without_loading_blocks_prediction(self) -> None:
        at = self.new_app()
        self.load_and_predict(at, 5, 100)

        at.selectbox(key="selected_engine").set_value(11).run()
        self.assert_clean(at)
        self.assertEqual(at.session_state["selected_cycle"], 100)
        self.assertTrue(at.button(key="predict_btn").disabled)
        self.assertIn("Telemetry loaded: Engine 5 • Cycle 100", html_text(at))

    def test_engine_change_clamps_cycle_to_engine_life(self) -> None:
        at = self.new_app()
        self.select(at, 5, 250)
        at.selectbox(key="selected_engine").set_value(13).run()  # engine 13 has 163 cycles
        self.assert_clean(at)
        self.assertEqual(at.session_state["selected_cycle"], 163)


class ManualInputTests(AppTestCase):
    def manual_app(self) -> AppTest:
        at = self.new_app()
        at.radio(key="input_mode").set_value("Manual Sensor Input").run()
        self.assert_clean(at)
        return at

    def set_manual(self, at: AppTest, values: dict) -> None:
        for feature, value in values.items():
            at.number_input(key=f"manual_{feature}").set_value(value)
        at.run()
        self.assert_clean(at)

    def test_default_values_are_training_medians_and_predict(self) -> None:
        at = self.manual_app()
        self.assertFalse(at.button(key="manual_predict_btn").disabled)
        at.button(key="manual_predict_btn").click().run()
        prediction = at.session_state["predictions"]["manual"]
        frame = pd.DataFrame([TRAIN_RANGES["median"].to_dict()])[FEATURE_NAMES]
        self.assertAlmostEqual(prediction["rul"], max(0.0, float(BUNDLE.model.predict(frame)[0])), places=6)
        self.assertIn("Manual input", html_text(at))

    def test_manual_values_from_dataset_row_match_direct_model(self) -> None:
        at = self.manual_app()
        row = row_for(5, 250)
        self.set_manual(at, {f: float(row[f]) for f in FEATURE_NAMES})
        at.button(key="manual_predict_btn").click().run()
        self.assertAlmostEqual(at.session_state["predictions"]["manual"]["rul"], direct_prediction(5, 250), places=6)

    def test_all_zero_values_are_blocked(self) -> None:
        at = self.manual_app()
        self.set_manual(at, {f: 0.0 for f in FEATURE_NAMES})
        self.assertTrue(at.button(key="manual_predict_btn").disabled)
        self.assertTrue(any("must be greater than zero" in e.value for e in at.error))
        self.assertIsNone(at.session_state["predictions"]["manual"])

    def test_out_of_range_value_requires_acknowledgement(self) -> None:
        at = self.manual_app()
        self.set_manual(at, {"sensor_11": float(TRAIN_RANGES.loc["sensor_11", "max"]) + 2.0})
        self.assertTrue(at.button(key="manual_predict_btn").disabled)
        self.assertTrue(any("outside the range" in w.value for w in at.warning))

        at.checkbox(key="manual_ack").check().run()
        self.assertFalse(at.button(key="manual_predict_btn").disabled)
        at.button(key="manual_predict_btn").click().run()
        prediction = at.session_state["predictions"]["manual"]
        self.assertEqual(prediction["out_of_range"], ["sensor_11"])
        self.assertIn("outside the training range", html_text(at))

    def test_editing_after_prediction_hides_stale_result(self) -> None:
        at = self.manual_app()
        at.button(key="manual_predict_btn").click().run()
        self.set_manual(at, {"sensor_4": float(TRAIN_RANGES.loc["sensor_4", "median"]) + 1.0})
        self.assertIn("Inputs changed", html_text(at))


class NavigationAndStateTests(AppTestCase):
    def test_no_sidebar_and_button_navigation_round_trip(self) -> None:
        at = self.new_app("Home")
        self.assertEqual(len(at.sidebar.children), 0)
        self.assertEqual(at.session_state["page"], "Home")
        self.assertFalse(at.error, [e.value for e in at.error])

        at.button(key="cta_prediction").click().run()
        self.assert_clean(at)
        self.assertEqual(at.session_state["page"], "Prediction")
        self.assertFalse(at.error, [e.value for e in at.error])

        at.button(key="back_home").click().run()
        self.assert_clean(at)
        self.assertEqual(at.session_state["page"], "Home")
        self.assertTrue(at.button(key="cta_prediction"))

    def test_home_call_to_action_opens_prediction(self) -> None:
        at = self.new_app("Home")
        at.button(key="cta_prediction").click().run()
        self.assert_clean(at)
        self.assertEqual(at.session_state["page"], "Prediction")
        self.assertTrue(at.button(key="load_btn"))

    def test_state_survives_switching_pages(self) -> None:
        at = self.new_app()
        prediction = self.load_and_predict(at, 5, 150)
        at.radio(key="input_mode").set_value("Manual Sensor Input").run()
        at.number_input(key="manual_sensor_4").set_value(1410.0).run()
        at.radio(key="input_mode").set_value("Load Sample Engine").run()

        at.button(key="back_home").click().run()
        at.button(key="cta_prediction").click().run()
        self.assert_clean(at)

        self.assertEqual((at.session_state["selected_engine"], at.session_state["selected_cycle"]), (5, 150))
        page = html_text(at)
        self.assertIn("Telemetry loaded: Engine 5 • Cycle 150", page)
        self.assertIn(f'{prediction["rul"]:.1f}<span> cycles</span>', page)
        self.assertEqual(at.session_state["manual_sensor_4"], 1410.0)


if __name__ == "__main__":
    unittest.main()
