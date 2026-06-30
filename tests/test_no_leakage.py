import unittest

import pandas as pd

from decision_layer.no_leakage import attach_actuals_for_evaluation, drop_future_outcomes, require_no_future_outcomes


class NoLeakageBoundaryTest(unittest.TestCase):
    def test_drop_future_outcomes_and_attach_for_evaluation(self):
        source = pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-01-01"), "series_id": "s1", "forecast": 10.0, "actual": 12.0},
                {"date": pd.Timestamp("2026-01-02"), "series_id": "s1", "forecast": 11.0, "actual": 13.0},
            ]
        )
        deployable = drop_future_outcomes(source)
        self.assertNotIn("actual", deployable.columns)
        require_no_future_outcomes(deployable, "unit_test")

        selected = deployable.assign(strategy="test_strategy", selected_model="test_model")
        evaluated = attach_actuals_for_evaluation(selected, source)
        self.assertEqual(evaluated["actual"].tolist(), [12.0, 13.0])

    def test_require_no_future_outcomes_rejects_actual(self):
        frame = pd.DataFrame([{"date": pd.Timestamp("2026-01-01"), "series_id": "s1", "actual": 1.0}])
        with self.assertRaises(ValueError):
            require_no_future_outcomes(frame, "unit_test")


if __name__ == "__main__":
    unittest.main()
