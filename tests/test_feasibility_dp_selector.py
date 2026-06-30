import unittest
import warnings

import pandas as pd

from decision_layer.feasibility_dp_selector import (
    BudgetedDPFeasibilitySelector,
    DPFeasibilitySelector,
    GreedyFeasibilitySelector,
    OracleDPFeasibilitySelector,
)


class FeasibilityDPSelectorTest(unittest.TestCase):
    def setUp(self):
        rows = []
        for date, forecasts in [
            ("2026-01-01", {"A": 100.0, "B": 100.0}),
            ("2026-01-02", {"A": 100.0, "B": 1000.0}),
        ]:
            for model_name, forecast in forecasts.items():
                rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "series_id": "item_1",
                        "family": "family_1",
                        "store_nbr": 1,
                        "model_name": model_name,
                        "selected_model": model_name,
                        "forecast": forecast,
                        "split": "test",
                        "horizon": 1,
                        "safety_stock": 0.0,
                    }
                )
        self.candidates = pd.DataFrame(rows)
        self.expected_losses = {
            ("item_1", "A"): {"wape": 0.20, "inventory_cost_per_demand_unit": 0.0},
            ("item_1", "B"): {"wape": 0.10, "inventory_cost_per_demand_unit": 0.0},
            ("global", "A"): {"wape": 0.20, "inventory_cost_per_demand_unit": 0.0},
            ("global", "B"): {"wape": 0.10, "inventory_cost_per_demand_unit": 0.0},
        }
        self.weights = {
            "alpha_forecast": 1.0,
            "beta_inventory": 0.0,
            "lambda_volatility": 1.0,
            "lambda_switch": 1.0,
            "lambda_execution": 0.0,
        }

    def test_greedy_and_dp_can_choose_different_paths(self):
        greedy = GreedyFeasibilitySelector(
            expected_losses=self.expected_losses,
            weights=self.weights,
            switch_penalty=1.0,
            calibration_group_column="series_id",
        ).select(self.candidates)
        dp = DPFeasibilitySelector(
            expected_losses=self.expected_losses,
            weights=self.weights,
            switch_penalty=1.0,
            calibration_group_column="series_id",
        ).select(self.candidates)

        self.assertEqual(greedy["selected_model"].tolist(), ["B", "A"])
        self.assertEqual(dp["selected_model"].tolist(), ["A", "A"])

    def test_budgeted_dp_enforces_switch_budget(self):
        budgeted = BudgetedDPFeasibilitySelector(
            expected_losses=self.expected_losses,
            weights=self.weights,
            switch_penalty=1.0,
            max_switches=0,
            calibration_group_column="series_id",
        ).select(self.candidates)

        self.assertEqual(budgeted["selected_model"].tolist(), ["A", "A"])
        switches = sum(left != right for left, right in zip(budgeted["selected_model"], budgeted["selected_model"].iloc[1:]))
        self.assertEqual(switches, 0)

    def test_budgeted_dp_falls_back_when_budget_path_is_missing(self):
        candidates = self.candidates[
            ((self.candidates["date"] == pd.Timestamp("2026-01-01")) & (self.candidates["model_name"] == "A"))
            | ((self.candidates["date"] == pd.Timestamp("2026-01-02")) & (self.candidates["model_name"] == "B"))
        ].copy()
        budgeted = BudgetedDPFeasibilitySelector(
            expected_losses=self.expected_losses,
            weights=self.weights,
            switch_penalty=1.0,
            max_switches=0,
            calibration_group_column="series_id",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selected = budgeted.select(candidates)

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected["selected_model"].tolist(), ["A", "B"])
        self.assertTrue(selected["fallback_used"].all())
        self.assertEqual(selected["fallback_type"].unique().tolist(), ["incumbent_stays"])
        fallback_reason = selected["fallback_reason"].unique().tolist()[0]
        self.assertIn("no_budget_feasible_path_under_max_switches_0", fallback_reason)
        self.assertIn("incumbent_missing_in_later_period_potentially_non_strict", fallback_reason)
        warning_messages = [str(item.message) for item in caught]
        self.assertTrue(any("Falling back to incumbent-stays policy" in message for message in warning_messages))
        self.assertTrue(any("potentially non-strict" in message for message in warning_messages))

    def test_budgeted_dp_incumbent_stays_fallback_uses_zero_switches_when_available(self):
        budgeted = BudgetedDPFeasibilitySelector(
            expected_losses=self.expected_losses,
            weights=self.weights,
            switch_penalty=1.0,
            max_switches=0,
            calibration_group_column="series_id",
        )

        rows = budgeted._incumbent_stays_fallback_series(self.candidates)
        selected_models = [str(row["model_name"]) for row in rows]

        self.assertEqual(selected_models, ["B", "B"])
        self.assertTrue(all(bool(row["fallback_used"]) for row in rows))
        self.assertEqual({str(row["fallback_type"]) for row in rows}, {"incumbent_stays"})
        switches = sum(left != right for left, right in zip(selected_models[:-1], selected_models[1:]))
        self.assertEqual(switches, 0)

    def test_oracle_dp_uses_realized_inventory_costs(self):
        oracle = OracleDPFeasibilitySelector(
            expected_losses=self.expected_losses,
            realized_inventory_costs={
                ("item_1", "A", pd.Timestamp("2026-01-01")): 10.0,
                ("item_1", "A", pd.Timestamp("2026-01-02")): 10.0,
                ("item_1", "B", pd.Timestamp("2026-01-01")): 0.0,
                ("item_1", "B", pd.Timestamp("2026-01-02")): 0.0,
            },
            weights={
                "alpha_forecast": 0.0,
                "beta_inventory": 1.0,
                "lambda_volatility": 0.0,
                "lambda_switch": 0.0,
                "lambda_execution": 0.0,
            },
            switch_penalty=1.0,
            calibration_group_column="series_id",
        )

        selected = oracle.select(self.candidates)

        self.assertEqual(selected["selected_model"].tolist(), ["B", "B"])

    def test_oracle_dp_uses_period_specific_realized_inventory_costs(self):
        oracle = OracleDPFeasibilitySelector(
            expected_losses=self.expected_losses,
            realized_inventory_costs={
                ("item_1", "A", pd.Timestamp("2026-01-01")): 0.0,
                ("item_1", "B", pd.Timestamp("2026-01-01")): 10.0,
                ("item_1", "A", pd.Timestamp("2026-01-02")): 10.0,
                ("item_1", "B", pd.Timestamp("2026-01-02")): 0.0,
            },
            weights={
                "alpha_forecast": 0.0,
                "beta_inventory": 1.0,
                "lambda_volatility": 0.0,
                "lambda_switch": 0.0,
                "lambda_execution": 0.0,
            },
            switch_penalty=0.0,
            calibration_group_column="series_id",
        )

        selected = oracle.select(self.candidates)

        self.assertEqual(selected["selected_model"].tolist(), ["A", "B"])

    def test_deployable_selectors_reject_test_actual(self):
        selector = DPFeasibilitySelector(
            expected_losses=self.expected_losses,
            weights=self.weights,
            switch_penalty=1.0,
            calibration_group_column="series_id",
        )
        modified = self.candidates.copy()
        modified["actual"] = [0.0, 1.0, 2.0, 3.0]

        with self.assertRaises(ValueError):
            selector.select(modified)


if __name__ == "__main__":
    unittest.main()
