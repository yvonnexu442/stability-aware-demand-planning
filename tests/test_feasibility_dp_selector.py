import unittest

import pandas as pd

from decision_layer.feasibility_dp_selector import (
    BudgetedDPFeasibilitySelector,
    DPFeasibilitySelector,
    GreedyFeasibilitySelector,
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
                        "actual": 9999.0,
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

    def test_deployable_selectors_do_not_use_test_actual(self):
        selector = DPFeasibilitySelector(
            expected_losses=self.expected_losses,
            weights=self.weights,
            switch_penalty=1.0,
            calibration_group_column="series_id",
        )
        first = selector.select(self.candidates)
        modified = self.candidates.copy()
        modified["actual"] = [0.0, 1.0, 2.0, 3.0]
        second = selector.select(modified)

        self.assertEqual(first["selected_model"].tolist(), second["selected_model"].tolist())


if __name__ == "__main__":
    unittest.main()
