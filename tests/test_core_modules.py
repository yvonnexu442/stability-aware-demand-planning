import unittest

import numpy as np
import pandas as pd

from evaluation.forecast_metrics import mean_absolute_error, weighted_absolute_percentage_error
from evaluation.inventory_metrics import compute_inventory_target, compute_service_level
from evaluation.planning_utility import add_oracle_gap_columns, compute_execution_penalty, compute_total_planning_loss
from evaluation.stability_metrics import compute_model_switch_count, compute_percentage_plan_change
from planning_environment.planning_simulator import evaluate_planning_strategy, simulate_planning_outcomes


class CoreModuleSmokeTest(unittest.TestCase):
    def test_core_metrics_and_simulator(self):
        actual = np.array([10.0, 12.0, 11.0, 18.0])
        forecast = np.array([9.0, 13.0, 12.0, 15.0])
        safety_stock = np.array([2.0, 2.0, 2.0, 2.0])
        target = compute_inventory_target(forecast, safety_stock)

        self.assertAlmostEqual(mean_absolute_error(actual, forecast), 1.5)
        self.assertGreater(weighted_absolute_percentage_error(actual, forecast), 0.0)
        self.assertGreaterEqual(compute_service_level(target, actual), 0.0)
        self.assertEqual(compute_model_switch_count(["a", "a", "b", "b"]), 1)
        self.assertEqual(compute_percentage_plan_change(target).shape[0], target.shape[0])

        execution_penalty = compute_execution_penalty(target, np.array([0.0, 2.0, 2.0, 2.0]))
        total_loss = compute_total_planning_loss(
            forecast_error=np.abs(actual - forecast),
            inventory_cost=np.ones_like(actual),
            planning_signal_volatility=compute_percentage_plan_change(target),
            model_switching_cost=np.array([0.0, 0.0, 1.0, 0.0]),
            execution_adaptation_penalty=execution_penalty,
            weights={
                "alpha_forecast": 1.0,
                "beta_inventory": 1.0,
                "lambda_volatility": 0.5,
                "lambda_switch": 0.5,
                "lambda_execution": 1.0,
            },
        )
        self.assertGreater(total_loss, 0.0)

        simulation = simulate_planning_outcomes(
            actual_demand=actual,
            forecast=forecast,
            safety_stock=safety_stock,
            holding_cost_rate=1.0,
            shortage_cost_rate=5.0,
            max_plan_change_rate=0.25,
        )
        summary = evaluate_planning_strategy(simulation)
        self.assertIn("weighted_absolute_percentage_error", summary)
        self.assertIn("execution_adaptation_penalty_total", summary)

    def test_missing_alpha_forecast_defaults_to_zero(self):
        total_loss = compute_total_planning_loss(
            forecast_error=np.array([100.0]),
            inventory_cost=np.array([2.0]),
            planning_signal_volatility=np.array([3.0]),
            model_switching_cost=np.array([4.0]),
            execution_adaptation_penalty=np.array([5.0]),
            weights={},
        )

        self.assertAlmostEqual(total_loss, 2.0 + 0.10 * 3.0 + 0.05 * 4.0 + 0.10 * 5.0)

    def test_oracle_gaps_are_group_scoped(self):
        table = add_oracle_gap_columns(
            pd.DataFrame(
                [
                    {
                        "scenario_name": "baseline",
                        "strategy": "oracle_dp_feasibility_selector",
                        "normalized_total_loss": 1.0,
                    },
                    {
                        "scenario_name": "baseline",
                        "strategy": "oracle_realized_demand",
                        "normalized_total_loss": 0.5,
                    },
                    {
                        "scenario_name": "baseline",
                        "strategy": "global_best_model",
                        "normalized_total_loss": 1.4,
                    },
                    {
                        "scenario_name": "stress",
                        "strategy": "oracle_dp_feasibility_selector",
                        "normalized_total_loss": 10.0,
                    },
                    {
                        "scenario_name": "stress",
                        "strategy": "oracle_realized_demand",
                        "normalized_total_loss": 6.0,
                    },
                    {
                        "scenario_name": "stress",
                        "strategy": "global_best_model",
                        "normalized_total_loss": 11.0,
                    },
                ]
            )
        )

        baseline = table[(table["scenario_name"] == "baseline") & (table["strategy"] == "global_best_model")].iloc[0]
        stress = table[(table["scenario_name"] == "stress") & (table["strategy"] == "global_best_model")].iloc[0]
        self.assertAlmostEqual(baseline["gap_to_dp_oracle"], 0.4)
        self.assertAlmostEqual(baseline["gap_to_perfect_oracle"], 0.9)
        self.assertAlmostEqual(stress["gap_to_dp_oracle"], 1.0)
        self.assertAlmostEqual(stress["gap_to_perfect_oracle"], 5.0)


if __name__ == "__main__":
    unittest.main()
