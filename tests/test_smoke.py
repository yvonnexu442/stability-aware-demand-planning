import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stability_demand_planning.data import make_synthetic_panel
from stability_demand_planning.decision import apply_decision_policies
from stability_demand_planning.forecast import forecast_holdout
from stability_demand_planning.metrics import evaluate_decisions, evaluate_forecasts


class PipelineSmokeTest(unittest.TestCase):
    def test_synthetic_pipeline_runs(self):
        panel = make_synthetic_panel(n_items=3, n_periods=90, seed=11)
        forecasts = forecast_holdout(panel, test_periods=14)
        decisions = apply_decision_policies(forecasts)
        forecast_metrics = evaluate_forecasts(panel, forecasts)
        decision_metrics = evaluate_decisions(panel, decisions)

        self.assertFalse(panel.empty)
        self.assertFalse(forecasts.empty)
        self.assertFalse(decisions.empty)
        self.assertFalse(forecast_metrics.empty)
        self.assertFalse(decision_metrics.empty)
        self.assertIn("wape", forecast_metrics.columns)
        self.assertIn("normalized_plan_variation", decision_metrics.columns)


if __name__ == "__main__":
    unittest.main()
