"""Run a baseline stability-aware demand planning experiment."""

import argparse
import json
from pathlib import Path

from stability_demand_planning.data import load_dataset
from stability_demand_planning.decision import apply_decision_policies
from stability_demand_planning.forecast import forecast_holdout
from stability_demand_planning.metrics import combine_metrics, evaluate_decisions, evaluate_forecasts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic_demo")
    parser.add_argument("--registry", default="configs/datasets.json")
    parser.add_argument("--experiment-config", default="configs/experiment_default.json")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--test-periods", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_json(args.experiment_config)
    test_periods = args.test_periods or int(config.get("test_periods", 28))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_dataset(
        args.dataset,
        registry_path=args.registry,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        force_download=args.force_download,
    )
    forecasts = forecast_holdout(panel, test_periods=test_periods)
    decisions = apply_decision_policies(forecasts, policies=config.get("decision_policies"))

    forecast_metrics = evaluate_forecasts(panel, forecasts)
    decision_metrics = evaluate_decisions(
        panel,
        decisions,
        underage_cost=float(config.get("underage_cost", 3.0)),
        overage_cost=float(config.get("overage_cost", 1.0)),
    )
    combined = combine_metrics(forecast_metrics, decision_metrics)

    prefix = output_dir / args.dataset
    forecasts.to_csv(str(prefix) + "_forecasts.csv", index=False)
    decisions.to_csv(str(prefix) + "_decisions.csv", index=False)
    forecast_metrics.to_csv(str(prefix) + "_forecast_metrics.csv", index=False)
    decision_metrics.to_csv(str(prefix) + "_decision_metrics.csv", index=False)
    combined.to_csv(str(prefix) + "_combined_metrics.csv", index=False)

    columns = [
        "model",
        "policy",
        "wape",
        "cost_per_demand_unit",
        "service_proxy",
        "normalized_plan_variation",
    ]
    print(combined[columns].to_string(index=False))
    print("")
    print("Wrote outputs with prefix: {}".format(prefix))


def read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
