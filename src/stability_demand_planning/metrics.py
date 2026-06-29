"""Forecast and decision metrics."""

import numpy as np
import pandas as pd


def evaluate_forecasts(actuals, forecasts):
    merged = _merge_actuals(actuals, forecasts)
    rows = []

    for model, group in merged.groupby("model"):
        actual = group["demand"].astype(float).values
        predicted = group["forecast"].astype(float).values
        error = predicted - actual
        rows.append(
            {
                "model": model,
                "n_obs": int(len(group)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "wape": _safe_divide(np.sum(np.abs(error)), np.sum(np.abs(actual))),
                "bias": _safe_divide(np.sum(error), np.sum(actual)),
            }
        )

    return pd.DataFrame(rows).sort_values(["wape", "mae"]).reset_index(drop=True)


def evaluate_decisions(actuals, decisions, underage_cost=3.0, overage_cost=1.0):
    merged = _merge_actuals(actuals, decisions)
    merged = merged.sort_values(["model", "policy", "item_id", "date"]).copy()
    merged["underage"] = np.maximum(merged["demand"] - merged["plan_qty"], 0.0)
    merged["overage"] = np.maximum(merged["plan_qty"] - merged["demand"], 0.0)
    merged["decision_cost"] = underage_cost * merged["underage"] + overage_cost * merged["overage"]
    merged["plan_delta"] = (
        merged.groupby(["model", "policy", "item_id"])["plan_qty"]
        .diff()
        .abs()
        .fillna(0.0)
    )

    rows = []
    for (model, policy), group in merged.groupby(["model", "policy"]):
        demand_sum = float(group["demand"].sum())
        plan_sum = float(group["plan_qty"].sum())
        underage_sum = float(group["underage"].sum())
        overage_sum = float(group["overage"].sum())
        variation_sum = float(group["plan_delta"].sum())
        rows.append(
            {
                "model": model,
                "policy": policy,
                "n_obs": int(len(group)),
                "underage_units": underage_sum,
                "overage_units": overage_sum,
                "decision_cost": float(group["decision_cost"].sum()),
                "cost_per_demand_unit": _safe_divide(group["decision_cost"].sum(), demand_sum),
                "service_proxy": 1.0 - _safe_divide(underage_sum, demand_sum),
                "total_plan_variation": variation_sum,
                "normalized_plan_variation": _safe_divide(variation_sum, plan_sum),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["cost_per_demand_unit", "normalized_plan_variation"])
        .reset_index(drop=True)
    )


def combine_metrics(forecast_metrics, decision_metrics):
    return decision_metrics.merge(forecast_metrics, on="model", suffixes=("_decision", "_forecast"))


def _merge_actuals(actuals, predictions):
    actual_frame = actuals[["date", "item_id", "demand"]].copy()
    actual_frame["date"] = pd.to_datetime(actual_frame["date"])
    actual_frame["item_id"] = actual_frame["item_id"].astype(str)
    actual_frame["demand"] = actual_frame["demand"].astype(float)

    prediction_frame = predictions.copy()
    prediction_frame["date"] = pd.to_datetime(prediction_frame["date"])
    prediction_frame["item_id"] = prediction_frame["item_id"].astype(str)

    merged = prediction_frame.merge(actual_frame, on=["date", "item_id"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping dates/items between actuals and predictions.")
    return merged


def _safe_divide(numerator, denominator):
    denominator = float(denominator)
    if denominator == 0.0:
        return 0.0
    return float(numerator) / denominator
