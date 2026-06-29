"""Simple baseline forecasts for daily demand panels."""

import numpy as np
import pandas as pd


DEFAULT_MODELS = ("naive_last", "rolling_mean_7", "rolling_mean_28", "seasonal_naive_7")


def forecast_holdout(panel, test_periods=28, models=DEFAULT_MODELS):
    """Create rolling-origin one-step forecasts across the holdout window."""
    panel = _validate_panel(panel)
    dates = sorted(panel["date"].unique())
    if len(dates) <= test_periods:
        raise ValueError("Need more dates than test_periods. Got {} dates.".format(len(dates)))

    forecast_dates = dates[-test_periods:]
    rows = []

    for item_id, item_frame in panel.groupby("item_id"):
        item_frame = item_frame.sort_values("date")
        for date in forecast_dates:
            train = item_frame[item_frame["date"] < date]
            if train.empty:
                continue
            train_values = train["demand"].astype(float).values
            for model in models:
                forecast = _predict_series(train_values, 1, model)[0]
                rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "item_id": str(item_id),
                        "model": model,
                        "forecast": max(0.0, float(forecast)),
                    }
                )

    return pd.DataFrame(rows, columns=["date", "item_id", "model", "forecast"])


def _predict_series(train_values, horizon, model):
    if model == "naive_last":
        return np.repeat(train_values[-1], horizon)
    if model == "rolling_mean_7":
        return np.repeat(np.mean(train_values[-7:]), horizon)
    if model == "rolling_mean_28":
        return np.repeat(np.mean(train_values[-28:]), horizon)
    if model == "seasonal_naive_7":
        pattern = train_values[-7:] if len(train_values) >= 7 else train_values
        repeats = int(np.ceil(float(horizon) / float(len(pattern))))
        return np.tile(pattern, repeats)[:horizon]
    raise ValueError("Unknown model '{}'".format(model))


def _validate_panel(panel):
    required = {"date", "item_id", "demand"}
    missing = required.difference(set(panel.columns))
    if missing:
        raise ValueError("Panel is missing required columns: {}".format(sorted(missing)))

    validated = panel[["date", "item_id", "demand"]].copy()
    validated["date"] = pd.to_datetime(validated["date"])
    validated["item_id"] = validated["item_id"].astype(str)
    validated["demand"] = validated["demand"].astype(float)
    return validated.sort_values(["item_id", "date"]).reset_index(drop=True)
