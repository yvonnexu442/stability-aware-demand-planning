"""Helpers for running candidate forecast models."""

from typing import Dict, Sequence

import numpy as np

from models.baselines import naive_last_value_forecast, rolling_mean_forecast, seasonal_naive_forecast


def run_baseline_forecast(
    history: Sequence[float],
    horizon: int,
    model_name: str,
    rolling_window_size: int = 56,
    season_length: int = 7,
) -> np.ndarray:
    """Run one transparent baseline forecast by name.

    This function gives later experiments a common entry point without adding
    full dataset logic yet. It matters for the paper because every forecast
    model should feed the same decision layer and planning utility metrics.
    """
    if model_name == "naive_last_value":
        return naive_last_value_forecast(history, horizon)
    if model_name == "rolling_mean":
        return rolling_mean_forecast(history, horizon, rolling_window_size)
    if model_name == "seasonal_naive":
        return seasonal_naive_forecast(history, horizon, season_length)
    raise ValueError("Unknown baseline model: {}".format(model_name))


def run_candidate_forecasts(
    history: Sequence[float],
    horizon: int,
    models_to_run: Sequence[str],
    rolling_window_size: int = 56,
) -> Dict[str, np.ndarray]:
    """Return forecasts for all requested transparent baseline models.

    Candidate forecasts are inputs to the decision layer. The function is small
    by design so research code can clearly separate forecasting from planning
    signal selection and execution-aware evaluation.
    """
    return {
        model_name: run_baseline_forecast(
            history,
            horizon=horizon,
            model_name=model_name,
            rolling_window_size=rolling_window_size,
        )
        for model_name in models_to_run
    }
