"""Planning utility functions that combine forecast, inventory, and stability terms."""

from typing import Dict, Mapping, Sequence

import numpy as np

from evaluation.stability_metrics import compute_absolute_plan_change


def _as_float_array(values: Sequence[float], name: str) -> np.ndarray:
    """Convert a numeric sequence into a one-dimensional float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("{} must be a one-dimensional sequence.".format(name))
    return array


def _component_sum(values: Sequence[float], name: str) -> float:
    """Return the finite sum of a planning-loss component."""
    array = _as_float_array(values, name)
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains non-finite values.".format(name))
    return float(np.sum(array))


def compute_execution_penalty(
    planning_signal: Sequence[float],
    execution_capacity: Sequence[float],
) -> np.ndarray:
    """Return the execution adaptation penalty for each period.

    The penalty is positive when the absolute plan change exceeds the execution
    capacity for that period. It directly measures the planning-infrastructure
    gap: the plan changes faster than the operation can absorb.

    This term makes execution constraints visible in the planning objective
    rather than treating them as an informal planner concern.
    """
    signal = _as_float_array(planning_signal, "planning_signal")
    capacity = _as_float_array(execution_capacity, "execution_capacity")
    if signal.shape != capacity.shape:
        raise ValueError("planning_signal and execution_capacity must have the same shape.")
    plan_change = compute_absolute_plan_change(signal)
    return np.maximum(plan_change - np.maximum(capacity, 0.0), 0.0)


def compute_total_planning_loss(
    forecast_error: Sequence[float],
    inventory_cost: Sequence[float],
    planning_signal_volatility: Sequence[float],
    model_switching_cost: Sequence[float],
    execution_adaptation_penalty: Sequence[float],
    weights: Mapping[str, float],
) -> float:
    """Return a weighted scalar planning loss.

    The scalar loss combines forecast accuracy, inventory cost, plan volatility,
    model switching, and execution adaptation. It matters because it gives a
    single objective for comparing planning strategies.

    The paper should not rely on this scalar alone. The scalar is a useful
    optimization device, while the individual components explain the tradeoffs
    created by the planning-infrastructure gap.
    """
    return float(
        float(weights.get("alpha_forecast", 1.0)) * _component_sum(forecast_error, "forecast_error")
        + float(weights.get("beta_inventory", 1.0)) * _component_sum(inventory_cost, "inventory_cost")
        + float(weights.get("lambda_volatility", 1.0))
        * _component_sum(planning_signal_volatility, "planning_signal_volatility")
        + float(weights.get("lambda_switch", 1.0)) * _component_sum(model_switching_cost, "model_switching_cost")
        + float(weights.get("lambda_execution", 1.0))
        * _component_sum(execution_adaptation_penalty, "execution_adaptation_penalty")
    )


def compute_multi_objective_summary(
    forecast_error: Sequence[float],
    inventory_cost: Sequence[float],
    planning_signal_volatility: Sequence[float],
    model_switching_cost: Sequence[float],
    execution_adaptation_penalty: Sequence[float],
) -> Dict[str, float]:
    """Return Pareto-style objective components for a planning strategy.

    The summary reports each objective separately so the paper can compare
    strategies without hiding tradeoffs inside one scalar score.

    This is important for the planning-infrastructure gap because a model can
    improve forecast error while worsening stability or execution adaptation.
    Separate objective columns make that failure mode visible.
    """
    components = {
        "forecast_error_total": _component_sum(forecast_error, "forecast_error"),
        "inventory_cost_total": _component_sum(inventory_cost, "inventory_cost"),
        "planning_signal_volatility_total": _component_sum(
            planning_signal_volatility, "planning_signal_volatility"
        ),
        "model_switching_cost_total": _component_sum(model_switching_cost, "model_switching_cost"),
        "execution_adaptation_penalty_total": _component_sum(
            execution_adaptation_penalty, "execution_adaptation_penalty"
        ),
    }
    components["forecast_error_mean"] = float(np.mean(_as_float_array(forecast_error, "forecast_error")))
    components["inventory_cost_mean"] = float(np.mean(_as_float_array(inventory_cost, "inventory_cost")))
    components["execution_penalty_period_share"] = float(
        np.mean(_as_float_array(execution_adaptation_penalty, "execution_adaptation_penalty") > 0.0)
    )
    return components
