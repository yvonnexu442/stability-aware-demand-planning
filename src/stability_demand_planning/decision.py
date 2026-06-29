"""Decision policies that convert forecasts into operational plan quantities."""

import pandas as pd


DEFAULT_POLICIES = (
    {"policy": "forecast_passthrough", "kind": "passthrough"},
    {"policy": "stability_blend_a0.35", "kind": "exponential_blend", "alpha": 0.35},
    {"policy": "change_cap_25pct", "kind": "change_cap", "max_change_fraction": 0.25, "min_change": 1.0},
)


def apply_decision_policies(forecasts, policies=DEFAULT_POLICIES):
    required = {"date", "item_id", "model", "forecast"}
    missing = required.difference(set(forecasts.columns))
    if missing:
        raise ValueError("Forecast table is missing required columns: {}".format(sorted(missing)))

    frames = []
    ordered = forecasts.copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered = ordered.sort_values(["model", "item_id", "date"])

    for policy in policies:
        policy_name = policy["policy"]
        kind = policy["kind"]
        for (model, item_id), group in ordered.groupby(["model", "item_id"]):
            plan_values = _apply_policy_to_group(group["forecast"].astype(float).values, policy)
            frame = group[["date", "item_id", "model", "forecast"]].copy()
            frame["policy"] = policy_name
            frame["plan_qty"] = plan_values
            frames.append(frame)

    decisions = pd.concat(frames, ignore_index=True)
    return decisions[["date", "item_id", "model", "policy", "forecast", "plan_qty"]]


def _apply_policy_to_group(forecasts, policy):
    kind = policy["kind"]
    if kind == "passthrough":
        return [max(0.0, float(value)) for value in forecasts]
    if kind == "exponential_blend":
        return _exponential_blend(forecasts, alpha=float(policy.get("alpha", 0.35)))
    if kind == "change_cap":
        return _change_cap(
            forecasts,
            max_change_fraction=float(policy.get("max_change_fraction", 0.25)),
            min_change=float(policy.get("min_change", 1.0)),
        )
    raise ValueError("Unknown decision policy kind '{}'".format(kind))


def _exponential_blend(forecasts, alpha):
    if len(forecasts) == 0:
        return []

    values = [max(0.0, float(forecasts[0]))]
    for forecast in forecasts[1:]:
        proposed = alpha * float(forecast) + (1.0 - alpha) * values[-1]
        values.append(max(0.0, proposed))
    return values


def _change_cap(forecasts, max_change_fraction, min_change):
    if len(forecasts) == 0:
        return []

    values = [max(0.0, float(forecasts[0]))]
    for forecast in forecasts[1:]:
        previous = values[-1]
        proposed = max(0.0, float(forecast))
        allowed_change = max(min_change, abs(previous) * max_change_fraction)
        lower = max(0.0, previous - allowed_change)
        upper = previous + allowed_change
        values.append(min(max(proposed, lower), upper))
    return values
