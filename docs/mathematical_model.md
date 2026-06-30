# Mathematical Model

This document defines the first research model for stability-aware operational demand planning under execution constraints.

## Notation

- `i`: planning unit, such as item-store or store-family.
- `t`: planning period.
- `m`: candidate forecasting model.
- `y_i,t`: actual demand.
- `yhat_i,t^m`: forecast from model `m`.
- `w_i,t^m`: model weight if using an ensemble.
- `ytilde_i,t`: final selected or combined forecast.
- `x_i,t`: final planning signal, such as an inventory target.
- `C_i,t`: execution capacity, meaning the maximum plan change the operation can absorb.
- `z_i,t`: selected model if using hard model selection.

## Forecast Selection and Composition

Hard model selection chooses one candidate model at each planning unit and period:

```text
ytilde_i,t = yhat_i,t^{z_i,t}
```

Soft composition combines candidate forecasts through ensemble weights:

```text
ytilde_i,t = sum_m w_i,t^m * yhat_i,t^m
```

The ensemble weights must satisfy:

```text
w_i,t^m >= 0
sum_m w_i,t^m = 1
```

## Planning Signal

The final planning signal converts the selected or combined forecast into an executable target:

```text
x_i,t = ytilde_i,t + safety_stock_i,t
```

The planning signal is the object that downstream operations execute. This distinction matters because an operation does not execute forecast accuracy; it executes inventory targets, replenishment quantities, staffing levels, or production signals.

## Smoothed Planning Signal

When execution infrastructure cannot absorb abrupt plan changes, the decision layer can adapt the executable plan gradually:

```text
candidate_plan_i,t = ytilde_i,t + safety_stock_i,t

final_plan_i,t =
  alpha_i,t * candidate_plan_i,t
  + (1 - alpha_i,t) * final_plan_i,t-1
```

The smoothing parameter `alpha_i,t` controls the adaptation speed. A value of `1.00` means the candidate plan is executed immediately with no smoothing. Lower values represent slower operational adaptation. The parameter can be fixed, scenario-based, or selected from a small validation grid using normalized planning loss.

Smoothing is used to model gradual operational adaptation when execution infrastructure cannot absorb abrupt plan changes. It does not change the candidate forecast itself; it changes how quickly the forecast-driven plan becomes the executable planning signal.

## Multi-objective Planning Loss

The project evaluates planning utility as a weighted multi-objective loss:

```text
total_loss =
  alpha * forecast_error
  + beta * inventory_cost
  + lambda_volatility * planning_signal_volatility
  + lambda_switch * model_switching_cost
  + lambda_execution * execution_adaptation_penalty
```

This objective makes forecast accuracy only one part of the evaluation. A model can reduce prediction error while increasing operational burden.

## Normalized Planning Loss

Raw operational metrics are still reported in their natural units. However, the main scalar ranking objective is also reported as a normalized planning loss so that the dollar-scale inventory term does not mechanically dominate execution feasibility terms.

For each dataset, run mode, and split, the accuracy-first BestAccuracy or Global Best strategy is used as the default reference:

```text
normalized_total_loss =
  beta * inventory_cost / inventory_cost_ref
  + lambda_volatility * planning_signal_volatility / volatility_ref
  + lambda_execution * execution_penalty / execution_penalty_ref
  + lambda_switch * model_switch_count / switch_count_ref
```

The reference values are computed within the same run mode and split. For example, quick-mode results use the quick-mode Global Best reference, while medium-mode results use the medium-mode Global Best reference. If a reference value is zero or unavailable, the implementation uses the median nonzero strategy value as a fallback and records the fallback in an audit table.

Normalized loss is not a replacement for raw metrics. It is a scalar comparison device that lets the paper examine how operational preferences change when execution feasibility receives more weight.

## Inventory Cost

The inventory component penalizes both excess and insufficient planning signals:

```text
holding_cost = holding_cost_rate * max(x_i,t - y_i,t, 0)
shortage_cost = shortage_cost_rate * max(y_i,t - x_i,t, 0)
inventory_cost = holding_cost + shortage_cost
```

Holding cost captures surplus inventory or over-allocation. Shortage cost captures unmet demand, service failures, or under-allocation.

## Planning Signal Volatility

The planning signal volatility term measures how sharply the executable plan changes across periods:

```text
plan_change_pct = abs(x_i,t - x_i,t-1) / max(abs(x_i,t-1), epsilon)
```

High plan volatility can create coordination cost even when forecast error improves.

## Model Switching Cost

For hard model selection, model switching cost records when the selected model changes:

```text
switch_cost = 1 if z_i,t != z_i,t-1 else 0
```

For ensemble selection, the analogous instability measure is total weight movement:

```text
weight_change = sum_m abs(w_i,t^m - w_i,t-1^m)
```

These terms capture model governance and infrastructure burden. Frequent model switching can require retraining, validation, deployment, monitoring, and planner communication.

## Execution Penalty

Execution capacity limits how much the operation can absorb in a single planning step:

```text
execution_penalty = max(0, abs(x_i,t - x_i,t-1) - C_i,t)
```

The execution penalty captures the planning-infrastructure gap. It becomes positive when the forecast or planning signal changes faster than execution infrastructure can absorb. In practical terms, the forecast layer may be capable of adapting every day, while procurement, staffing, production, transportation, or software infrastructure may only absorb smaller or slower changes.

## Reporting Requirements

The project should report both:

- A weighted scalar planning loss, which is useful for optimization and policy comparison.
- Pareto-style multi-objective outputs, which preserve the tradeoffs among forecast error, inventory cost, stability, switching, and execution adaptation.

The Pareto-style view is essential for the paper because it shows when an apparently better forecast creates operational instability.
