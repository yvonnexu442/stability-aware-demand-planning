# Beyond Forecast Accuracy: A Stability-Aware Decision Layer for Operational Demand Planning

## Working Claim

Operational planning quality is not equivalent to forecast accuracy. In many settings, teams need plans that are accurate enough, stable enough, and feasible enough to execute. This paper studies a simple decision layer that sits after the forecasting model and explicitly exposes the accuracy-stability-executability tradeoff.

## Motivation

Forecasting research often optimizes point accuracy metrics such as MAE, RMSE, MAPE, or WAPE. In operational demand planning, however, the forecast is usually an input to downstream decisions: purchase orders, replenishment quantities, production schedules, labor plans, or allocation decisions.

Real planning systems do not only need accurate forecasts. They also need stable and executable planning signals. A numerically better forecast can still require frequent model switching, large plan changes, or infrastructure updates that the operation cannot absorb.

## Research Gap

Existing operational planning workflows often treat execution stability as an implicit planner judgment rather than an explicit optimization or evaluation target. This creates a planning-infrastructure gap: the forecasting layer can adapt faster than the execution system. As a result, standard forecast benchmarks can favor models whose outputs are difficult to execute.

## Proposed Framework

1. Normalize public demand tables into a common daily panel schema.
2. Generate baseline forecasts using simple transparent models.
3. Convert forecasts into operational plan quantities.
4. Compare a direct forecast passthrough policy with stability-aware decision policies.
5. Evaluate both forecast error and decision outcomes.

## Initial Hypotheses

- H1: Lower forecast error does not always imply lower decision cost.
- H2: Stability-aware decision layers can materially reduce plan variation while preserving acceptable service-level performance.
- H3: Reporting an accuracy-stability frontier can reveal tradeoffs that are hidden by forecast metrics alone.
- H4: Forecasting improvements are most operationally valuable when their update frequency and plan volatility remain within the execution system's absorption capacity.

## Experimental Design

Initial datasets:

- Synthetic panel demand for smoke tests and controlled perturbations.
- UCI Bike Sharing daily demand as a clean public time-series table.
- UCI Online Retail transactions aggregated into daily SKU-level demand.

Initial baselines:

- Last-value naive forecast.
- 7-day rolling mean forecast.
- 28-day rolling mean forecast.
- 7-day seasonal naive forecast.

Initial decision policies:

- Forecast passthrough.
- Exponential stability blend.
- Maximum period-to-period change cap.

Initial metrics:

- Forecast: MAE, RMSE, WAPE, bias.
- Decision: underage, overage, asymmetric decision cost, service proxy, total plan variation.

## Notes for NIW Narrative

This project can be framed around improving operational resilience and planning quality in data-driven supply, retail, and service systems. The early contribution is not a complex forecasting model; it is a decision-aware evaluation layer that makes execution capability visible.
