# Problem Statement

## Current Draft

Existing forecasting research usually evaluates models by prediction error. However, real operational planning systems do not only need accurate forecasts; they also need stable and executable planning signals. A numerically better forecast may require frequent model switching, large plan changes, or infrastructure updates that the operation cannot absorb. This creates a planning-infrastructure gap: the forecasting layer can adapt faster than the execution system.

## Research Formulation

The central problem is not simply that forecasts are wrong. It is that forecast optimization and operational execution often move at different speeds. A forecasting layer can update models, parameters, or demand signals more frequently than procurement, staffing, production, allocation, or software infrastructure can respond.

This mismatch means that a forecast with lower prediction error may still degrade operational planning quality if it increases plan churn, forces frequent model switching, or requires execution capabilities that are not available in the planning environment.

## Mechanism to Test

1. Forecast models are evaluated on prediction error.
2. Operational plans are evaluated on cost, service, feasibility, and stability.
3. A more accurate forecast can produce more volatile planning signals.
4. A stability-aware decision layer can smooth or constrain those signals.
5. The relevant empirical question is whether the decision layer improves execution-facing outcomes without sacrificing too much demand responsiveness.

## Candidate Contribution

This paper proposes and tests a stability-aware decision layer for operational demand planning. Instead of replacing forecasting models, the layer evaluates and transforms forecast outputs according to downstream execution criteria. The contribution is a decision-aware evaluation framework that makes the planning-infrastructure gap measurable.
