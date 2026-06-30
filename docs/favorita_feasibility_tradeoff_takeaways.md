# Favorita Feasibility Tradeoff Takeaways

These notes summarize the current quick-mode Favorita feasibility stress-test and Pareto-analysis outputs. They are preliminary research takeaways, not final paper claims. The results should be rechecked after larger runs, additional datasets, and any changes to the planning-loss weights.

## Main Takeaway

The Favorita feasibility analysis supports the central thesis that operational demand planning is a multi-objective feasibility problem, not a pure forecast-accuracy leaderboard. Under the current quick-mode Favorita setting, the accuracy-only LightGBM/global-best strategy remains strongest on weighted forecast error, inventory cost, and baseline total planning loss. However, the stability-aware and feasibility-aware strategies reduce planning volatility, execution adaptation penalty, and execution violation rate, which makes their value visible when execution capacity is constrained or execution burden receives more weight.

This is a useful research signal because the desired claim is not that a stability-aware selector should always dominate. The stronger claim is that forecast accuracy, inventory cost, planning stability, and execution burden should be reported separately and jointly.

## Weight Sensitivity

The weight-sensitivity grid shows that total-loss winners depend on the planning-loss weights.

- For low execution-weight settings, the global-best/LightGBM strategy is preferred because the forecast-accuracy and inventory-cost advantages dominate the loss.
- When `lambda_execution` increases, feasibility-aware or stability-aware selection can become preferred.
- In the current grid, global-best/LightGBM ties for best total loss in 18 of 30 weight configurations, feasibility-aware selection is best in 4 configurations, stability-aware selection is best in 6 configurations, and XGBoost is best in 2 configurations.
- This pattern supports a tradeoff-surface interpretation: feasibility-aware planning becomes more attractive when execution burden is expensive, but it should not be forced to win under all weights.

## Execution Capacity Stress Test

The execution-capacity stress test directly probes the planning-infrastructure gap.

- As capacity tightens from high capacity to severe constraint, accuracy-only planning experiences sharply higher execution violation rates and execution penalties.
- The global-best/LightGBM execution violation rate rises from about 3.8 percent in the high-capacity scenario to about 76.2 percent in the severe-constraint scenario.
- The feasibility-aware selector has lower execution burden across the same scenarios. Its violation rate rises from about 2.9 percent to about 68.5 percent, which is still high under severe constraints but lower than the accuracy-only strategy.
- Under the current baseline weights, the accuracy-only strategy can still retain lower total planning loss because its forecast and inventory-cost advantages are large.
- This is an important result: constrained execution infrastructure changes the operational risk profile even when scalar total loss does not fully flip.

## Pareto Analysis

The Pareto analysis shows why scalar loss alone is insufficient.

- Global-best/LightGBM, family-best, XGBoost, stability-aware, feasibility-aware, and moving-average strategies are Pareto efficient under the current objective set.
- Seasonal naive, exponential smoothing, and naive last-value baselines are dominated in the current quick-mode result.
- The moving-average baseline is Pareto efficient only because it is extremely stable and has near-zero execution burden. It is not operationally attractive on forecast accuracy or inventory cost.
- Feasibility-aware selection is not the most accurate strategy, but it occupies a meaningful tradeoff position with lower volatility and lower execution penalty than the accuracy-only model.

## Interpretation for the Paper

The feasibility stress-test and Pareto-analysis layer should be framed as evidence for a feasibility-aware evaluation layer rather than a claim that one selector universally wins. The results currently suggest:

- If execution infrastructure is flexible, accuracy-only planning may be reasonable.
- If execution changes are costly or capacity is constrained, feasibility-aware decision layers become more important.
- If inventory shortage or service-level costs dominate the objective, accuracy may still be prioritized.
- Therefore, the paper should report separate metrics, scalar planning loss, and Pareto tradeoffs together.

The empirical story is strongest when the paper emphasizes where each method sits on the accuracy-stability-execution tradeoff surface.
