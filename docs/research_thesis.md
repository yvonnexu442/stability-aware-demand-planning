# Research Thesis

Forecast-driven demand planning should be treated as a feasibility-constrained model selection problem rather than a pure forecast-accuracy ranking problem. The forecast with the best WAPE is not necessarily the best deployable planning strategy once inventory exposure, planning-signal volatility, model-switching burden, and execution-capacity violations are considered.

Using real retail demand data and supply-chain execution evidence, this project tests the working thesis that execution risk is empirically measurable and can reorder model deployment choices. The decision layer compares forecast accuracy with inventory impact, planning-signal volatility, model-switching cost, execution adaptation burden, and finite-horizon selection effects.

The central claim is that operational planning utility should be evaluated separately and jointly across these dimensions. The paper therefore quantifies how often accuracy-first ranking disagrees with operational-loss ranking, how large the planning-execution gap rate is, what finite-horizon DP buys over greedy selection, and how switch-budget constraints change the accuracy-feasibility tradeoff. Feasibility-aware model selection is valuable when it improves executable planning quality relative to accuracy-first approaches, even if it does not always produce the lowest forecast error.
