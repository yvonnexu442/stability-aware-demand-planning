# Favorita Sample-Size Reproducibility Findings

These findings summarize the current Favorita quick and medium reproducibility runs. The full-mode run was attempted but is not reported because the local native ML stack failed during model fitting after successful data loading.

## Completed Runs

- Quick mode: 100 store-family series and 168,400 rows.
- Medium mode: 500 store-family series and 842,000 rows.
- Full-mode attempts: 1000, 750, 600, and 550 series all loaded data successfully, but each run failed during native LightGBM/XGBoost fitting with `SIGSEGV`. The repository therefore reports quick and medium robustness results for this environment and records the full-mode limitation explicitly.

## Data-Driven Findings

The broad method ranking is reasonably stable from quick to medium mode. The Spearman rank correlation of total planning loss versus quick mode is 0.899. However, the identity of the top scalar-loss method is not identical: quick mode favors Global Best/LightGBM, while medium mode favors Family Best by a narrow margin.

The strongest accuracy-first conclusion becomes more nuanced at medium scale. In quick mode, Global Best/LightGBM has the lowest WAPE at 0.1086. In medium mode, Family Best has the lowest WAPE at 0.1307, while Global Best/XGBoost is very close at 0.1311 and LightGBM is 0.1329. This means the quick-mode LightGBM finding should be treated as a strong initial signal, not a universal result.

The feasibility tradeoff is stable across both completed sample sizes. Feasibility-Aware has higher WAPE and inventory cost than Global Best in both modes, but it lowers execution burden in both modes.

- Quick mode: Feasibility-Aware reduces execution penalty from 546,133 to 365,117 and reduces violation rate from 0.284 to 0.233.
- Medium mode: Feasibility-Aware reduces execution penalty from 914,083 to 538,853 and reduces violation rate from 0.278 to 0.216.

The moving-average baseline remains useful as an over-stable reference. It has near-zero execution burden but poor WAPE and high inventory cost in both completed modes, so it should not be interpreted as an operationally strong method.

## Research Interpretation

The sample-size robustness run supports the paper thesis with an important caveat. The tradeoff between forecast accuracy, inventory cost, planning volatility, and execution burden remains visible as the sample size increases from 100 to 500 series. Feasibility-aware planning consistently reduces execution burden, but it does not dominate scalar total loss under the baseline weights.

The medium-mode result also shows why the paper should avoid overclaiming a single best forecasting model. The best accuracy-first model can change with sample size, while the need to report execution feasibility remains stable.
