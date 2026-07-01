# operational_planning_stability

Working title: **Beyond Forecast Accuracy: Stability-Aware Operational Demand Planning under Execution Constraints**

This repository is a research-grade Python skeleton for studying stability-aware operational demand planning. It is designed for paper development, mathematical clarity, and reproducible evaluation logic. It is not intended to become a Kaggle-style forecasting leaderboard.

## Current Thesis

Forecast-driven demand planning should be treated as a feasibility-constrained model selection problem rather than a pure forecast accuracy problem. The most accurate forecast is not always the most deployable forecast. This project tests whether execution risk, planning volatility, switching cost, and inventory impact can reorder model deployment choices.

The concise thesis note is maintained in `docs/research_thesis.md`.

## Core Problem Statement

Forecast accuracy alone is not enough for real operational planning. Real supply chain and demand planning systems execute planning signals, not raw forecasts. A numerically better forecast can still create unstable planning signals, frequent model switching, large inventory target jumps, and execution burden that infrastructure, planners, or downstream operational systems cannot absorb.

This creates a planning-infrastructure gap: the forecasting layer can adapt faster than the execution system. The central research question is:

> When forecast signals change faster than execution infrastructure can absorb, how should an operational planning system balance forecast accuracy, planning stability, model switching cost, and execution adaptability?

This repository evaluates planning utility as a multi-objective concept. Forecast error is only one part of the evaluation. The project also tracks inventory cost, planning signal volatility, model switching cost, and execution adaptation penalties.

## First Research Questions

1. When do forecast improvements fail to improve operational decision quality?
2. When does a more accurate forecast create planning instability that the operation cannot absorb?
3. How should a decision layer trade off forecast error, inventory cost, planning stability, and execution capacity?
4. Which outputs should be reported as weighted scalar losses, and which should remain visible as Pareto-style tradeoffs?

## Repository Scope

The repository now supports the core paper workflow across DataCo, Favorita, M5, and Walmart robustness modules. It remains a research repository rather than a Kaggle-style forecasting benchmark. Rossmann is retained as a future extension.

## Repository Layout

```text
configs/default.yaml             Research configuration template
docs/mathematical_model.md       Formal notation and objective functions
docs/system_flow.md              End-to-end system flow diagram
paper/main.tex                   LaTeX manuscript skeleton
paper/sections/                  LaTeX manuscript sections
paper/tables/                    LaTeX-ready table assets
paper/figures/                   LaTeX-ready figure assets
data/raw/                        Raw public datasets, ignored by git
data/processed/                  Processed tables, ignored by git
src/data_loaders/                Dataset loader interfaces
src/features/                    Forecast and planning signal features
src/models/                      Forecast model interfaces and baselines
src/planning_environment/        Execution capacity and planning simulator logic
src/decision_layer/              Hard selection, ensemble, and stability-aware selection
src/evaluation/                  Forecast, inventory, stability, and planning utility metrics
src/reporting/                   LaTeX export utilities for paper assets
src/visualization/               Plotting helpers
src/utils/                       Config, logging, and time split utilities
scripts/                         Experiment entry points
outputs/                         Tables, figures, logs, and config snapshots, ignored by git
tests/                           Smoke tests for core modules
```

## Quickstart

Run the smoke test:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Evaluation Philosophy

The repository separates forecast quality from operational planning utility:

- Forecast metrics measure prediction error.
- Inventory metrics measure holding, shortage, and service outcomes.
- Stability metrics measure planning signal volatility and model switching.
- Planning utility combines forecast, inventory, stability, switching, and execution adaptation terms.

The paper should report both a weighted scalar planning loss and Pareto-style multi-objective outputs. A single score is useful for optimization, but the tradeoff surface is essential for explaining the planning-infrastructure gap.

## LaTeX Paper Workflow

The final paper is written in LaTeX under `paper/`. Python experiments should write intermediate result tables to `outputs/tables/` and figures to `outputs/figures/`. The export script converts those outputs into manuscript-ready assets:

```bash
PYTHONPATH=src python3 scripts/export_latex_assets.py
```

Major result tables should be saved as both `.csv` and `.tex`. Research figures should be saved as PDF for LaTeX compatibility, with optional PNG copies for quick viewing. The LaTeX manuscript can then import generated assets from `paper/tables/` and `paper/figures/`.

Paper-ready figure and table conventions are documented in `docs/figure_style_guide.md`.

## Research Progress Log

### Favorita Feasibility Stress Tests and Pareto Analysis

The Favorita feasibility analysis added weight-sensitivity analysis,
execution-capacity stress tests, Pareto summaries, and a more interpretable
feasibility-aware selector for the quick-mode proof of concept. The current
results support a tradeoff interpretation: accuracy-only LightGBM/global-best
remains strong under the baseline scalar loss, while feasibility-aware and
stability-aware strategies reduce planning volatility and execution burden under
more constrained or more execution-sensitive settings. The main takeaway is
documented in `docs/favorita_feasibility_tradeoff_takeaways.md`.

### Favorita Sample-Size Reproducibility

The Favorita reproducibility layer runs the same pipeline across larger sample
sizes and exports sample-size comparison tables and figures. In the current
local environment, quick mode and medium mode completed successfully, while
larger full-mode attempts loaded data but failed during native ML fitting. The
completed runs show that the accuracy-first winner can change with sample size,
but the execution-feasibility tradeoff remains visible. The detailed findings
are documented in `docs/favorita_reproducibility_findings.md`.

### Thesis Wording Update

The repository now includes a concise thesis statement that frames
forecast-driven demand planning as feasibility-constrained model selection. The
wording is stored in `docs/research_thesis.md` and reflected in the manuscript
abstract and introduction placeholders.

### Normalized Loss and DataCo-Informed Execution Risk

The pipeline now exports a baseline strategy comparison, normalized planning-loss
components, normalization reference audits, and DataCo-informed execution-risk
scenario tables. DataCo late-delivery behavior is used as an empirical anchor
for execution-risk sensitivity scenarios when local DataCo files are available;
otherwise, the pipeline records a clear fallback to configured default scenario
values. The scenario design is documented in `docs/dataco_execution_scenarios.md`.

### Improved Feasibility-Aware Methods

The method-improvement layer adds feasibility-aware smoothed planning and
feasibility-aware ensembles to the DataCo-informed Favorita re-evaluation. The
smoothed strategies use fixed, scenario-based, and validation-CV-selected
adaptation rates. The ensemble strategies use inverse validation accuracy,
inverse validation operational loss, and a transparent constrained validation
grid with nonnegative weights that sum to one. The current quick-mode results
show that smoothing can sharply reduce execution burden while increasing
inventory cost, and that operational-loss-weighted ensembles can improve the
baseline-scenario deployability tradeoff without changing the DataCo scenario
mapping.

### Favorita Method-Family Summary and M5 Robustness

The paper-ready consolidation layer now exports a compact Favorita method-family
summary that compares accuracy-first, reference ensemble, feasibility-aware
selector, feasibility-aware smoothing, feasibility-aware ensemble,
stability-first, and oracle reference strategies. It also exports frontier plots
for accuracy versus execution burden, inventory cost versus execution burden,
normalized loss across DataCo-informed scenarios, and switching versus execution
penalty. The M5 robustness layer adds a real loader and transparent baseline
pipeline for large-scale hierarchical retail demand checks, including hierarchy
sensitivity, intermittent-demand stress, and DataCo scenario robustness. M5
quick mode completed locally and writes LaTeX-ready tables and PDF figures for
the manuscript workflow.

### LaTeX Paper Drafting Mode

The manuscript has moved from placeholder structure to an integrated LaTeX draft
using the current DataCo, Favorita, and M5 outputs. The draft frames operational
demand planning as feasibility-constrained model selection, keeps Related Work
citation-light with TODO placeholders, uses generated tables and figures where
available, and preserves Walmart and Rossmann as future robustness extensions.
A standalone `paper/full_draft_snapshot.tex` file provides a single-file review
snapshot of the current paper narrative.

### Finite-Horizon Decision Layer

The decision layer now includes explicit deployable finite-horizon selectors:
GreedyFeasibilitySelector, DPFeasibilitySelector, and
BudgetedDPFeasibilitySelector. Greedy selection minimizes one-step expected
operational cost, DP selection minimizes cumulative expected operational loss
over the available horizon, and Budgeted DP adds a hard switch budget. These
selectors use validation-derived expected costs and forecast-implied planning
signals; realized test demand remains reserved for Oracle-labeled diagnostics.
The Realized-Inventory Oracle DP is non-deployable and uses period-specific
realized test inventory outcomes as an information-access upper bound, not as a
perfect-forecast oracle. Budgeted DP keeps an incumbent-stays fallback for
pipeline safety when no budget-feasible path remains; fallback rows are
diagnostic and are explicitly marked in output metadata.

### Decision-Layer Audit and Oracle Semantics

The decision-layer audit now distinguishes deployable strategies from
non-deployable Oracle diagnostics. Oracle DP is documented as a
Realized-Inventory Oracle DP: it uses period-specific realized test inventory
outcomes keyed by series, candidate model, and date, while direct forecast loss
remains diagnostic unless explicitly weighted in a sensitivity setting. It is
not a perfect-forecast oracle. The Budgeted DP fallback
is also explicit in outputs through fallback metadata. The fallback freezes the
first-period lowest-cost incumbent model when that model remains available, and
it marks later missing-incumbent cases as potentially non-strict diagnostics.
The audit table is exported to `outputs/tables/decision_layer_audit.csv` and
`paper/tables/decision_layer_audit_table.tex`.

### Operational-Loss Objective and Oracle Gap Split

The default planning objective now treats forecast accuracy as a diagnostic
metric rather than as a direct term in normalized operational loss.
`planning_loss_weights.alpha_forecast` defaults to `0.0`, while inventory,
volatility, execution, and switching terms define the main planning objective.
Paper-facing outputs now split Oracle gaps into `gap_to_dp_oracle` and
`gap_to_perfect_oracle` so model-selection suboptimality and realized-demand
uncertainty are reported separately.

### Walmart Business-Context Robustness

The Walmart module now implements a weekly store-department robustness check for
business-context planning signals. It loads `train.csv`, `features.csv`, and
`stores.csv` from `data/raw/walmart/`, builds history-only and
history-plus-context forecast candidate sets, and reuses the existing
Global Best, Simple Ensemble, Operational Ensemble, Greedy Feasibility, DP
Feasibility, Budgeted DP, and Oracle diagnostic strategies. Quick mode completed
locally and exports context comparison, holiday/markdown stress-window, weekly
cadence constraint, and overall robustness tables and figures for the LaTeX
paper workflow. Walmart is treated as a narrow robustness check, not as a new
forecasting leaderboard.

## Raw Data Workflow

Raw benchmark datasets are stored under `data/raw/` with one directory per source:

```text
data/raw/
  favorita/
  m5/
  walmart/
  rossmann/
  dataco/
```

These data files are not committed to Git. After installing and authenticating the Kaggle CLI, raw datasets can be downloaded with:

```bash
PYTHONPATH=src python3 scripts/download_raw_data.py --unzip
```

To create the expected folder structure without downloading data, run:

```bash
PYTHONPATH=src python3 scripts/download_raw_data.py --create-directories-only
```

## How to Run the Favorita Minimal Pipeline

The first real-data proof-of-concept uses the Favorita Store Sales dataset. The
pipeline does not download Kaggle data automatically. Manually download and
unzip the dataset, then place these CSV files under `data/raw/favorita/`:

```text
data/raw/favorita/
  train.csv
  stores.csv
  oil.csv
  holidays_events.csv
  transactions.csv
  test.csv                  Optional
```

Quick mode uses the top eligible store-family series configured by
`data.quick_mode_max_series` in `configs/default.yaml`. Full mode uses
`data.full_mode_max_series`; set that value to `null` to run every eligible
series.

Run the minimal pipeline with:

```bash
PYTHONPATH=src python3 scripts/run_favorita_minimal_pipeline.py
```

Run full mode or override the series count with:

```bash
PYTHONPATH=src python3 scripts/run_favorita_minimal_pipeline.py --run-mode full
PYTHONPATH=src python3 scripts/run_favorita_minimal_pipeline.py --max-series 250
```

If the local native ML stack fails while rebuilding LightGBM or XGBoost
forecasts, reuse an existing standardized forecast table and rerun the decision
layer and paper exports:

```bash
PYTHONPATH=src python3 scripts/run_favorita_minimal_pipeline.py \
  --run-mode quick \
  --reuse-forecast-table outputs/tables/favorita_forecasts.csv
```

The pipeline writes analysis tables to `outputs/tables/`, figures to
`outputs/figures/`, and the run log to
`outputs/logs/favorita_minimal_pipeline.log`. LaTeX-ready tables are written to
`paper/tables/`, and PDF figures for the manuscript are written to
`paper/figures/`.

If LightGBM or XGBoost is installed, the pipeline keeps them as separate global
machine-learning forecast candidates. If neither package is available, it uses a
scikit-learn fallback model and records that choice in the run log.

The same command also runs the Favorita feasibility analysis layer. It writes
weight-sensitivity, execution-capacity stress-test, and Pareto summary tables to
`outputs/tables/`, with LaTeX-ready versions under `paper/tables/`. It also
writes PNG figures to `outputs/figures/` and manuscript PDF figures to
`paper/figures/`. These analyses are intended to show the tradeoff surface, not
to tune weights until one stability-aware strategy always wins.

## How to Run the Walmart Robustness Pipeline

The Walmart business-context robustness module expects manually downloaded
Walmart Recruiting Store Sales Forecasting files under `data/raw/walmart/`:

```text
data/raw/walmart/
  train.csv
  features.csv
  stores.csv
```

Run quick mode with:

```bash
PYTHONPATH=src:scripts python3 scripts/run_walmart_robustness_pipeline.py --run-mode quick
```

Quick mode samples eligible store-department series, compares history-only and
history-plus-context feature sets, evaluates holiday and markdown stress
windows, and applies weekly cadence constraints. Medium mode is supported, while
full mode is disabled by default in `configs/default.yaml`.

## Current Status

The current implementation contains the core planning-stability modules, the
LaTeX paper workflow, DataCo execution-risk calibration, Favorita proof-of-
concept and method-improvement pipelines, M5 hierarchy/intermittency
robustness, and Walmart business-context robustness. Rossmann remains a future
extension.
