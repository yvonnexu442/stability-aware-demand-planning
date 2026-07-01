# operational_planning_stability

Working title: **Beyond Forecast Accuracy: Stability-Aware Operational Demand Planning under Execution Constraints**

This repository is a research-grade Python skeleton for studying stability-aware operational demand planning. It is designed for paper development, mathematical clarity, and reproducible evaluation logic. It is not intended to become a Kaggle-style forecasting leaderboard.

## Compiled Paper Draft

The current compiled manuscript draft is available at:

[`paper/stability_aware_demand_planning_draft.pdf`](paper/stability_aware_demand_planning_draft.pdf)

The LaTeX source remains under `paper/main.tex` and `paper/sections/`. The local build artifact `paper/main.pdf` is ignored by git, so the tracked draft PDF above is the GitHub-facing copy.

## Current Thesis

Forecast-driven demand planning should be treated as a feasibility-constrained model selection problem rather than a pure forecast-accuracy ranking problem. The forecast with the best WAPE is not necessarily the best deployable planning strategy once inventory exposure, planning-signal volatility, model-switching burden, and execution-capacity violations are considered.

The paper's purpose is to quantify this gap. It asks how often the accuracy-first strategy differs from the operational-loss-optimal deployable strategy, how frequently planning signals exceed execution capacity, how much finite-horizon DP improves over one-step greedy selection, and how DataCo execution evidence can anchor relative execution-risk scenarios without claiming exact cost calibration.

The concise thesis note is maintained in `docs/research_thesis.md`.

## Core Problem Statement

Most forecasting workflows answer the question: which model predicts demand most accurately? Operational planning asks a different question: which forecast-driven strategy should be deployed when the operation has finite execution capacity, inventory exposure, governance constraints, and limited tolerance for abrupt plan changes?

Real supply-chain and demand-planning systems execute planning signals, not raw forecasts. A forecast is converted into an inventory target, replenishment recommendation, staffing plan, production target, or capacity allocation. A numerically stronger forecast can still create unstable or infeasible plans if it requires large period-to-period target jumps, frequent model switching, infrastructure updates, or planner interventions that the execution system cannot absorb.

This creates a planning-execution gap: the forecasting layer can adapt faster than downstream operations, enterprise systems, and governance processes. This repository operationalizes the gap as execution violations and planning-execution gap rate, then evaluates whether decision-layer strategies can reduce that gap without hiding inventory or service tradeoffs.

The core research problem is therefore:

> Given a set of candidate forecasts, choose a deployable planning strategy that balances inventory exposure, planning stability, model-switching burden, and execution feasibility, while reporting forecast accuracy separately as a diagnostic metric.

The repository studies this problem with interpretable decision-layer strategies: accuracy-first baselines, ensembles, smoothing, feasibility-aware selectors, one-step Greedy selection, finite-horizon DP, Budgeted DP, and explicitly non-deployable Oracle diagnostics.

## First Research Questions

1. How often does the best-WAPE deployable strategy differ from the lowest-operational-loss deployable strategy?
2. What planning-execution gap rate does each strategy create, and how much can feasibility-aware selection reduce it?
3. What does finite-horizon DP buy over one-step greedy selection under switching and execution costs?
4. How does a hard switch budget `K` change the accuracy-feasibility tradeoff for Budgeted DP?
5. How should DataCo execution-risk evidence be used as a scenario anchor without treating it as exact cost calibration for Favorita, M5, or Walmart?
6. Which tradeoffs should be summarized by normalized planning loss, and which should remain visible through raw metrics and Pareto-style figures?

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

- Forecast metrics measure prediction error and are reported as diagnostics.
- Inventory metrics measure holding, shortage, and service outcomes.
- Stability metrics measure planning signal volatility and model switching.
- Execution metrics measure whether plan changes exceed modeled execution capacity.
- The default operational planning objective excludes direct forecast-error terms and combines inventory exposure, planning volatility, execution burden, and model-switching burden.

The paper reports both normalized scalar planning loss and Pareto-style multi-objective outputs. A single score is useful for optimization, but the tradeoff surface is essential for explaining the planning-infrastructure gap.

## LaTeX Paper Workflow

The final paper is written in LaTeX under `paper/`. Python experiments should write intermediate result tables to `outputs/tables/` and figures to `outputs/figures/`. The export script converts those outputs into manuscript-ready assets:

```bash
PYTHONPATH=src python3 scripts/export_latex_assets.py
```

Major result tables should be saved as both `.csv` and `.tex`. Research figures should be saved as PDF for LaTeX compatibility, with optional PNG copies for quick viewing. The LaTeX manuscript can then import generated assets from `paper/tables/` and `paper/figures/`.

Paper-ready figure and table conventions are documented in `docs/figure_style_guide.md`.

## Current Research Layers

The repository now has four empirical roles:

- `DataCo`: execution-risk profiling. Late-delivery behavior is used as an empirical anchor for relative execution-risk scenarios, not as exact cost calibration for retail demand datasets.
- `Favorita`: main daily store-family proof of concept. It supports forecast metrics, inventory metrics, execution-capacity stress tests, DataCo-informed scenario analysis, smoothing/ensemble method comparisons, and sample-size robustness outputs.
- `M5`: hierarchical retail robustness. It supports item-store, department-store, and category-store views, intermittency stress, DataCo-informed scenario robustness, and switch-budget sensitivity.
- `Walmart`: weekly business-context robustness. It compares history-only and history-plus-context candidate forecasts, holiday/markdown stress windows, weekly cadence constraints, and switch-budget sensitivity.

The decision layer includes accuracy-first baselines, simple and operational-loss-weighted ensembles, stability-first references, feasibility-aware selectors, smoothed planning strategies, Greedy finite-horizon selection, unrestricted DP selection, Budgeted DP selection with hard switch budget `K`, and explicitly non-deployable Oracle diagnostics.

## Paper-Facing Outputs

The manuscript imports LaTeX-ready tables from `paper/tables/` and PDF figures from `paper/figures/`. Current paper-facing outputs include:

- Thesis quantification summaries: accuracy-ranking mismatch, planning-execution gap rate, Greedy-vs-DP value, and split Oracle gaps.
- DataCo execution-risk calibration and generated scenario tables.
- Favorita forecast, method-family, smoothing, ensemble, Pareto, weight-sensitivity, execution-capacity, and sample-size robustness outputs.
- M5 hierarchy, intermittency, DataCo-scenario, and switch-budget sensitivity outputs.
- Walmart context robustness, holiday/markdown stress, weekly cadence, and switch-budget sensitivity outputs.

The `paper/full_draft_snapshot.tex` file is a synchronized review build that imports the current manuscript sections. The main manuscript is `paper/main.tex`.

## Methodology Notes

- Forecast accuracy is reported separately as WAPE and is not part of the default normalized operational objective.
- `planning_loss_weights.alpha_forecast` defaults to `0.0`; inventory, volatility, execution, and switching terms define the main objective.
- Paper-facing Oracle gaps are split into `gap_to_dp_oracle` and `gap_to_perfect_oracle`.
- Realized-Inventory Oracle DP and Full-Outcome Oracle DP are non-deployable diagnostics. They are not perfect-forecast deployment strategies.
- DataCo transfers empirical heterogeneity in execution risk into sensitivity scenarios. It does not estimate exact dollar costs for Favorita, M5, or Walmart execution violations.
- Budgeted DP uses an incumbent-stays fallback when no strict budget-feasible path remains; fallback rows are diagnostic and explicitly marked.

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

## Unified Experiment Entry Points

Run one current module through the stable wrapper:

```bash
PYTHONPATH=src:scripts python3 scripts/run_single_experiment.py walmart --run-mode quick
PYTHONPATH=src:scripts python3 scripts/run_single_experiment.py switch_budget --dataset walmart --run-mode quick
PYTHONPATH=src:scripts python3 scripts/run_single_experiment.py thesis_quantification
```

Preview the full command sequence without running it:

```bash
PYTHONPATH=src:scripts python3 scripts/run_all_experiments.py --dry-run
```

Run quick-mode modules in sequence:

```bash
PYTHONPATH=src:scripts python3 scripts/run_all_experiments.py \
  --run-mode quick \
  --include-switch-budget \
  --include-thesis-quantification \
  --continue-on-error
```

Switch-budget sensitivity can also be run directly:

```bash
PYTHONPATH=src:scripts python3 scripts/run_switch_budget_sensitivity.py \
  --dataset walmart \
  --run-mode quick \
  --k-values 0 1 2 4 8

PYTHONPATH=src:scripts python3 scripts/run_switch_budget_sensitivity.py \
  --dataset m5 \
  --run-mode quick \
  --k-values 0 1 2 4 8
```

Thesis-level summary tables can be refreshed from the current result CSV files:

```bash
PYTHONPATH=src python3 scripts/run_thesis_quantification.py
```

## Current Status

The current implementation contains the core planning-stability modules, the
LaTeX paper workflow, DataCo execution-risk calibration, Favorita proof-of-
concept and method-improvement pipelines, M5 hierarchy/intermittency
robustness, Walmart business-context robustness, and a thesis-quantification
layer that summarizes accuracy-ranking mismatch, planning-execution gap rate,
Greedy-vs-DP value, switch-budget sensitivity, and split Oracle gaps. Rossmann
remains a future extension.
