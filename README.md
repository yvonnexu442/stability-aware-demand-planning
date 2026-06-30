# operational_planning_stability

Working title: **Beyond Forecast Accuracy: Stability-Aware Operational Demand Planning under Execution Constraints**

This repository is a research-grade Python skeleton for studying stability-aware operational demand planning. It is designed for paper development, mathematical clarity, and reproducible evaluation logic. It is not intended to become a Kaggle-style forecasting leaderboard.

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

This first step implements the repository skeleton and core research modules only. Dataset-specific loaders and full experiments are intentionally left as later work.

Candidate public datasets for future experiments include Favorita, M5/Walmart-style retail demand, Walmart recruiting demand data, and Rossmann sales. These loaders are represented as placeholders so the paper logic can develop before the project becomes a dataset ingestion exercise.

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

## Current Status

The current implementation contains the core planning-stability modules, the
LaTeX paper workflow, DataCo profiling utilities, and the minimal Favorita
real-data proof-of-concept pipeline. It does not yet implement the full
multi-dataset experiment suite.
