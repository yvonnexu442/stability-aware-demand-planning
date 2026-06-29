# Beyond Forecast Accuracy

Working title: **Beyond Forecast Accuracy: A Stability-Aware Decision Layer for Operational Demand Planning**

This repository starts a research workflow for testing whether operational demand planning should evaluate forecasts through downstream decision quality, not forecast accuracy alone. The first framework compares simple demand forecasts with a decision layer that trades off expected demand coverage against plan stability.

## Core Problem Statement

Existing forecasting research usually evaluates models by prediction error. However, real operational planning systems do not only need accurate forecasts; they also need stable and executable planning signals. A numerically better forecast may require frequent model switching, large plan changes, or infrastructure updates that the operation cannot absorb. This creates a planning-infrastructure gap: the forecasting layer can adapt faster than the execution system.

This project studies a stability-aware decision layer that converts forecasts into operational plans and evaluates the combined system on both accuracy and execution stability.

## First Research Questions

1. When do forecast improvements fail to improve operational decision quality?
2. Can a lightweight stability-aware layer reduce plan churn with limited service or shortage cost?
3. Which metrics make the accuracy-stability tradeoff visible enough for operational planning?

## Initial Public Data Tables

- UCI Bike Sharing: daily and hourly bike rental counts from the Capital Bikeshare system.
- UCI Online Retail: transactional product quantities for a UK-based online retail business.
- Synthetic demo: generated panel demand data used for smoke tests and framework development.

Large benchmark datasets such as M5/Favorita are useful later, but the first pass keeps the download and preprocessing path simple.

## Repository Layout

```text
configs/                         Dataset registry and experiment defaults
data/raw/                        Downloaded source files, ignored by git
data/processed/                  Standardized demand tables, ignored by git
experiments/                     Command-line experiment runners
paper/                           Paper outline and research notes
results/                         Experiment outputs, ignored by git
src/stability_demand_planning/   Reusable framework code
tests/                           Smoke tests
```

## Quickstart

Run the smoke test:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Run the first synthetic experiment:

```bash
PYTHONPATH=src python3 -m experiments.run_experiment --dataset synthetic_demo
```

Download and run a public UCI dataset:

```bash
PYTHONPATH=src python3 -m experiments.run_experiment --dataset uci_bike_sharing_day
```

The Online Retail dataset is an Excel workbook. Install an Excel reader such as `openpyxl` if your Python environment cannot read `.xlsx` files.

## Standard Table Schema

All datasets are normalized into:

```text
date,item_id,demand
```

- `date`: daily timestamp
- `item_id`: SKU, product, service, or aggregate demand entity
- `demand`: non-negative observed demand quantity

## Metrics

Forecast layer:

- MAE
- RMSE
- WAPE
- bias

Decision layer:

- underage units
- overage units
- service level proxy
- asymmetric decision cost
- total plan variation
- normalized plan variation

## Current Status

This is an initial research scaffold. The title, problem statement, metrics, and decision policies are expected to evolve as experiments expose sharper claims.
