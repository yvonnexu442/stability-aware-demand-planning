# IEOM Role and Metric Audit

This audit documents the manuscript role cleanup and the data-driven checks used in the revision.

## Dataset Roles

- DataCo is an execution-risk anchor only. It supplies late-delivery scenario anchors and is not treated as a demand-planning or inventory-control dataset.
- Favorita is the main forecast-to-plan proof-of-concept dataset.
- M5 is the hierarchy and intermittency robustness dataset.
- Walmart is secondary business-context and weekly-cadence robustness evidence.

The paper-facing phrasing is: one execution-risk anchor and three demand-planning datasets.

## Favorita Baseline Audit

Source: `paper/tables/favorita_method_family_summary_table.csv`.

| Scenario | Method | WAPE | Execution violation rate | Execution penalty | Normalized loss |
|---|---:|---:|---:|---:|---:|
| baseline | Global Best | 0.109 | 0.284 | 546133.345 | 1.550 |
| baseline | Operational Ensemble | 0.119 | 0.133 | 118441.378 | 1.422 |
| baseline | Simple Ensemble | 0.129 | 0.066 | 47421.135 | 1.434 |
| baseline | Smoothed Alpha 0.25 | 0.109 | 0.001 | 102.215 | 1.646 |
| baseline | Best Stability | 0.188 | 0.000 | 0.000 | 1.766 |
| dataco_severe | Global Best | 0.109 | 0.284 | 546133.345 | 2.000 |
| dataco_severe | Operational Ensemble | 0.121 | 0.116 | 95796.860 | 1.496 |
| dataco_severe | Simple Ensemble | 0.129 | 0.066 | 47421.135 | 1.473 |

Revision decision: the manuscript now describes Global Best as the accuracy-first reference and states that ensemble methods accept higher WAPE for lower execution burden. It no longer claims that Simple Ensemble has lower WAPE than Global Best.

## M5 Hierarchy Audit

Source: `paper/tables/m5_robustness_summary_table.csv`.

| Grain | Best deployable method | WAPE | Execution violation rate | Normalized loss |
|---|---:|---:|---:|---:|
| item-store | Smoothed Alpha 0.25 | 0.368 | 0.010 | 1.223 |
| department-store | Operational Ensemble | 0.162 | 0.100 | 1.177 |
| category-store | Operational Ensemble | 0.159 | 0.063 | 1.261 |

Revision decision: the M5 paragraph now uses the grain-level audited values rather than mixing them with intermittency-bucket values.

## Walmart Role Audit

Sources: `paper/tables/thesis_quantification_summary_table.csv`, `paper/tables/walmart_robustness_summary_table.csv`, and `paper/tables/walmart_context_robustness_summary_table.csv`.

- Walmart appears only as `dataset_name = walmart`.
- Walmart modules include `context_robustness`, `holiday_markdown_stress`, and `weekly_cadence_constraints`.
- The main Walmart robustness summary contains 20 evaluation groups and an 80.0% accuracy-first mismatch rate.

Revision decision: Walmart is retained as secondary business-context and weekly-cadence robustness, not as the main empirical proof of concept.

## Information Access and Leakage Prevention

The tightened Methods section separates three information layers:

- Calibration uses validation-period realized demand to estimate expected operational costs.
- Deployable test-period selection does not use test-period realized demand. It uses candidate forecasts, previous executed plans, execution-capacity scenarios, validation-derived expected costs, switching penalties, and policy rules.
- Ex-post evaluation uses realized test-period demand only after the horizon is complete.

Only the Realized-Inventory Oracle DP and the perfect-demand diagnostic reference use test-period realized outcomes during selection, and both are labeled as non-deployable diagnostic references.
