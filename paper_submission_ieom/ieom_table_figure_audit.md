# IEOM Table and Figure Audit

This audit covers tables and figures included in `paper_submission_ieom/main.tex` through section inputs after the surgical tightening pass.

| Number | Type | Title or caption | Source file | Source exists | Mentioned in text | Interpreted in text | Formatting check | Main purpose |
|---|---|---|---|---|---|---|---|---|
| Figure 1 | Figure | Forecast-to-plan feasibility gate for demand-planning deployment. | `paper_submission_ieom/figures/ieom_framework_flowchart.pdf` | Yes | Yes | Yes | Caption below figure | Defines the paper's deployment gate. |
| Table 1 | Table | Dataset roles and experimental purpose. | `paper_submission_ieom/tables/dataset_roles_table.tex` | Yes | Yes | Yes | Title above table | Defines DataCo, Favorita, M5, Walmart, and Rossmann scope. |
| Table 2 | Table | Core numerical evidence for the planning-execution gap. | `paper_submission_ieom/tables/core_numerical_summary_table.tex` | Yes | Yes | Yes | Title above table | Supports Finding 1 and thesis quantification. |
| Figure 2 | Figure | Favorita accuracy and execution penalty frontier. | `paper_submission_ieom/figures/favorita_accuracy_execution_frontier.png` | Yes | Yes | Yes | Caption below figure | Shows that WAPE and execution penalty do not move together. |
| Figure 3 | Figure | Favorita normalized planning loss by DataCo-informed execution scenario. | `paper_submission_ieom/figures/favorita_normalized_loss_by_execution_scenario.png` | Yes | Yes | Yes | Caption below figure | Shows that execution-risk scenario changes ranking. |
| Table 3 | Table | Deployment strategy families and rollout conditions. | `paper_submission_ieom/tables/deployment_guidance_table.tex` | Yes | Yes | Yes | Title above table | Translates results into a model-rollout protocol. |

## Audit Decision

- The main manuscript now uses three tables and three figures, which is below the requested maximum of four tables and three figures.
- The dense method-component evidence table was removed from the manuscript.
- The separate M5 and Walmart robustness tables are not included in the main manuscript; their key verified findings are summarized in text.
- No table or figure is included solely as an artifact dump.
