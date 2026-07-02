# IEOM Table and Figure Audit

This audit covers tables and figures included in `paper_submission_ieom/main.tex` through section inputs after the role-and-metric tightening pass.

| Number | Type | Title or caption | Source file | Source exists | Mentioned in text | Interpreted in text | Formatting check | Main purpose |
|---|---|---|---|---|---|---|---|---|
| Figure 1 | Figure | Feasibility-constrained decision layer for forecast-driven demand planning. | `paper_submission_ieom/figures/framework_diagram.tex` | Yes | Yes | Yes | Caption below figure | Defines the shared operational decision layer. |
| Table 1 | Table | Empirical validation design for feasibility-constrained model selection. | `paper_submission_ieom/tables/dataset_roles_table.tex` | Yes | Yes | Yes | Title above table | Explains the validation architecture: one execution-risk anchor and three demand-planning datasets. |
| Figure 2 | Figure | Favorita accuracy-execution frontier, baseline scenario. | `paper_submission_ieom/figures/favorita_accuracy_execution_frontier.pdf` | Yes | Yes | Yes | Caption below figure | Shows that WAPE and execution penalty do not move together. |
| Figure 3 | Figure | Strategy-selection procedure under operational constraints. | `paper_submission_ieom/figures/deployment_protocol_diagram.tex` | Yes | Yes | Yes | Caption below figure | Converts evidence into operational strategy selection. |
| Table 2 | Table | Strategy families and operating conditions. | `paper_submission_ieom/tables/deployment_guidance_table.tex` | Yes | Yes | Yes | Title above table | Maps operating conditions to deployable policy choices. |
| Table 3 | Table | Implementation agenda for a feasibility-aware planning system. | `paper_submission_ieom/tables/implementation_agenda_table.tex` | Yes | Yes | Yes | Title above table | Lists organizational capabilities needed to implement the strategy-selection procedure. |
| Figure 4 | Figure | Favorita normalized planning loss across execution-risk scenarios. | `paper_submission_ieom/figures/favorita_normalized_loss_by_execution_scenario.pdf` | Yes | Yes | Yes | Caption below figure | Shows that execution-risk severity changes strategy rankings. |

## Audit Decision

- The main manuscript now uses three tables and four figures.
- Table 1 is no longer a source inventory; it is a validation-design table.
- The deleted core numerical table, M5 table, and Walmart table are not included in the main manuscript; their key verified values are summarized in text and audited separately.
- No table or figure is included solely as an artifact dump.
