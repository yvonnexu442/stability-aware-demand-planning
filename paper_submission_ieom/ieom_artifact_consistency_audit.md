# IEOM Artifact Consistency Audit

This audit was generated from repository artifacts before revising the IEOM manuscript. It records whether claims about DataCo, Favorita, M5, Walmart, and Rossmann are supported by generated artifacts.

## Summary by Dataset or Section
dataset_or_section,exists,count
DataCo,True,7
Favorita,True,8
IEOM submission,True,4
M5,True,7
Rossmann,False,2
Rossmann,True,2
Walmart,True,13


## Key Findings
- DataCo, Favorita, M5, and Walmart have generated outputs suitable for IEOM paper use.
- Walmart has completed context, holiday/markdown, weekly-cadence, and switch-budget artifacts.
- Rossmann raw data and loader exist, but no completed Rossmann robustness pipeline or result outputs were found; Rossmann should remain future work.

## Detailed Audit
artifact_path,artifact_type,dataset_or_section,exists,purpose,paper_claim_supported,last_modified_if_available,notes
data/raw/dataco/DataCoSupplyChainDataset.csv,raw_data,DataCo,True,Raw DataCo supply-chain fulfillment records.,Execution-risk anchor from late-delivery behavior.,2026-06-29T19:30:27,
src/data_loaders/dataco_loader.py,data_loader,DataCo,True,Data profiling and loading support.,Supports use of DataCo as execution-risk anchor.,2026-06-29T23:41:56,
scripts/profile_dataco.py,experiment_script,DataCo,True,Profiles DataCo execution risk.,Supports DataCo scenario design.,2026-06-29T19:40:44,
outputs/tables/dataco_execution_risk_calibration.csv,output_table_csv,DataCo,True,Generated execution-risk calibration table.,Supports DataCo-informed execution weights.,2026-06-30T07:53:37,
outputs/tables/dataco_execution_risk_percentiles.csv,output_table_csv,DataCo,True,Generated DataCo risk percentiles.,Supports low-to-severe execution-risk anchors.,2026-06-30T07:53:37,
paper/tables/dataco_execution_risk_calibration_table.tex,paper_table_tex,DataCo,True,LaTeX-ready DataCo calibration table.,Supports paper DataCo scenario claims.,2026-06-30T07:53:37,
paper/figures/dataco_execution_lambda_scenarios.pdf,paper_figure_pdf,DataCo,True,LaTeX-ready DataCo scenario figure.,Supports execution-risk scenario visualization.,2026-06-30T07:53:38,
data/raw/favorita/train.csv,raw_data,Favorita,True,Raw Favorita demand panel.,Main forecast-to-plan proof of concept.,2026-06-29T19:18:30,
src/data_loaders/favorita_loader.py,data_loader,Favorita,True,Favorita loading and feature support.,Supports Favorita analysis.,2026-06-29T19:55:42,
scripts/run_favorita_minimal_pipeline.py,experiment_script,Favorita,True,Runs Favorita proof-of-concept pipeline.,Supports main proof of concept.,2026-06-30T20:28:19,
scripts/consolidate_favorita_method_family_results.py,experiment_script,Favorita,True,Consolidates Favorita method-family outputs.,Supports paper-facing Favorita results.,2026-06-30T20:26:31,
outputs/tables/favorita_method_family_summary.csv,output_table_csv,Favorita,True,Generated method-family summary.,"Supports ensemble, scenario, and ranking-shift claims.",2026-06-30T16:40:36,
paper/tables/favorita_method_family_summary_table.csv,paper_table_csv,Favorita,True,Paper-facing Favorita method-family summary.,Supports numeric Favorita claims.,2026-06-30T16:40:36,
paper/figures/favorita_method_family_accuracy_vs_execution_penalty.pdf,paper_figure_pdf,Favorita,True,LaTeX-ready Favorita tradeoff plot.,Supports accuracy-execution frontier claim.,2026-06-30T08:41:34,
paper_submission_ieom/figures/favorita_accuracy_execution_frontier.png,paper_submission_figure,Favorita,True,IEOM-facing Favorita frontier plot.,Supports graphical result.,2026-07-01T08:06:40,
data/raw/m5/sales_train_evaluation.csv,raw_data,M5,True,Raw M5 sales data.,Hierarchy and intermittency robustness.,2026-06-29T19:22:55,
src/data_loaders/m5_loader.py,data_loader,M5,True,M5 loading and feature support.,Supports M5 robustness analysis.,2026-06-30T08:36:50,
scripts/run_m5_robustness_pipeline.py,experiment_script,M5,True,Runs M5 robustness pipeline.,Supports hierarchy and intermittency claims.,2026-06-30T20:25:25,
outputs/tables/m5_robustness_summary.csv,output_table_csv,M5,True,Generated M5 robustness summary.,Supports M5 robustness claims.,2026-06-30T16:40:36,
paper/tables/m5_robustness_summary_table.csv,paper_table_csv,M5,True,Paper-facing M5 robustness summary.,Supports M5 numeric claims.,2026-06-30T16:40:36,
paper_submission_ieom/tables/m5_robustness_table.tex,paper_submission_table,M5,True,Compact IEOM M5 table.,Supports M5 hierarchy/intermittency finding.,2026-07-01T08:11:22,
paper/figures/m5_hierarchy_sensitivity.pdf,paper_figure_pdf,M5,True,LaTeX-ready M5 hierarchy figure.,Supports M5 hierarchy robustness.,2026-06-30T08:44:06,
data/raw/walmart/train.csv,raw_data,Walmart,True,Raw Walmart weekly sales data.,Business-context and weekly-cadence robustness.,2026-06-29T19:25:24,
src/data_loaders/walmart_loader.py,data_loader,Walmart,True,Walmart loading and feature support.,Supports Walmart robustness analysis.,2026-06-30T19:59:10,
scripts/run_walmart_robustness_pipeline.py,experiment_script,Walmart,True,Runs Walmart robustness pipeline.,Supports Walmart completion.,2026-06-30T20:25:29,
outputs/tables/walmart_context_robustness_summary.csv,output_table_csv,Walmart,True,Generated context robustness summary.,Supports W1 context-aware comparison.,2026-06-30T20:06:26,
outputs/tables/walmart_context_aware_vs_context_free_comparison.csv,output_table_csv,Walmart,True,Generated context-aware vs context-free table.,Supports W1 context feature claims.,2026-06-30T20:06:26,
outputs/tables/walmart_holiday_markdown_stress_by_window.csv,output_table_csv,Walmart,True,Generated holiday and markdown stress table.,Supports W2 stress-window claims.,2026-06-30T20:06:27,
outputs/tables/walmart_weekly_cadence_constraints.csv,output_table_csv,Walmart,True,Generated weekly-cadence constraints table.,Supports W3 weekly-cadence claims.,2026-06-30T20:06:27,
outputs/tables/walmart_switch_budget_sensitivity.csv,output_table_csv,Walmart,True,Generated switch-budget sensitivity table.,Supports switch-budget claims.,2026-06-30T20:52:20,
outputs/tables/walmart_robustness_summary.csv,output_table_csv,Walmart,True,Generated overall Walmart robustness summary.,Supports Walmart inclusion decision.,2026-06-30T20:06:27,
paper/tables/walmart_robustness_summary_table.csv,paper_table_csv,Walmart,True,Paper-facing Walmart robustness summary.,Supports Walmart numeric claims.,2026-06-30T20:06:27,
paper/figures/walmart_context_aware_vs_context_free_tradeoff.pdf,paper_figure_pdf,Walmart,True,LaTeX-ready Walmart context tradeoff plot.,Supports context robustness visualization.,2026-06-30T20:06:28,
paper/figures/walmart_weekly_cadence_constraints.pdf,paper_figure_pdf,Walmart,True,LaTeX-ready Walmart cadence figure.,Supports cadence robustness visualization.,2026-06-30T20:06:30,
paper/figures/walmart_switch_budget_sensitivity_normalized_loss.pdf,paper_figure_pdf,Walmart,True,LaTeX-ready Walmart switch-budget figure.,Supports switch-budget sensitivity.,2026-06-30T20:52:21,
data/raw/rossmann/train.csv,raw_data,Rossmann,True,Raw Rossmann sales data.,Future work only in IEOM draft.,2026-06-29T19:27:26,
src/data_loaders/rossmann_loader.py,data_loader,Rossmann,True,Rossmann loader exists.,Does not by itself support completed experiment claims.,2026-06-29T18:56:12,
outputs/tables/rossmann_robustness_summary.csv,output_table_csv,Rossmann,False,Potential Rossmann result table.,No current IEOM completed-experiment claim.,,Missing; Rossmann should remain future work.
scripts/run_rossmann_robustness_pipeline.py,experiment_script,Rossmann,False,Potential Rossmann pipeline script.,No current IEOM completed-experiment claim.,,Missing; Rossmann should remain future work.
paper_submission_ieom/main.tex,paper_submission_file,IEOM submission,True,Current IEOM submission artifact.,Supports submission package consistency.,2026-07-01T08:08:52,
paper_submission_ieom/main.pdf,paper_submission_file,IEOM submission,True,Current IEOM submission artifact.,Supports submission package consistency.,2026-07-01T17:19:39,
paper_submission_ieom/README.md,paper_submission_file,IEOM submission,True,Current IEOM submission artifact.,Supports submission package consistency.,2026-07-01T17:19:48,
paper_submission_ieom/ieom_compliance_checklist.md,paper_submission_file,IEOM submission,True,Current IEOM submission artifact.,Supports submission package consistency.,2026-07-01T17:19:11,
