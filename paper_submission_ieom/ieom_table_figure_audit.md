# IEOM Table and Figure Audit

This audit covers tables and figures included in `paper_submission_ieom/main.tex` through section inputs.

number,type,title_or_caption,source_file,source_exists,mentioned_in_text,interpreted_in_text,table_title_above_or_figure_caption_below,specific_finding_supported
Figure 1,Figure,Four-layer framework for feasibility-constrained demand planning.,paper_submission_ieom/figures/ieom_framework_flowchart.pdf,yes,yes,yes,yes,Shows all candidate strategies passing through the same feasibility scoring layer.
Table 1,Table,"Framework components, evidence, and operational use.",paper_submission_ieom/tables/method_alignment_table.tex,yes,yes,yes,yes,Connects method components to evidence and operational use.
Table 2,Table,Dataset roles and experimental purpose.,paper_submission_ieom/tables/dataset_roles_table.tex,yes,yes,yes,yes,"Defines DataCo, Favorita, M5, Walmart, and Rossmann scope."
Table 3,Table,Core numerical evidence for the planning-execution gap.,paper_submission_ieom/tables/core_numerical_summary_table.tex,yes,yes,yes,yes,Supports Finding 1 and overall thesis quantification.
Table 4,Table,M5 hierarchy and intermittency robustness summary.,paper_submission_ieom/tables/m5_robustness_table.tex,yes,yes,yes,yes,Supports Finding 5 on hierarchy and intermittency.
Table 5,Table,Walmart business-context robustness summary.,paper_submission_ieom/tables/walmart_business_context_table.tex,yes,yes,yes,yes,Supports Finding 6 on business context and weekly cadence.
Figure 2,Figure,Favorita accuracy and execution penalty frontier.,paper_submission_ieom/figures/favorita_accuracy_execution_frontier.png,yes,yes,yes,yes,Supports graphical claim that WAPE and execution penalty do not move together.
Figure 3,Figure,Favorita normalized planning loss by DataCo-informed execution scenario.,paper_submission_ieom/figures/favorita_normalized_loss_by_execution_scenario.png,yes,yes,yes,yes,Supports graphical scenario-validation claim.
Table 6,Table,Deployment guidance from empirical findings.,paper_submission_ieom/tables/deployment_guidance_table.tex,yes,yes,yes,yes,Translates empirical findings into practitioner recommendations.


## Audit Decision

- All included tables and figures support distinct findings or required methodology explanation.
- The previous dense strategy-family table was removed from the manuscript.
- No table or figure is included solely as an artifact dump.