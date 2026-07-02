# IEOM Submission Draft

This folder contains the IEOM-focused manuscript draft for:

**Beyond Forecast Accuracy: Feasibility-Constrained Demand Planning**

The current source of truth is `main.tex`. The draft uses DataCo, Favorita, M5, and Walmart evidence that was already generated in the main project. No new experiments were run for this revision pass.

## Scope

- DataCo is used as an empirical execution-risk anchor.
- Favorita is used as the main forecast-to-plan proof of concept.
- M5 is used for hierarchy and intermittent-demand robustness.
- Walmart is used for weekly business-context robustness.
- Rossmann is not treated as a completed IEOM experiment in this draft.

## Main Included Evidence

- Accuracy-first mismatch rate: 91.1% across 123 paper-facing evaluation groups.
- Accuracy-first planning-execution gap rate: 15.4%.
- Best deployable operational-loss planning-execution gap rate: 2.7%.
- DataCo-informed execution-risk scenarios from baseline to severe.
- M5 robustness across planning grain and intermittency groups.
- Walmart robustness across context-aware features, holiday and markdown windows, weekly cadence constraints, and switch-budget sensitivity.

## Build

From this folder:

```bash
pdflatex main.tex
pdflatex main.tex
```

or, if available:

```bash
latexmk -pdf main.tex
```

The draft was compiled successfully with `pdflatex` on July 2, 2026. The current generated PDF is `main.pdf`.

## Open Items Before Submission

- Replace author, affiliation, email, acknowledgement, and biography placeholders.
- Confirm final IEOM formatting against the official template.
- Re-check all table and figure numbering after any future additions.
- Convert to the final submission format required by IEOM if the portal requires a Word document.

## File Map

- `main.tex`: manuscript driver.
- `sections/`: manuscript sections.
- `tables/`: compact IEOM-facing LaTeX tables.
- `figures/`: compact IEOM-facing figures.
- `references.tex`: alphabetized author-year reference list.
- `ieom_compliance_checklist.md`: submission-readiness checklist.
- `ieom_artifact_consistency_audit.md`: artifact-level consistency audit.
- `ieom_claim_number_audit.md`: manuscript number traceability audit.
- `ieom_table_figure_audit.md`: table and figure consistency audit.
- `walmart_completion_audit.md`: Walmart completion audit.
- `walmart_ieom_inclusion_decision.md`: Walmart IEOM inclusion decision.

## Revision Notes

- Figure 1 is a four-layer framework diagram covering candidate forecasts and policies, forecast-to-plan conversion, feasibility scoring, and decision/validation.
- Algorithm 1 summarizes the feasibility-constrained selection procedure in compact IEOM-style steps.
- Section 5 is organized around six mechanism-level findings rather than dataset chronology.
- Section 5.4 answers baseline, scenario, algorithmic, and robustness validation questions.
- Walmart is now included as a completed robustness module, consistent with the generated Walmart tables and figures in the main paper asset folders.
- Phase 0 audit deliverables were added before the manuscript rewrite.
- Output CSV audits are saved under `outputs/tables/`.
