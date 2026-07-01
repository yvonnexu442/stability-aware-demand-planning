# IEOM Submission Draft

This folder contains the IEOM-focused manuscript draft for:

**From Accurate Forecasts to Feasible Plans: Feasibility-Constrained Model Selection for Operational Demand Planning**

The current source of truth is `main.tex`. The draft uses only DataCo, Favorita, and M5 evidence that was already generated in the main project. No new experiments were run for this revision pass.

## Scope

- DataCo is used as an empirical execution-risk anchor.
- Favorita is used as the main forecast-to-plan proof of concept.
- M5 is used for hierarchy and intermittent-demand robustness.
- Walmart and Rossmann are not treated as completed IEOM experiments in this draft.

## Main Included Evidence

- Accuracy-first mismatch rate: 91.1% across 123 paper-facing evaluation groups.
- Accuracy-first planning-execution gap rate: 15.4%.
- Best deployable operational-loss planning-execution gap rate: 2.7%.
- DataCo-informed execution-risk scenarios from baseline to severe.
- M5 robustness across planning grain and intermittency groups.

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

The draft was compiled successfully with `pdflatex` on July 1, 2026. The current generated PDF is `main.pdf`.

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
