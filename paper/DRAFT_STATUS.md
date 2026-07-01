# Paper Draft Status and Next Steps

## Current State

The paper draft is in progress. The repository now contains a compilable LaTeX
paper structure, a formal problem formulation, a framework and methods section,
algorithm pseudocode for the decision layer, an expanded experimental setup, and
a first results narrative based on generated DataCo, Favorita, M5, and Walmart
robustness artifacts.

This status file is a working checklist, not a claim that the manuscript is
submission-ready.

## Generated Empirical Assets

### DataCo

- DataCo execution-risk profiling has been generated.
- Late-delivery-rate percentiles are exported as paper-ready tables.
- DataCo-informed execution-risk scenarios have been generated with an additive
  clipped mapping from late-delivery anchors to `lambda_execution` values.
- DataCo is used as an execution-risk sensitivity anchor, not as an exact
  economic-cost calibration for Favorita or M5.

### Favorita

- Favorita quick-mode, medium-mode, and full-mode outputs have been consolidated
  into paper-ready tables and figures.
- Favorita is treated as a daily store-family demand-planning panel.
- The current results narrative uses Favorita to show the accuracy-feasibility
  tradeoff and the effect of DataCo-informed execution-risk scenarios.

### M5

- M5 quick-mode robustness outputs have been generated.
- The active paper draft includes M5 large-scale replication, hierarchy
  sensitivity, intermittency stress, and DataCo-scenario robustness assets.
- Medium and full M5 runs remain optional future extensions.

### Walmart

- Walmart quick-mode business-context robustness outputs have been generated.
- The active paper draft includes Walmart context-aware versus context-free
  comparison assets and keeps holiday, markdown, and weekly-cadence outputs as
  supplementary robustness material.
- Walmart is treated as a narrow business-context robustness check, not as a new
  forecasting leaderboard.

### Rossmann

- Rossmann raw data are available for a future robustness check.
- Rossmann is not used to support current empirical claims in the active draft.

## Active Manuscript Sections

- `sections/00_abstract.tex`: drafted.
- `sections/01_introduction.tex`: drafted.
- `sections/02_related_work.tex`: drafted, but citations and coverage remain TODO.
- `sections/03_problem_formulation.tex`: drafted.
- `sections/04_framework_and_methods.tex`: drafted.
- `sections/algorithm_pseudocode.tex`: drafted and included by `main.tex`.
- `sections/05_experimental_setup.tex`: expanded and aligned with current code.
- `sections/06_results.tex`: active results narrative with generated assets.
- `sections/07_discussion.tex`: drafted but should be revised after final results
  wording is locked.
- `sections/08_conclusion.tex`: drafted but should be revised after final results
  wording is locked.
- `sections/appendix.tex`: placeholder structure for supplementary materials.

## Current Priorities

1. Keep `paper/main.tex` compiling after each paper-structure change.
2. Tighten the results narrative without adding unsupported claims.
3. Verify that every numeric claim in `sections/06_results.tex` is traceable to a
   generated table or figure.
4. Expand related work with verified citations.
5. Decide whether Rossmann robustness is needed for the paper scope.
6. Move overly large sensitivity tables to the appendix if the main paper becomes
   too long.

## Known Caveats

- Forecast accuracy is reported as a diagnostic metric; it is not part of the
  default normalized operational planning loss.
- Oracle diagnostics are non-deployable and should remain clearly labeled.
- Additional robustness may change wording in the discussion and conclusion.
- The manuscript still needs citation polishing, table pruning, and final
  submission-format cleanup.
