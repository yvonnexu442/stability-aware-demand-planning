# Paper Figure Style Guide

This repository treats paper figures and tables as research artifacts, not as ad hoc screenshots. Experiment scripts should generate assets that can be imported directly into the LaTeX manuscript.

## Figure Standards

- Save each research figure as PDF under `paper/figures/` for LaTeX inclusion.
- Save a PNG copy under `outputs/figures/` for quick review.
- Use English titles, axis labels, legends, annotations, and filenames.
- Use restrained journal-style formatting: white background, subtle gridlines, readable axis labels, and minimal decorative elements.
- Use colorblind-safe colors and stable marker shapes so strategy identity does not depend only on color.
- Prefer concise figure titles. The manuscript caption should carry the detailed interpretation.
- Avoid fabricating annotations or result claims. Figures should display measured outputs from the pipeline.

## Table Standards

- Save major result tables as both CSV and LaTeX `.tex` files.
- Use booktabs-style LaTeX tables.
- Use English column names and paper-friendly display labels.
- Round numeric columns consistently across related tables.
- Keep raw or audit-friendly CSV files available so table values can be traced back to experiment outputs.

## Flowchart Standards

- Use LaTeX-native flowcharts for paper diagrams when possible.
- Keep flowcharts structural: they should explain data flow, modeling stages, decision layers, and evaluation outputs.
- Avoid putting empirical conclusions in framework flowcharts before the corresponding experiments have been run.

## Current Plotting Utility

The shared plotting style lives in `src/visualization/plots.py`. It applies a seaborn-based paper theme when seaborn is available and falls back to Matplotlib-compatible settings otherwise. New experiment scripts should reuse these helpers instead of defining custom colors, marker styles, or save behavior in each plotting function.
