"""Export experiment outputs into LaTeX-ready paper assets."""

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from reporting.latex_export import (
    copy_figure_file,
    export_summary_table,
    latex_figure_snippet,
    latex_table_snippet,
    write_asset_manifest,
)
from utils.logging_utils import setup_logger


DEFAULT_TABLE_NAMES = {
    "forecast_metrics_table",
    "inventory_cost_table",
    "stability_metrics_table",
    "planning_utility_table",
    "ablation_results_table",
    "favorita_forecast_metrics",
    "favorita_inventory_metrics",
    "favorita_stability_metrics",
    "favorita_planning_utility",
}

DEFAULT_TABLE_OUTPUT_NAMES = {
    "favorita_forecast_metrics": "favorita_forecast_metrics_table",
    "favorita_inventory_metrics": "favorita_inventory_metrics_table",
    "favorita_stability_metrics": "favorita_stability_metrics_table",
    "favorita_planning_utility": "favorita_planning_utility_table",
}

DEFAULT_FIGURE_NAMES = {
    "accuracy_vs_inventory_cost.pdf",
    "accuracy_vs_volatility.pdf",
    "execution_capacity_stress_test.pdf",
    "planning_signal_example.pdf",
    "pareto_accuracy_stability.pdf",
    "favorita_accuracy_vs_inventory_cost.pdf",
    "favorita_accuracy_vs_volatility.pdf",
    "favorita_example_planning_signal.pdf",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for LaTeX asset export."""
    parser = argparse.ArgumentParser(description="Export outputs into LaTeX-ready paper assets.")
    parser.add_argument("--input-table-dir", default="outputs/tables")
    parser.add_argument("--input-figure-dir", default="outputs/figures")
    parser.add_argument("--paper-table-dir", default="paper/tables")
    parser.add_argument("--paper-figure-dir", default="paper/figures")
    parser.add_argument("--manifest-path", default="paper/asset_manifest.md")
    parser.add_argument("--numeric-precision", type=int, default=3)
    parser.add_argument("--all-tables", action="store_true", help="Export every CSV table in the input table directory.")
    parser.add_argument("--all-figures", action="store_true", help="Copy every supported figure in the input figure directory.")
    return parser.parse_args()


def main() -> None:
    """Export available result tables and figures for the LaTeX manuscript."""
    args = parse_args()
    logger = setup_logger("export_latex_assets")

    input_table_dir = Path(args.input_table_dir)
    input_figure_dir = Path(args.input_figure_dir)
    paper_table_dir = Path(args.paper_table_dir)
    paper_figure_dir = Path(args.paper_figure_dir)
    paper_table_dir.mkdir(parents=True, exist_ok=True)
    paper_figure_dir.mkdir(parents=True, exist_ok=True)

    exported_tables = export_tables(
        input_table_dir=input_table_dir,
        paper_table_dir=paper_table_dir,
        numeric_precision=args.numeric_precision,
        include_all=args.all_tables,
        logger=logger,
    )
    exported_figures = export_figures(
        input_figure_dir=input_figure_dir,
        paper_figure_dir=paper_figure_dir,
        include_all=args.all_figures,
        logger=logger,
    )
    manifest = write_asset_manifest(args.manifest_path, exported_tables, exported_figures)
    logger.info("Wrote paper asset manifest to %s", manifest)


def export_tables(input_table_dir: Path, paper_table_dir: Path, numeric_precision: int, include_all: bool, logger) -> List[Path]:
    """Export selected CSV result tables into CSV and LaTeX paper assets."""
    if not input_table_dir.exists():
        logger.info("Input table directory does not exist yet: %s", input_table_dir)
        return []

    csv_files = sorted(input_table_dir.glob("*.csv"))
    if not include_all:
        csv_files = [path for path in csv_files if path.stem in DEFAULT_TABLE_NAMES]

    exported_tex_paths: List[Path] = []
    for csv_path in csv_files:
        data = pd.read_csv(csv_path)
        table_stem = csv_path.stem
        output_stem = DEFAULT_TABLE_OUTPUT_NAMES.get(table_stem, table_stem)
        paths = export_summary_table(
            data=data,
            table_name=output_stem,
            output_dir=paper_table_dir,
            caption=table_stem.replace("_", " ").title(),
            label="tab:{}".format(table_stem.replace("_", "-")),
            numeric_precision=numeric_precision,
        )
        exported_tex_paths.append(paths["tex"])
        snippet_path = paper_table_dir / "{}_snippet.tex".format(output_stem)
        snippet_path.write_text(latex_table_snippet("tables/{}.tex".format(output_stem)), encoding="utf-8")
        logger.info("Exported table %s to %s", csv_path, paths["tex"])

    if not exported_tex_paths:
        logger.info("No matching CSV tables were found in %s", input_table_dir)
    return exported_tex_paths


def export_figures(input_figure_dir: Path, paper_figure_dir: Path, include_all: bool, logger) -> List[Path]:
    """Copy selected figure assets into the paper figure directory."""
    if not input_figure_dir.exists():
        logger.info("Input figure directory does not exist yet: %s", input_figure_dir)
        return []

    supported_suffixes = {".pdf", ".png", ".jpg", ".jpeg"}
    figure_files = [path for path in sorted(input_figure_dir.iterdir()) if path.suffix.lower() in supported_suffixes]
    if not include_all:
        figure_files = [path for path in figure_files if path.name in DEFAULT_FIGURE_NAMES]

    exported_paths: List[Path] = []
    for figure_path in figure_files:
        exported = copy_figure_file(figure_path, target_dir=paper_figure_dir)
        exported_paths.append(exported)
        if exported.suffix.lower() == ".pdf":
            snippet_path = paper_figure_dir / "{}_snippet.tex".format(exported.stem)
            snippet_path.write_text(
                latex_figure_snippet(
                    "figures/{}".format(exported.name),
                    caption=exported.stem.replace("_", " ").title(),
                    label="fig:{}".format(exported.stem.replace("_", "-")),
                ),
                encoding="utf-8",
            )
        logger.info("Copied figure %s to %s", figure_path, exported)

    if not exported_paths:
        logger.info("No matching figure files were found in %s", input_figure_dir)
    return exported_paths


if __name__ == "__main__":
    main()
