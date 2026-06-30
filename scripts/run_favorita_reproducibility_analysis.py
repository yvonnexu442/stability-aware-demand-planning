"""Run Favorita sample-size reproducibility analysis.

This script reuses the existing Favorita pipeline across quick, medium, and
full sample sizes, then builds cross-mode comparison tables and paper figures.
It is intentionally a reproducibility wrapper rather than a second pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from reporting.latex_export import export_summary_table
from utils.logging_utils import setup_logger
from visualization.plots import (
    apply_paper_style,
    format_axis,
    place_legend,
    save_paper_figure,
    strategy_color,
    strategy_linestyle,
    strategy_marker,
)


RUN_MODE_ORDER = ["quick", "medium", "full"]
RUN_MODE_LABELS = {
    "quick": "quick_mode",
    "medium": "medium_mode",
    "full": "full_mode",
}
KEY_METHODS = [
    "global_best_model",
    "individual_global_lightgbm",
    "individual_global_xgboost",
    "stability_aware_selector",
    "feasibility_aware_selector",
    "individual_moving_average",
]


def parse_args() -> argparse.Namespace:
    """Parse reproducibility-analysis arguments."""
    parser = argparse.ArgumentParser(description="Run Favorita medium/full reproducibility analysis.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the experiment config file.")
    parser.add_argument("--raw-data-dir", default=None, help="Directory containing local Favorita CSV files.")
    parser.add_argument("--output-root", default="outputs/favorita_sample_size_runs", help="Per-mode pipeline output root.")
    parser.add_argument("--output-table-dir", default="outputs/tables", help="Directory for comparison CSV tables.")
    parser.add_argument("--output-figure-dir", default="outputs/figures", help="Directory for comparison PNG figures.")
    parser.add_argument("--paper-table-dir", default="paper/tables", help="Directory for LaTeX comparison tables.")
    parser.add_argument("--paper-figure-dir", default="paper/figures", help="Directory for LaTeX comparison figures.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["quick", "medium", "full"],
        help="Run modes to evaluate: quick, medium, full, or *_mode aliases.",
    )
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable used to run the base pipeline.")
    parser.add_argument(
        "--full-max-ml-training-rows",
        type=int,
        default=100000,
        help="Full-mode global ML training row cap used to avoid native memory failures.",
    )
    parser.add_argument(
        "--full-max-series",
        type=int,
        default=None,
        help="Optional full-mode series cap used when the default full run exceeds local memory.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Do not rerun a mode when its output tables already exist.")
    parser.add_argument("--skip-pipeline", action="store_true", help="Only rebuild comparison outputs from existing per-mode tables.")
    parser.add_argument("--skip-ml", action="store_true", help="Pass --skip-ml to the base Favorita pipeline.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop immediately if a per-mode pipeline run fails.")
    parser.add_argument("--full-failure-note", default=None, help="Optional note documenting why full-mode results are absent.")
    return parser.parse_args()


def main() -> None:
    """Run sample-size reproducibility analysis and export paper assets."""
    args = parse_args()
    logger = setup_logger("favorita_reproducibility_analysis")

    modes = [_normalize_mode(mode) for mode in args.modes]
    output_root = Path(args.output_root)
    paper_mode_root = Path("paper/favorita_sample_size_runs")
    output_root.mkdir(parents=True, exist_ok=True)
    paper_mode_root.mkdir(parents=True, exist_ok=True)

    failed_modes: Dict[str, str] = {}
    if not args.skip_pipeline:
        for mode in modes:
            try:
                _run_pipeline_for_mode(
                    mode=mode,
                    args=args,
                    output_root=output_root,
                    paper_mode_root=paper_mode_root,
                    logger=logger,
                )
            except subprocess.CalledProcessError as exc:
                mode_label = RUN_MODE_LABELS[mode]
                failure_note = (
                    "{} failed while running the existing Favorita pipeline. "
                    "Exit code: {}."
                ).format(mode_label, exc.returncode)
                failed_modes[mode] = failure_note
                logger.error(failure_note)
                if args.stop_on_failure:
                    raise
    else:
        logger.info("Skipping per-mode pipeline runs and rebuilding comparison outputs from existing tables.")

    available_modes, missing_modes = _available_modes(output_root, modes)
    for mode in missing_modes:
        if mode not in failed_modes:
            mode_label = RUN_MODE_LABELS[mode]
            failed_modes[mode] = "{} output tables are unavailable in {}.".format(mode_label, output_root)
    if not available_modes:
        raise FileNotFoundError("No completed Favorita sample-size runs were found under {}.".format(output_root))

    comparison = build_sample_size_comparison(output_root, available_modes)
    summary = build_reproducibility_summary(
        comparison,
        full_failure_note=args.full_failure_note,
        failed_modes=failed_modes,
    )

    output_table_dir = Path(args.output_table_dir)
    output_figure_dir = Path(args.output_figure_dir)
    paper_table_dir = Path(args.paper_table_dir)
    paper_figure_dir = Path(args.paper_figure_dir)
    output_table_dir.mkdir(parents=True, exist_ok=True)
    output_figure_dir.mkdir(parents=True, exist_ok=True)
    paper_table_dir.mkdir(parents=True, exist_ok=True)
    paper_figure_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(output_table_dir / "favorita_sample_size_comparison.csv", index=False)
    summary.to_csv(output_table_dir / "favorita_reproducibility_summary.csv", index=False)
    export_reproducibility_tables(comparison, summary, paper_table_dir)
    make_reproducibility_figures(comparison, output_figure_dir, paper_figure_dir)

    logger.info("Saved Favorita sample-size comparison to %s.", output_table_dir / "favorita_sample_size_comparison.csv")
    logger.info("Saved Favorita reproducibility summary to %s.", output_table_dir / "favorita_reproducibility_summary.csv")


def _available_modes(output_root: Path, modes: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split requested modes into modes with completed output tables and missing modes."""
    available = []
    missing = []
    for mode in modes:
        mode_label = RUN_MODE_LABELS[mode]
        table_dir = output_root / mode_label / "tables"
        required_tables = [
            table_dir / "favorita_planning_utility.csv",
            table_dir / "favorita_data_quality_report.csv",
        ]
        if all(path.exists() for path in required_tables):
            available.append(mode)
        else:
            missing.append(mode)
    return available, missing


def _run_pipeline_for_mode(
    mode: str,
    args: argparse.Namespace,
    output_root: Path,
    paper_mode_root: Path,
    logger,
) -> None:
    """Run the existing Favorita pipeline for a single sample-size mode."""
    mode_label = RUN_MODE_LABELS[mode]
    output_dir = output_root / mode_label
    required_tables = [
        output_dir / "tables" / "favorita_planning_utility.csv",
        output_dir / "tables" / "execution_capacity_stress_test.csv",
        output_dir / "tables" / "weight_sensitivity_results.csv",
        output_dir / "tables" / "pareto_summary.csv",
    ]
    if args.skip_existing and all(path.exists() for path in required_tables):
        logger.info("Skipping %s because required output tables already exist.", mode_label)
        return

    paper_table_dir = paper_mode_root / mode_label / "tables"
    paper_figure_dir = paper_mode_root / mode_label / "figures"
    manifest_path = paper_mode_root / mode_label / "asset_manifest.md"
    command = [
        args.python_executable,
        "scripts/run_favorita_minimal_pipeline.py",
        "--config",
        args.config,
        "--run-mode",
        mode,
        "--output-dir",
        str(output_dir),
        "--paper-table-dir",
        str(paper_table_dir),
        "--paper-figure-dir",
        str(paper_figure_dir),
        "--asset-manifest-path",
        str(manifest_path),
    ]
    if args.raw_data_dir is not None:
        command.extend(["--raw-data-dir", args.raw_data_dir])
    if mode == "full" and args.full_max_series is not None:
        command.extend(["--max-series", str(int(args.full_max_series))])
    if mode == "full" and args.full_max_ml_training_rows is not None:
        command.extend(["--max-ml-training-rows", str(int(args.full_max_ml_training_rows))])
    if args.skip_ml:
        command.append("--skip-ml")

    logger.info("Running Favorita pipeline for %s.", mode_label)
    subprocess.run(command, check=True)


def build_sample_size_comparison(output_root: Path, modes: Sequence[str]) -> pd.DataFrame:
    """Build the cross-mode strategy comparison table."""
    records: List[pd.DataFrame] = []
    for mode in modes:
        mode_label = RUN_MODE_LABELS[mode]
        table_dir = output_root / mode_label / "tables"
        planning_path = table_dir / "favorita_planning_utility.csv"
        quality_path = table_dir / "favorita_data_quality_report.csv"
        if not planning_path.exists():
            raise FileNotFoundError("Missing planning utility table for {}: {}".format(mode_label, planning_path))
        if not quality_path.exists():
            raise FileNotFoundError("Missing data quality report for {}: {}".format(mode_label, quality_path))

        planning = pd.read_csv(planning_path)
        quality = pd.read_csv(quality_path).iloc[0]
        frame = pd.DataFrame(
            {
                "run_mode": mode_label,
                "number_of_series": int(quality["series_count"]),
                "number_of_rows": int(quality["row_count"]),
                "method_name": planning["strategy"].map(_short_strategy_label),
                "strategy": planning["strategy"],
                "WAPE": planning["weighted_absolute_percentage_error"],
                "inventory_cost": planning["total_inventory_cost"],
                "planning_volatility": planning["planning_signal_volatility_total"],
                "execution_penalty": planning["execution_adaptation_penalty_total"],
                "execution_violation_rate": planning["execution_violation_rate"],
                "total_planning_loss": planning["total_planning_loss"],
            }
        )
        records.append(frame)
    comparison = pd.concat(records, ignore_index=True)
    comparison["_mode_order"] = comparison["run_mode"].map({RUN_MODE_LABELS[mode]: index for index, mode in enumerate(RUN_MODE_ORDER)})
    comparison["_method_order"] = comparison["strategy"].map({strategy: index for index, strategy in enumerate(KEY_METHODS)}).fillna(99)
    return comparison.sort_values(["_mode_order", "_method_order", "total_planning_loss"]).drop(columns=["_mode_order", "_method_order"])


def build_reproducibility_summary(
    comparison: pd.DataFrame,
    full_failure_note: Optional[str] = None,
    failed_modes: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Return data-driven answers to the reproducibility questions."""
    records = [
        _ranking_stability_record(comparison),
        _accuracy_first_record(comparison),
        _execution_burden_record(comparison),
        _tradeoff_visibility_record(comparison),
    ]
    if full_failure_note:
        records.append(
            {
                "question": "Was full-mode completed in this environment?",
                "answer": "no",
                "evidence": str(full_failure_note),
            }
        )
    for mode, note in sorted((failed_modes or {}).items(), key=lambda item: RUN_MODE_ORDER.index(item[0])):
        mode_label = RUN_MODE_LABELS[mode]
        if mode == "full" and full_failure_note:
            continue
        records.append(
            {
                "question": "Was {} completed in this environment?".format(mode_label),
                "answer": "no",
                "evidence": str(note),
            }
        )
    return pd.DataFrame(records)


def _ranking_stability_record(comparison: pd.DataFrame) -> Dict[str, object]:
    """Summarize whether total-loss rankings remain broadly stable."""
    rank_table = comparison.pivot_table(index="strategy", columns="run_mode", values="total_planning_loss", aggfunc="first")
    quick_label = RUN_MODE_LABELS["quick"]
    correlations = []
    if quick_label in rank_table.columns:
        quick_rank = rank_table[quick_label].rank(method="average")
        for column in rank_table.columns:
            if column == quick_label:
                continue
            joined = pd.concat([quick_rank, rank_table[column].rank(method="average")], axis=1).dropna()
            if len(joined) > 1:
                correlations.append(float(joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman")))
    average_correlation = float(pd.Series(correlations).mean()) if correlations else float("nan")
    top_methods = (
        comparison.sort_values("total_planning_loss")
        .groupby("run_mode")
        .first()["method_name"]
        .to_dict()
    )
    result = "stable" if correlations and average_correlation >= 0.8 else "mixed"
    return {
        "question": "Are method rankings broadly stable?",
        "answer": result,
        "evidence": "Top total-loss methods by mode: {}. Mean Spearman rank correlation versus quick mode: {:.3f}.".format(
            _format_mapping(top_methods),
            average_correlation,
        ),
    }


def _accuracy_first_record(comparison: pd.DataFrame) -> Dict[str, object]:
    """Summarize whether LightGBM/global-best remains the strongest accuracy-first method."""
    best_accuracy = comparison.sort_values("WAPE").groupby("run_mode").first()
    best_labels = best_accuracy["method_name"].to_dict()
    all_lightgbm = all(label in {"Global Best", "LightGBM"} for label in best_labels.values())
    return {
        "question": "Does LightGBM/global-best remain the strongest accuracy-first method?",
        "answer": "yes" if all_lightgbm else "mixed",
        "evidence": "Lowest-WAPE methods by mode: {}.".format(_format_mapping(best_labels)),
    }


def _execution_burden_record(comparison: pd.DataFrame) -> Dict[str, object]:
    """Summarize whether feasibility-aware selection reduces execution burden."""
    comparisons = _compare_strategy_to_global_best(comparison, "feasibility_aware_selector")
    lower_penalty_count = int(comparisons["execution_penalty_reduction_pct"].gt(0).sum())
    lower_violation_count = int(comparisons["violation_rate_reduction_pct"].gt(0).sum())
    total_modes = len(comparisons)
    answer = "yes" if lower_penalty_count == total_modes and lower_violation_count == total_modes else "mixed"
    return {
        "question": "Do feasibility-aware methods consistently reduce execution burden?",
        "answer": answer,
        "evidence": "Feasibility-aware lowers execution penalty in {}/{} modes and violation rate in {}/{} modes. Mean penalty reduction versus global-best: {:.1f}%.".format(
            lower_penalty_count,
            total_modes,
            lower_violation_count,
            total_modes,
            float(comparisons["execution_penalty_reduction_pct"].mean()),
        ),
    }


def _tradeoff_visibility_record(comparison: pd.DataFrame) -> Dict[str, object]:
    """Summarize whether the accuracy-feasibility tradeoff remains visible."""
    comparisons = _compare_strategy_to_global_best(comparison, "feasibility_aware_selector")
    visible = comparisons[
        (comparisons["WAPE_gap"] > 0)
        & (comparisons["inventory_cost_gap"] > 0)
        & (comparisons["execution_penalty_reduction_pct"] > 0)
    ]
    answer = "yes" if len(visible) == len(comparisons) else "mixed"
    return {
        "question": "Does the tradeoff remain visible as sample size increases?",
        "answer": answer,
        "evidence": "Feasibility-aware has higher WAPE and inventory cost but lower execution penalty than global-best in {}/{} modes.".format(
            len(visible),
            len(comparisons),
        ),
    }


def _compare_strategy_to_global_best(comparison: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Return per-mode deltas for one strategy against global-best."""
    global_best = comparison[comparison["strategy"] == "global_best_model"].set_index("run_mode")
    candidate = comparison[comparison["strategy"] == strategy].set_index("run_mode")
    joined = candidate.join(global_best, lsuffix="_candidate", rsuffix="_global", how="inner")
    joined["WAPE_gap"] = joined["WAPE_candidate"] - joined["WAPE_global"]
    joined["inventory_cost_gap"] = joined["inventory_cost_candidate"] - joined["inventory_cost_global"]
    joined["execution_penalty_reduction_pct"] = (
        (joined["execution_penalty_global"] - joined["execution_penalty_candidate"])
        / joined["execution_penalty_global"].replace(0, pd.NA)
        * 100.0
    )
    joined["violation_rate_reduction_pct"] = (
        (joined["execution_violation_rate_global"] - joined["execution_violation_rate_candidate"])
        / joined["execution_violation_rate_global"].replace(0, pd.NA)
        * 100.0
    )
    return joined.reset_index()


def export_reproducibility_tables(comparison: pd.DataFrame, summary: pd.DataFrame, paper_table_dir: Path) -> None:
    """Export comparison and summary tables to the LaTeX paper directory."""
    export_summary_table(
        data=comparison.drop(columns=["strategy"]),
        table_name="favorita_sample_size_comparison",
        output_dir=paper_table_dir,
        caption="Favorita sample-size comparison across quick, medium, and full reproducibility runs.",
        label="tab:favorita-sample-size-comparison",
        numeric_precision=3,
        column_renames={
            "run_mode": "Run Mode",
            "number_of_series": "Series",
            "number_of_rows": "Rows",
            "method_name": "Method",
            "WAPE": "WAPE",
            "inventory_cost": "Inventory Cost",
            "planning_volatility": "Planning Volatility",
            "execution_penalty": "Execution Penalty",
            "execution_violation_rate": "Violation Rate",
            "total_planning_loss": "Total Planning Loss",
        },
        resize_to_textwidth=True,
    )
    export_summary_table(
        data=summary,
        table_name="favorita_reproducibility_summary",
        output_dir=paper_table_dir,
        caption="Favorita reproducibility summary across sample sizes.",
        label="tab:favorita-reproducibility-summary",
        numeric_precision=3,
        column_renames={
            "question": "Question",
            "answer": "Answer",
            "evidence": "Evidence",
        },
        resize_to_textwidth=True,
    )


def make_reproducibility_figures(comparison: pd.DataFrame, output_figure_dir: Path, paper_figure_dir: Path) -> None:
    """Create sample-size robustness figures."""
    apply_paper_style()
    plot_data = comparison[comparison["strategy"].isin(KEY_METHODS)].copy()
    mode_order = {RUN_MODE_LABELS[mode]: index for index, mode in enumerate(RUN_MODE_ORDER)}
    plot_data["_mode_order"] = plot_data["run_mode"].map(mode_order)
    plot_data = plot_data.sort_values(["strategy", "_mode_order"])

    _save_accuracy_feasibility_plot(
        plot_data=plot_data,
        png_path=output_figure_dir / "favorita_sample_size_accuracy_vs_feasibility.png",
        pdf_path=paper_figure_dir / "favorita_sample_size_accuracy_vs_feasibility.pdf",
    )
    _save_execution_penalty_plot(
        plot_data=plot_data,
        png_path=output_figure_dir / "favorita_sample_size_execution_penalty.png",
        pdf_path=paper_figure_dir / "favorita_sample_size_execution_penalty.pdf",
    )


def _save_accuracy_feasibility_plot(plot_data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save WAPE versus execution penalty across sample sizes."""
    fig, ax = plt.subplots(figsize=(7.25, 4.35))
    for index, strategy in enumerate(KEY_METHODS):
        strategy_data = plot_data[plot_data["strategy"] == strategy]
        if strategy_data.empty:
            continue
        ax.plot(
            strategy_data["WAPE"],
            strategy_data["execution_penalty"],
            color=strategy_color(strategy, index),
            linestyle=strategy_linestyle(strategy),
            marker=strategy_marker(strategy),
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.55,
            linewidth=1.8,
            label=_short_strategy_label(strategy),
        )
    format_axis(ax, x_label="Weighted Absolute Percentage Error", y_label="Execution Penalty", grid_axis="both")
    place_legend(ax, columns=3)
    save_paper_figure(fig, png_path, pdf_path)


def _save_execution_penalty_plot(plot_data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save execution penalty by sample-size mode."""
    fig, ax = plt.subplots(figsize=(7.25, 4.35))
    observed_modes = [
        RUN_MODE_LABELS[mode]
        for mode in RUN_MODE_ORDER
        if RUN_MODE_LABELS[mode] in set(plot_data["run_mode"])
    ]
    mode_positions = {mode: index for index, mode in enumerate(observed_modes)}
    for index, strategy in enumerate(KEY_METHODS):
        strategy_data = plot_data[plot_data["strategy"] == strategy].sort_values("_mode_order")
        if strategy_data.empty:
            continue
        ax.plot(
            strategy_data["run_mode"].map(mode_positions),
            strategy_data["execution_penalty"],
            color=strategy_color(strategy, index),
            linestyle=strategy_linestyle(strategy),
            marker=strategy_marker(strategy),
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.55,
            linewidth=1.8,
            label=_short_strategy_label(strategy),
        )
    ax.set_xticks(list(mode_positions.values()))
    ax.set_xticklabels([mode.replace("_mode", "").title() for mode in observed_modes])
    format_axis(ax, x_label="Run Mode", y_label="Execution Penalty")
    place_legend(ax, columns=3)
    save_paper_figure(fig, png_path, pdf_path)


def _normalize_mode(mode: str) -> str:
    """Normalize run-mode aliases."""
    value = str(mode).strip().lower()
    aliases = {
        "quick": "quick",
        "quick_mode": "quick",
        "medium": "medium",
        "medium_mode": "medium",
        "full": "full",
        "full_mode": "full",
    }
    if value not in aliases:
        raise ValueError("Unsupported run mode: {}".format(mode))
    return aliases[value]


def _short_strategy_label(strategy: str) -> str:
    """Return a compact English label for strategy names."""
    replacements = {
        "individual_naive_last_value": "Naive",
        "individual_seasonal_naive": "Seasonal",
        "individual_moving_average": "Moving Avg.",
        "individual_exponential_smoothing": "Exp. Smooth",
        "individual_global_lightgbm": "LightGBM",
        "individual_global_xgboost": "XGBoost",
        "individual_global_sklearn": "sklearn",
        "global_best_model": "Global Best",
        "family_best_model": "Family Best",
        "stability_aware_selector": "Stability-Aware",
        "feasibility_aware_selector": "Feasibility-Aware",
    }
    return replacements.get(strategy, str(strategy).replace("_", " ").title())


def _format_mapping(values: Mapping[str, object]) -> str:
    """Format a mapping compactly for reproducibility evidence text."""
    ordered_items = [(key, values[key]) for key in sorted(values.keys(), key=lambda item: RUN_MODE_ORDER.index(_normalize_mode(item)))]
    return "; ".join("{}={}".format(key, value) for key, value in ordered_items)


if __name__ == "__main__":
    main()
