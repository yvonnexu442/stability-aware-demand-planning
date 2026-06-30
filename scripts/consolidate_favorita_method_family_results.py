"""Consolidate improved Favorita method-family results for paper assets."""

import argparse
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from evaluation.planning_utility import add_oracle_gap_columns
from reporting.latex_export import export_summary_table
from visualization.plots import apply_paper_style, format_axis, save_paper_figure, strategy_color, strategy_marker


REPRESENTATIVE_STRATEGIES = [
    "global_best_model",
    "family_best_model",
    "simple_ensemble",
    "feasibility_aware_ensemble_inverse_operational_loss",
    "feasibility_aware_smoothed_alpha_0_25",
    "feasibility_aware_smoothed_utility_alpha",
    "feasibility_aware_selector",
    "greedy_feasibility_selector",
    "dp_feasibility_selector",
    "budgeted_dp_feasibility_selector",
    "best_stability_model",
    "individual_moving_average",
    "oracle_dp_feasibility_selector",
    "oracle_realized_demand",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build paper-ready Favorita method-family summary assets.")
    parser.add_argument(
        "--input-table",
        default="outputs/tables/favorita_improved_feasibility_methods.csv",
        help="Improved Favorita method comparison table.",
    )
    parser.add_argument("--output-table-dir", default="outputs/tables")
    parser.add_argument("--output-figure-dir", default="outputs/figures")
    parser.add_argument("--paper-table-dir", default="paper/tables")
    parser.add_argument("--paper-figure-dir", default="paper/figures")
    return parser.parse_args()


def main() -> None:
    """Create the consolidated table and frontier figures."""
    args = parse_args()
    input_path = Path(args.input_table)
    if not input_path.exists():
        raise FileNotFoundError(
            "Improved Favorita result table does not exist: {}. Run the Favorita pipeline first.".format(input_path)
        )

    data = pd.read_csv(input_path)
    summary = build_method_family_summary(data)
    output_table_dir = Path(args.output_table_dir)
    output_table_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_table_dir / "favorita_method_family_summary.csv", index=False)
    export_summary_table(
        data=summary,
        table_name="favorita_method_family_summary_table",
        output_dir=args.paper_table_dir,
        caption="Representative Favorita method-family comparison under DataCo-informed execution scenarios.",
        label="tab:favorita-method-family-summary",
        numeric_precision=3,
        resize_to_textwidth=True,
    )

    output_figure_dir = Path(args.output_figure_dir)
    paper_figure_dir = Path(args.paper_figure_dir)
    output_figure_dir.mkdir(parents=True, exist_ok=True)
    paper_figure_dir.mkdir(parents=True, exist_ok=True)
    plot_slice = representative_plot_slice(summary)
    save_accuracy_execution_plot(
        plot_slice,
        output_figure_dir / "favorita_method_family_accuracy_vs_execution_penalty.png",
        paper_figure_dir / "favorita_method_family_accuracy_vs_execution_penalty.pdf",
    )
    save_inventory_execution_plot(
        plot_slice,
        output_figure_dir / "favorita_method_family_inventory_vs_execution_penalty.png",
        paper_figure_dir / "favorita_method_family_inventory_vs_execution_penalty.pdf",
    )
    save_normalized_loss_by_scenario_plot(
        summary,
        output_figure_dir / "favorita_method_family_normalized_loss_by_scenario.png",
        paper_figure_dir / "favorita_method_family_normalized_loss_by_scenario.pdf",
    )
    save_switching_execution_plot(
        plot_slice,
        output_figure_dir / "favorita_method_family_switching_vs_execution.png",
        paper_figure_dir / "favorita_method_family_switching_vs_execution.pdf",
    )


def build_method_family_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return representative method-family rows from generated Favorita outputs."""
    data = add_oracle_gap_columns(data)
    required = {
        "scenario_name",
        "strategy",
        "method_name",
        "WAPE",
        "inventory_cost",
        "planning_volatility",
        "execution_penalty",
        "execution_violation_rate",
        "model_switch_count",
        "max_period_plan_change_pct",
        "normalized_total_loss",
        "gap_to_dp_oracle",
        "gap_to_perfect_oracle",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Favorita improved results are missing required columns: {}".format(sorted(missing)))

    available_strategies = list(dict.fromkeys(data["strategy"].tolist()))
    representative = [strategy for strategy in REPRESENTATIVE_STRATEGIES if strategy in available_strategies]
    if "best_stability_model" in representative and "individual_moving_average" in representative:
        representative.remove("individual_moving_average")

    table = data[data["strategy"].isin(representative)].copy()
    if "run_mode" not in table.columns:
        table["run_mode"] = "unknown"
    table["method_family"] = table["strategy"].map(method_family).fillna(
        table["method_family"] if "method_family" in table.columns else "Other"
    )

    output_columns = [
        "run_mode",
        "scenario_name",
        "method_family",
        "method_name",
        "strategy",
        "WAPE",
        "inventory_cost",
        "planning_volatility",
        "execution_penalty",
        "execution_violation_rate",
        "model_switch_count",
        "max_period_plan_change_pct",
        "normalized_total_loss",
        "gap_to_dp_oracle",
        "gap_to_perfect_oracle",
    ]
    return table[output_columns].sort_values(["scenario_name", "normalized_total_loss", "method_name"]).reset_index(drop=True)


def method_family(strategy: str) -> str:
    """Map a strategy identifier to a method family."""
    if strategy in {"oracle_dp_feasibility_selector", "oracle_realized_demand"}:
        return "Oracle"
    if strategy in {"global_best_model", "family_best_model"}:
        return "AccuracyFirst"
    if strategy == "simple_ensemble":
        return "ReferenceEnsemble"
    if strategy.startswith("feasibility_aware_ensemble"):
        return "FeasibilityAwareEnsemble"
    if strategy.startswith("feasibility_aware_smoothed"):
        return "FeasibilityAwareSmoothed"
    if strategy in {"greedy_feasibility_selector", "dp_feasibility_selector", "budgeted_dp_feasibility_selector"}:
        return "FeasibilityAwareDP"
    if strategy == "feasibility_aware_selector":
        return "FeasibilityAwareSelector"
    if strategy in {"best_stability_model", "individual_moving_average"}:
        return "StabilityFirst"
    return "Other"


def representative_plot_slice(summary: pd.DataFrame) -> pd.DataFrame:
    """Return one execution-risk scenario for readable frontier scatter plots."""
    scenario_order = summary[["scenario_name", "normalized_total_loss"]].copy()
    if "lambda_execution" in summary.columns:
        scenario_name = summary.sort_values("lambda_execution")["scenario_name"].iloc[-1]
    elif "dataco_severe" in set(summary["scenario_name"]):
        scenario_name = "dataco_severe"
    else:
        scenario_name = scenario_order["scenario_name"].iloc[-1]
    return summary[summary["scenario_name"] == scenario_name].copy()


def save_accuracy_execution_plot(data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save WAPE versus execution penalty frontier plot."""
    save_scatter_with_labels(
        data=data,
        x_column="WAPE",
        y_column="execution_penalty",
        x_label="Weighted Absolute Percentage Error",
        y_label="Execution Penalty",
        png_path=png_path,
        pdf_path=pdf_path,
        color_column="normalized_total_loss",
        size_column="inventory_cost",
    )


def save_inventory_execution_plot(data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save inventory cost versus execution penalty frontier plot."""
    save_scatter_with_labels(
        data=data,
        x_column="inventory_cost",
        y_column="execution_penalty",
        x_label="Inventory Cost",
        y_label="Execution Penalty",
        png_path=png_path,
        pdf_path=pdf_path,
    )


def save_switching_execution_plot(data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save switching burden versus execution penalty plot."""
    save_scatter_with_labels(
        data=data,
        x_column="model_switch_count",
        y_column="execution_penalty",
        x_label="Model Switch Count",
        y_label="Execution Penalty",
        png_path=png_path,
        pdf_path=pdf_path,
    )


def save_scatter_with_labels(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    png_path: Path,
    pdf_path: Path,
    color_column: str = "",
    size_column: str = "",
) -> None:
    """Save a labeled scatter plot for representative methods."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.35, 4.45))
    plot_data = data.dropna(subset=[x_column, y_column]).copy()
    if color_column and color_column in plot_data.columns:
        color_values = plot_data[color_column]
        sizes = scaled_sizes(plot_data[size_column]) if size_column and size_column in plot_data.columns else 70
        scatter = ax.scatter(
            plot_data[x_column],
            plot_data[y_column],
            c=color_values,
            s=sizes,
            cmap="viridis_r",
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label("Normalized Total Loss")
    else:
        for index, row in enumerate(plot_data.itertuples(index=False)):
            ax.scatter(
                getattr(row, x_column),
                getattr(row, y_column),
                s=74,
                marker=strategy_marker(getattr(row, "strategy")),
                color=strategy_color(getattr(row, "strategy"), index),
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
    annotate_points(ax, plot_data, x_column=x_column, y_column=y_column)
    format_axis(ax, x_label=x_label, y_label=y_label, grid_axis="both")
    save_paper_figure(fig, png_path, pdf_path)


def scaled_sizes(values: pd.Series) -> pd.Series:
    """Scale numeric values into readable point sizes."""
    numeric = pd.to_numeric(values, errors="coerce")
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if maximum <= minimum:
        return pd.Series(80, index=values.index)
    return 55 + 95 * (numeric - minimum) / (maximum - minimum)


def annotate_points(ax, data: pd.DataFrame, x_column: str, y_column: str) -> None:
    """Annotate method points with compact labels."""
    for index, row in enumerate(data.itertuples(index=False)):
        x_offset = 5 if index % 2 == 0 else -5
        y_offset = 5 if index % 3 else -9
        ha = "left" if x_offset > 0 else "right"
        ax.annotate(
            str(getattr(row, "method_name")),
            (getattr(row, x_column), getattr(row, y_column)),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=7.1,
            color="#222222",
            ha=ha,
        )


def save_normalized_loss_by_scenario_plot(data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save normalized loss by execution scenario for method families."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.35, 4.45))
    scenario_order = list(dict.fromkeys(data["scenario_name"].tolist()))
    positions = {scenario: index for index, scenario in enumerate(scenario_order)}
    for index, method_name in enumerate(method_line_order(data)):
        method_data = data[data["method_name"] == method_name].copy()
        method_data["scenario_position"] = method_data["scenario_name"].map(positions)
        method_data = method_data.sort_values("scenario_position")
        if method_data.empty:
            continue
        strategy = str(method_data["strategy"].iloc[0])
        ax.plot(
            method_data["scenario_position"],
            method_data["normalized_total_loss"],
            marker=strategy_marker(strategy),
            color=strategy_color(strategy, index),
            linewidth=1.7,
            label=method_name,
        )
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([scenario.replace("_", "\n").title() for scenario in scenario_order])
    format_axis(ax, x_label="DataCo-Informed Execution Scenario", y_label="Normalized Total Loss")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False, fontsize=7.7)
    save_paper_figure(fig, png_path, pdf_path)


def method_line_order(data: pd.DataFrame) -> List[str]:
    """Return a stable line order for the normalized-loss plot."""
    severe = representative_plot_slice(data)
    return severe.sort_values("normalized_total_loss")["method_name"].tolist()


if __name__ == "__main__":
    main()
