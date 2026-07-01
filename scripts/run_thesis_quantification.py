"""Generate thesis-level quantification tables from experiment outputs."""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from reporting.latex_export import export_summary_table


GROUP_COLUMN_CANDIDATES = [
    "dataset_name",
    "run_mode",
    "scenario_name",
    "capacity_scenario",
    "scenario_id",
    "grain_level",
    "intermittency_bucket",
    "feature_set",
    "window_type",
    "evaluation_mode",
    "experiment_name",
    "lambda_volatility",
    "lambda_switch",
    "lambda_execution",
    "alpha_forecast",
    "beta_inventory",
]

ORACLE_STRATEGIES = {
    "oracle_realized_demand",
    "oracle_dp_feasibility_selector",
    "full_outcome_oracle_dp_feasibility_selector",
}

NON_METHOD_TABLE_PREFIXES = (
    "dataco_",
    "decision_layer_audit",
    "normalization_reference_values",
    "thesis_",
)

STRATEGY_LABELS = {
    "global_best_model": "Global Best",
    "best_accuracy": "Best Accuracy",
    "family_best_model": "Family Best",
    "simple_ensemble": "Simple Ensemble",
    "operational_loss_ensemble": "Operational Ensemble",
    "feasibility_aware_selector": "Feasibility-Aware",
    "stability_aware_selector": "Stability-Aware",
    "best_inventory_cost_model": "Best Inventory",
    "best_stability_model": "Best Stability",
    "greedy_feasibility_selector": "Greedy Feasibility Selector",
    "dp_feasibility_selector": "DP Feasibility Selector",
    "budgeted_dp_feasibility_selector": "Budgeted DP Feasibility Selector",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate thesis-level quantification tables.")
    parser.add_argument("--input-dir", default="outputs/tables")
    parser.add_argument("--output-dir", default="outputs/tables")
    parser.add_argument("--paper-table-dir", default="paper/tables")
    parser.add_argument("--max-paper-rows", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    """Build all thesis quantification outputs."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    paper_table_dir = Path(args.paper_table_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_table_dir.mkdir(parents=True, exist_ok=True)

    method_tables = load_method_tables(input_dir)
    if not method_tables:
        raise RuntimeError("No method-level result tables were found in {}.".format(input_dir))

    accuracy_mismatch = summarize_accuracy_ranking_mismatch(method_tables)
    planning_gap = summarize_planning_execution_gap(method_tables)
    greedy_vs_dp = summarize_greedy_vs_dp(method_tables)
    integrated = summarize_integrated_thesis(accuracy_mismatch, planning_gap, greedy_vs_dp, method_tables)

    write_output_pair(
        accuracy_mismatch,
        "thesis_accuracy_ranking_mismatch",
        output_dir,
        paper_table_dir,
        "Accuracy-first versus operational-loss-optimal deployable strategy mismatch.",
        "tab:thesis-accuracy-ranking-mismatch",
        max_rows=args.max_paper_rows,
    )
    write_output_pair(
        planning_gap,
        "thesis_planning_execution_gap_summary",
        output_dir,
        paper_table_dir,
        "Planning-execution gap rate summary.",
        "tab:thesis-planning-execution-gap-summary",
        max_rows=args.max_paper_rows,
    )
    write_output_pair(
        greedy_vs_dp,
        "thesis_greedy_vs_dp_summary",
        output_dir,
        paper_table_dir,
        "Greedy versus finite-horizon DP selector summary.",
        "tab:thesis-greedy-vs-dp-summary",
        max_rows=args.max_paper_rows,
    )
    write_output_pair(
        integrated,
        "thesis_quantification_summary",
        output_dir,
        paper_table_dir,
        "Integrated thesis quantification summary.",
        "tab:thesis-quantification-summary",
        max_rows=args.max_paper_rows,
    )


def load_method_tables(input_dir: Path) -> List[Tuple[str, pd.DataFrame]]:
    """Load result tables that contain deployable method-level metrics."""
    tables: List[Tuple[str, pd.DataFrame]] = []
    for csv_path in sorted(input_dir.glob("*.csv")):
        if should_skip_table(csv_path):
            continue
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        normalized = normalize_method_table(frame, csv_path.stem)
        if normalized is not None and not normalized.empty:
            tables.append((csv_path.stem, normalized))
    return tables


def should_skip_table(csv_path: Path) -> bool:
    """Return True for non-method or generated thesis tables."""
    stem = csv_path.stem
    return any(stem.startswith(prefix) for prefix in NON_METHOD_TABLE_PREFIXES)


def normalize_method_table(frame: pd.DataFrame, source_table: str) -> Optional[pd.DataFrame]:
    """Return a standardized method table, or None when metrics are missing."""
    if frame.empty:
        return None
    data = frame.copy()
    if "WAPE" not in data.columns and "weighted_absolute_percentage_error" in data.columns:
        data["WAPE"] = data["weighted_absolute_percentage_error"]
    if "method_name" not in data.columns and "strategy" in data.columns:
        data["method_name"] = data["strategy"].map(lambda value: STRATEGY_LABELS.get(str(value), str(value)))
    if "strategy" not in data.columns and "method_name" in data.columns:
        data["strategy"] = data["method_name"].astype(str)
    required = {"strategy", "method_name", "WAPE", "normalized_total_loss", "execution_violation_rate"}
    if required.difference(data.columns):
        return None

    for column in [
        "WAPE",
        "normalized_total_loss",
        "execution_violation_rate",
        "execution_penalty",
        "model_switch_count",
        "gap_to_dp_oracle",
        "gap_to_perfect_oracle",
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["source_table"] = source_table
    if "dataset_name" not in data.columns:
        data["dataset_name"] = infer_dataset_name(source_table)
    data["deployable_inferred"] = infer_deployable(data)
    data = data.dropna(subset=["WAPE", "normalized_total_loss", "execution_violation_rate"])
    return data


def infer_dataset_name(source_table: str) -> str:
    """Infer dataset name from the result table name."""
    lower = source_table.lower()
    if lower.startswith("favorita") or lower in {"execution_capacity_stress_test", "weight_sensitivity_results", "pareto_summary"}:
        return "favorita"
    if lower.startswith("m5"):
        return "m5"
    if lower.startswith("walmart"):
        return "walmart"
    return "unknown"


def infer_deployable(data: pd.DataFrame) -> pd.Series:
    """Infer deployability from explicit metadata or strategy name."""
    if "deployable" in data.columns:
        deployable = data["deployable"].map(parse_bool)
    else:
        deployable = pd.Series(True, index=data.index)
    if "non_deployable_upper_bound" in data.columns:
        deployable = deployable & ~data["non_deployable_upper_bound"].map(parse_bool)
    strategy = data["strategy"].astype(str)
    method = data["method_name"].astype(str)
    oracle_like = strategy.isin(ORACLE_STRATEGIES) | strategy.str.contains("oracle", case=False, na=False)
    oracle_like = oracle_like | method.str.contains("oracle", case=False, na=False)
    return deployable.fillna(True) & ~oracle_like


def parse_bool(value: object) -> bool:
    """Parse common boolean encodings."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return True
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def summarize_accuracy_ranking_mismatch(method_tables: Sequence[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Summarize how often accuracy-first differs from operational best."""
    rows = []
    for source_table, table in method_tables:
        group_records = []
        for group_key, group in iter_groups(table):
            deployable = group[group["deployable_inferred"]].copy()
            if len(deployable) < 2:
                continue
            if not contains_explicit_accuracy_first(deployable):
                continue
            accuracy_first = sort_accuracy_first(deployable).iloc[0]
            operational_best = deployable.sort_values(["normalized_total_loss", "method_name", "strategy"]).iloc[0]
            loss_gap = float(accuracy_first["normalized_total_loss"] - operational_best["normalized_total_loss"])
            group_records.append(
                {
                    "source_table": source_table,
                    "group_key": group_key,
                    "accuracy_first_method": accuracy_first["method_name"],
                    "operational_best_method": operational_best["method_name"],
                    "accuracy_first_normalized_loss": float(accuracy_first["normalized_total_loss"]),
                    "operational_best_normalized_loss": float(operational_best["normalized_total_loss"]),
                    "accuracy_first_loss_gap": loss_gap,
                    "mismatch": str(accuracy_first["strategy"]) != str(operational_best["strategy"]),
                }
            )
        if not group_records:
            continue
        detail = pd.DataFrame(group_records)
        rows.append(
            {
                "source_table": source_table,
                "dataset_name": first_non_null(table, "dataset_name"),
                "evaluation_group_count": int(len(detail)),
                "accuracy_first_not_operational_best_rate": float(detail["mismatch"].mean()),
                "accuracy_first_normalized_loss": float(detail["accuracy_first_normalized_loss"].mean()),
                "operational_best_normalized_loss": float(detail["operational_best_normalized_loss"].mean()),
                "mean_accuracy_first_loss_gap": float(detail["accuracy_first_loss_gap"].mean()),
                "median_accuracy_first_loss_gap": float(detail["accuracy_first_loss_gap"].median()),
                "accuracy_first_examples": join_top_values(detail["accuracy_first_method"]),
                "operational_best_examples": join_top_values(detail["operational_best_method"]),
            }
        )
    return with_overall_row(pd.DataFrame(rows), metric_columns=[
        "accuracy_first_not_operational_best_rate",
        "accuracy_first_normalized_loss",
        "operational_best_normalized_loss",
        "mean_accuracy_first_loss_gap",
        "median_accuracy_first_loss_gap",
    ])


def summarize_planning_execution_gap(method_tables: Sequence[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Summarize planning-execution gap rates by table."""
    rows = []
    for source_table, table in method_tables:
        group_records = []
        for group_key, group in iter_groups(table):
            deployable = group[group["deployable_inferred"]].copy()
            if len(deployable) < 2:
                continue
            if not contains_explicit_accuracy_first(deployable):
                continue
            accuracy_first = sort_accuracy_first(deployable).iloc[0]
            best_deployable = deployable.sort_values(["normalized_total_loss", "method_name", "strategy"]).iloc[0]
            accuracy_gap_rate = float(accuracy_first["execution_violation_rate"])
            best_gap_rate = float(best_deployable["execution_violation_rate"])
            relative_reduction = safe_relative_reduction(accuracy_gap_rate, best_gap_rate)
            group_records.append(
                {
                    "source_table": source_table,
                    "group_key": group_key,
                    "accuracy_first_method": accuracy_first["method_name"],
                    "best_deployable_method": best_deployable["method_name"],
                    "accuracy_first_planning_execution_gap_rate": accuracy_gap_rate,
                    "best_deployable_planning_execution_gap_rate": best_gap_rate,
                    "relative_gap_reduction": relative_reduction,
                }
            )
        if not group_records:
            continue
        detail = pd.DataFrame(group_records)
        rows.append(
            {
                "source_table": source_table,
                "dataset_name": first_non_null(table, "dataset_name"),
                "evaluation_group_count": int(len(detail)),
                "accuracy_first_planning_execution_gap_rate": float(detail["accuracy_first_planning_execution_gap_rate"].mean()),
                "best_deployable_planning_execution_gap_rate": float(detail["best_deployable_planning_execution_gap_rate"].mean()),
                "relative_gap_reduction": float(detail["relative_gap_reduction"].dropna().mean()) if detail["relative_gap_reduction"].notna().any() else np.nan,
                "accuracy_first_examples": join_top_values(detail["accuracy_first_method"]),
                "best_deployable_examples": join_top_values(detail["best_deployable_method"]),
            }
        )
    return with_overall_row(pd.DataFrame(rows), metric_columns=[
        "accuracy_first_planning_execution_gap_rate",
        "best_deployable_planning_execution_gap_rate",
        "relative_gap_reduction",
    ])


def summarize_greedy_vs_dp(method_tables: Sequence[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Summarize myopic greedy suboptimality relative to finite-horizon DP."""
    rows = []
    for source_table, table in method_tables:
        group_records = []
        for group_key, group in iter_groups(table):
            greedy = strategy_row(group, "greedy_feasibility_selector")
            dp = strategy_row(group, "dp_feasibility_selector")
            if greedy is None or dp is None:
                continue
            group_records.append(
                {
                    "source_table": source_table,
                    "group_key": group_key,
                    "greedy_to_dp_gap": diff_metric(greedy, dp, "normalized_total_loss"),
                    "greedy_to_dp_execution_penalty_gap": diff_metric(greedy, dp, "execution_penalty"),
                    "greedy_to_dp_switch_count_gap": diff_metric(greedy, dp, "model_switch_count"),
                    "greedy_to_dp_violation_rate_gap": diff_metric(greedy, dp, "execution_violation_rate"),
                }
            )
        if not group_records:
            continue
        detail = pd.DataFrame(group_records)
        rows.append(
            {
                "source_table": source_table,
                "dataset_name": first_non_null(table, "dataset_name"),
                "evaluation_group_count": int(len(detail)),
                "greedy_to_dp_gap": float(detail["greedy_to_dp_gap"].mean()),
                "median_greedy_to_dp_gap": float(detail["greedy_to_dp_gap"].median()),
                "greedy_to_dp_execution_penalty_gap": float(detail["greedy_to_dp_execution_penalty_gap"].mean()),
                "greedy_to_dp_switch_count_gap": float(detail["greedy_to_dp_switch_count_gap"].mean()),
                "greedy_to_dp_violation_rate_gap": float(detail["greedy_to_dp_violation_rate_gap"].mean()),
            }
        )
    return with_overall_row(pd.DataFrame(rows), metric_columns=[
        "greedy_to_dp_gap",
        "median_greedy_to_dp_gap",
        "greedy_to_dp_execution_penalty_gap",
        "greedy_to_dp_switch_count_gap",
        "greedy_to_dp_violation_rate_gap",
    ])


def summarize_integrated_thesis(
    accuracy_mismatch: pd.DataFrame,
    planning_gap: pd.DataFrame,
    greedy_vs_dp: pd.DataFrame,
    method_tables: Sequence[Tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Combine the thesis metrics into one paper-facing summary."""
    rows = []
    source_tables = sorted(
        set(accuracy_mismatch.get("source_table", pd.Series(dtype=str)).dropna())
        | set(planning_gap.get("source_table", pd.Series(dtype=str)).dropna())
        | set(greedy_vs_dp.get("source_table", pd.Series(dtype=str)).dropna())
    )
    source_tables = [source for source in source_tables if source != "overall"]
    oracle_gap_summary = summarize_oracle_gaps(method_tables)
    for source_table in source_tables:
        mismatch_row = select_source_row(accuracy_mismatch, source_table)
        gap_row = select_source_row(planning_gap, source_table)
        dp_row = select_source_row(greedy_vs_dp, source_table)
        oracle_row = select_source_row(oracle_gap_summary, source_table)
        rows.append(
            {
                "source_table": source_table,
                "dataset_name": first_available([mismatch_row, gap_row, dp_row, oracle_row], "dataset_name"),
                "evaluation_group_count": first_numeric_available(
                    [mismatch_row, gap_row, dp_row],
                    "evaluation_group_count",
                ),
                "accuracy_first_not_operational_best_rate": row_value(mismatch_row, "accuracy_first_not_operational_best_rate"),
                "mean_accuracy_first_loss_gap": row_value(mismatch_row, "mean_accuracy_first_loss_gap"),
                "accuracy_first_planning_execution_gap_rate": row_value(gap_row, "accuracy_first_planning_execution_gap_rate"),
                "best_deployable_planning_execution_gap_rate": row_value(gap_row, "best_deployable_planning_execution_gap_rate"),
                "relative_gap_reduction": row_value(gap_row, "relative_gap_reduction"),
                "greedy_to_dp_gap": row_value(dp_row, "greedy_to_dp_gap"),
                "mean_gap_to_dp_oracle": row_value(oracle_row, "mean_gap_to_dp_oracle"),
                "mean_gap_to_perfect_oracle": row_value(oracle_row, "mean_gap_to_perfect_oracle"),
            }
        )
    return with_overall_row(pd.DataFrame(rows), metric_columns=[
        "accuracy_first_not_operational_best_rate",
        "mean_accuracy_first_loss_gap",
        "accuracy_first_planning_execution_gap_rate",
        "best_deployable_planning_execution_gap_rate",
        "relative_gap_reduction",
        "greedy_to_dp_gap",
        "mean_gap_to_dp_oracle",
        "mean_gap_to_perfect_oracle",
    ])


def summarize_oracle_gaps(method_tables: Sequence[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Return mean oracle gaps by source table for deployable methods."""
    rows = []
    for source_table, table in method_tables:
        deployable = table[table["deployable_inferred"]].copy()
        record = {
            "source_table": source_table,
            "dataset_name": first_non_null(table, "dataset_name"),
            "mean_gap_to_dp_oracle": np.nan,
            "mean_gap_to_perfect_oracle": np.nan,
        }
        if "gap_to_dp_oracle" in deployable.columns:
            record["mean_gap_to_dp_oracle"] = numeric_mean(deployable["gap_to_dp_oracle"])
        if "gap_to_perfect_oracle" in deployable.columns:
            record["mean_gap_to_perfect_oracle"] = numeric_mean(deployable["gap_to_perfect_oracle"])
        rows.append(record)
    return pd.DataFrame(rows)


def iter_groups(table: pd.DataFrame) -> Iterable[Tuple[str, pd.DataFrame]]:
    """Yield comparison groups using all applicable grouping columns."""
    group_columns = [column for column in GROUP_COLUMN_CANDIDATES if column in table.columns]
    if not group_columns:
        yield "all", table
        return
    for key, group in table.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_parts = ["{}={}".format(column, value) for column, value in zip(group_columns, key)]
        yield "; ".join(key_parts), group


def strategy_row(group: pd.DataFrame, strategy_name: str) -> Optional[pd.Series]:
    """Return one strategy row from a group."""
    rows = group[group["strategy"].astype(str) == strategy_name]
    if rows.empty:
        return None
    return rows.sort_values(["method_name", "strategy"]).iloc[0]


def sort_accuracy_first(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort candidate rows by WAPE with accuracy-first tie-breaking."""
    sortable = frame.copy()
    sortable["_accuracy_first_priority"] = sortable["strategy"].map(accuracy_first_priority)
    return sortable.sort_values(["WAPE", "_accuracy_first_priority", "method_name", "strategy"]).drop(
        columns=["_accuracy_first_priority"]
    )


def accuracy_first_priority(strategy: object) -> int:
    """Return a low value for explicit accuracy-first strategies."""
    strategy_name = str(strategy)
    if strategy_name in {"global_best_model", "best_accuracy"}:
        return 0
    if strategy_name in {"individual_global_lightgbm", "individual_global_xgboost"}:
        return 1
    if strategy_name == "family_best_model":
        return 2
    return 3


def contains_explicit_accuracy_first(frame: pd.DataFrame) -> bool:
    """Return True if a comparison group contains an accuracy-first baseline."""
    strategies = set(frame["strategy"].astype(str))
    return bool(
        strategies.intersection(
            {
                "global_best_model",
                "best_accuracy",
                "individual_global_lightgbm",
                "individual_global_xgboost",
            }
        )
    )


def diff_metric(left: pd.Series, right: pd.Series, metric: str) -> float:
    """Return metric(left) - metric(right), or NaN if unavailable."""
    if metric not in left.index or metric not in right.index:
        return np.nan
    return float(pd.to_numeric(left[metric], errors="coerce") - pd.to_numeric(right[metric], errors="coerce"))


def safe_relative_reduction(reference: float, candidate: float) -> float:
    """Return relative reduction, guarding zero references."""
    if not np.isfinite(reference) or abs(reference) < 1e-12:
        return np.nan
    return float((reference - candidate) / reference)


def first_non_null(table: pd.DataFrame, column: str) -> str:
    """Return first non-null value from a column."""
    if column not in table.columns:
        return "unknown"
    values = table[column].dropna()
    if values.empty:
        return "unknown"
    return str(values.iloc[0])


def join_top_values(values: pd.Series, limit: int = 3) -> str:
    """Return common method names as a compact semicolon-separated string."""
    counts = values.astype(str).value_counts()
    return "; ".join(counts.head(limit).index.tolist())


def numeric_mean(values: pd.Series) -> float:
    """Return mean of numeric values, or NaN."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(numeric.mean())


def with_overall_row(table: pd.DataFrame, metric_columns: Sequence[str]) -> pd.DataFrame:
    """Append an overall mean row when table is non-empty."""
    if table.empty:
        return table
    if "evaluation_group_count" in table.columns:
        evaluation_group_count = int(pd.to_numeric(table["evaluation_group_count"], errors="coerce").fillna(0).sum())
    else:
        evaluation_group_count = int(len(table))
    overall: Dict[str, object] = {
        "source_table": "overall",
        "dataset_name": "all",
        "evaluation_group_count": evaluation_group_count,
    }
    weights = (
        pd.to_numeric(table["evaluation_group_count"], errors="coerce")
        if "evaluation_group_count" in table.columns
        else pd.Series(1.0, index=table.index)
    )
    for column in metric_columns:
        if column in table.columns:
            overall[column] = weighted_numeric_mean(table[column], weights)
    for column in table.columns:
        if column not in overall:
            overall[column] = "multiple" if table[column].dtype == object else np.nan
    result = pd.concat([pd.DataFrame([overall]), table], ignore_index=True)
    return result


def weighted_numeric_mean(values: pd.Series, weights: pd.Series) -> float:
    """Return weighted mean of numeric values, or NaN."""
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = numeric.notna() & numeric_weights.gt(0)
    if not mask.any():
        return numeric_mean(values)
    return float(np.average(numeric[mask], weights=numeric_weights[mask]))


def select_source_row(table: pd.DataFrame, source_table: str) -> Mapping[str, object]:
    """Return one row matching source table, or an empty mapping."""
    if table.empty or "source_table" not in table.columns:
        return {}
    rows = table[table["source_table"] == source_table]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def row_value(row: Mapping[str, object], column: str) -> float:
    """Return numeric row value, or NaN."""
    if not row or column not in row:
        return np.nan
    return float(pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0])


def first_available(rows: Sequence[Mapping[str, object]], column: str) -> str:
    """Return the first available non-empty string value."""
    for row in rows:
        if row and column in row and pd.notna(row[column]):
            return str(row[column])
    return "unknown"


def first_numeric_available(rows: Sequence[Mapping[str, object]], column: str) -> float:
    """Return the first finite numeric row value, or NaN."""
    for row in rows:
        if row and column in row:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return np.nan


def write_output_pair(
    data: pd.DataFrame,
    stem: str,
    output_dir: Path,
    paper_table_dir: Path,
    caption: str,
    label: str,
    max_rows: int,
) -> None:
    """Write CSV output and a compact LaTeX version."""
    csv_path = output_dir / "{}.csv".format(stem)
    data.to_csv(csv_path, index=False)
    paper_data = data.head(max_rows).copy()
    numeric_columns = paper_data.select_dtypes(include=["number"]).columns
    paper_data.loc[:, numeric_columns] = paper_data.loc[:, numeric_columns].round(3)
    paper_data = paper_data.where(pd.notna(paper_data), "")
    export_summary_table(
        data=paper_data,
        table_name="{}_table".format(stem),
        output_dir=paper_table_dir,
        caption=caption,
        label=label,
        numeric_precision=3,
        resize_to_textwidth=True,
    )


if __name__ == "__main__":
    main()
