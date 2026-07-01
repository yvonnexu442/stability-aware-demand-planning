"""Run Walmart business-context robustness checks for the paper."""

import argparse
import copy
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loaders.walmart_loader import load_walmart_modeling_table, validate_walmart_files
from evaluation.forecast_metrics import mean_absolute_error, weighted_absolute_percentage_error
from evaluation.planning_utility import add_normalized_planning_loss, add_oracle_gap_columns, planning_loss_weight
from reporting.latex_export import export_summary_table
from utils.config import load_config
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

from run_m5_robustness_pipeline import (
    build_decision_outputs as build_common_decision_outputs,
    evaluate_selected_decisions,
    load_dataco_execution_scenarios,
    short_strategy_label,
    summarize_planning_utility,
)


FEATURE_SETS = {
    "history_only": [
        "demand_lag_1",
        "demand_lag_4",
        "demand_lag_13",
        "demand_lag_26",
        "demand_rolling_mean_4",
        "demand_rolling_mean_13",
        "demand_rolling_std_4",
        "demand_rolling_std_13",
        "demand_ewm_alpha_0_3",
        "zero_demand_rate_13",
        "week_of_year",
        "month",
        "year",
    ],
    "history_plus_context": [
        "demand_lag_1",
        "demand_lag_4",
        "demand_lag_13",
        "demand_lag_26",
        "demand_rolling_mean_4",
        "demand_rolling_mean_13",
        "demand_rolling_std_4",
        "demand_rolling_std_13",
        "demand_ewm_alpha_0_3",
        "zero_demand_rate_13",
        "week_of_year",
        "month",
        "year",
        "is_holiday",
        "temperature",
        "fuel_price",
        "markdown_1",
        "markdown_2",
        "markdown_3",
        "markdown_4",
        "markdown_5",
        "markdown_total",
        "cpi",
        "unemployment",
        "store_size",
        "store_type",
    ],
}

BASELINE_FORECAST_FEATURES = {
    "naive_last_value": "demand_lag_1",
    "seasonal_4_week": "demand_lag_4",
    "moving_average_4": "demand_rolling_mean_4",
    "moving_average_13": "demand_rolling_mean_13",
    "exponential_smoothing": "demand_ewm_alpha_0_3",
}

WALMART_STRATEGY_ORDER = [
    "global_best_model",
    "simple_ensemble",
    "operational_loss_ensemble",
    "greedy_feasibility_selector",
    "dp_feasibility_selector",
    "budgeted_dp_feasibility_selector",
    "oracle_dp_feasibility_selector",
    "oracle_realized_demand",
]

SUMMARY_COLUMNS = [
    "dataset_name",
    "run_mode",
    "feature_set",
    "experiment_name",
    "window_type",
    "scenario_name",
    "method_name",
    "deployable",
    "oracle_type",
    "fallback_used",
    "fallback_type",
    "fallback_reason",
    "WAPE",
    "inventory_cost",
    "planning_volatility",
    "execution_penalty",
    "execution_violation_rate",
    "model_switch_count",
    "max_period_plan_change_pct",
    "normalized_inventory_component",
    "normalized_volatility_component",
    "normalized_execution_component",
    "normalized_switch_component",
    "normalized_total_loss",
    "gap_to_dp_oracle",
    "gap_to_perfect_oracle",
    "rank_by_WAPE",
    "rank_by_execution_penalty",
    "rank_by_normalized_total_loss",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Walmart business-context robustness checks.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--run-mode", choices=["quick", "medium", "full", "quick_mode", "medium_mode", "full_mode"], default="quick")
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--paper-table-dir", default="paper/tables")
    parser.add_argument("--paper-figure-dir", default="paper/figures")
    return parser.parse_args()


def main() -> None:
    """Run Walmart robustness checks and export paper-ready assets."""
    args = parse_args()
    config = load_config(args.config)
    run_mode = normalize_run_mode(args.run_mode)
    output_dir = Path(args.output_dir or config.get("project", {}).get("output_dir", "outputs"))
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    log_dir = output_dir / "logs"
    paper_table_dir = Path(args.paper_table_dir)
    paper_figure_dir = Path(args.paper_figure_dir)
    for path in [table_dir, figure_dir, log_dir, paper_table_dir, paper_figure_dir]:
        path.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("walmart_robustness_pipeline", log_file=log_dir / "walmart_robustness_pipeline.log")
    raw_data_dir = Path(args.raw_data_dir or config.get("walmart_pipeline", {}).get("raw_data_dir", "data/raw/walmart"))
    if run_mode == "full" and not bool(config.get("walmart_pipeline", {}).get("enable_full_mode", False)):
        logger.warning("Walmart full_mode is disabled by default. Set walmart_pipeline.enable_full_mode=true to run it.")
        return

    try:
        validate_walmart_files(raw_data_dir)
    except (FileNotFoundError, ValueError) as error:
        logger.warning("%s", error)
        write_missing_data_instructions(table_dir, raw_data_dir, str(error))
        return

    max_series = args.max_series if args.max_series is not None else resolve_walmart_max_series(run_mode, config)
    logger.info("Running Walmart robustness pipeline in %s mode with max_series=%s.", run_mode, max_series)
    modeling_table, quality_report = load_walmart_modeling_table(
        raw_data_dir=raw_data_dir,
        run_mode=run_mode,
        max_series=max_series,
        min_history_length=int(config.get("walmart_pipeline", {}).get("min_history_length", 80)),
        min_nonzero_observations=int(config.get("walmart_pipeline", {}).get("min_nonzero_observations", 20)),
        random_seed=int(config.get("project", {}).get("random_seed", 42)),
        output_table_dir=table_dir,
    )
    quality_report.to_csv(table_dir / "walmart_data_quality_report.csv", index=False)
    modeling_table = assign_walmart_splits(modeling_table, config)

    scenarios = load_dataco_execution_scenarios(config, table_dir, logger)
    feature_outputs = []
    all_decisions = []
    all_forecasts = []
    forecast_metric_outputs = []
    for feature_set in FEATURE_SETS:
        forecasts = build_walmart_candidate_forecasts(modeling_table, feature_set, config)
        forecast_metrics = summarize_forecast_metrics(forecasts)
        decisions = build_walmart_decision_outputs(forecasts, modeling_table, forecast_metrics, config)
        forecasts["feature_set"] = feature_set
        forecast_metrics["feature_set"] = feature_set
        decisions["feature_set"] = feature_set
        all_forecasts.append(forecasts)
        forecast_metric_outputs.append(forecast_metrics)
        all_decisions.append(decisions)

        for _, scenario in scenarios.iterrows():
            feature_outputs.append(
                summarize_walmart_decisions(
                    decisions=decisions,
                    config=config,
                    run_mode=run_mode,
                    feature_set=feature_set,
                    experiment_name="context_robustness",
                    window_type="all",
                    scenario_name=str(scenario["scenario_name"]),
                    lambda_execution=float(scenario["lambda_execution"]),
                )
            )

    context_summary = pd.concat(feature_outputs, ignore_index=True)
    decisions_all = pd.concat(all_decisions, ignore_index=True)
    forecasts_all = pd.concat(all_forecasts, ignore_index=True)
    forecast_metrics_all = pd.concat(forecast_metric_outputs, ignore_index=True)

    context_comparison = build_context_comparison(context_summary)
    window_stress = run_holiday_markdown_window_stress(decisions_all, modeling_table, config, run_mode)
    cadence_constraints = run_weekly_cadence_constraints(forecasts_all, decisions_all, modeling_table, config, run_mode)
    robustness_summary = pd.concat(
        [
            context_summary.assign(module_name="context_robustness"),
            window_stress.assign(module_name="holiday_markdown_stress"),
            cadence_constraints.assign(module_name="weekly_cadence_constraints"),
        ],
        ignore_index=True,
        sort=False,
    )

    write_walmart_outputs(
        context_summary=context_summary,
        context_comparison=context_comparison,
        window_stress=window_stress,
        cadence_constraints=cadence_constraints,
        robustness_summary=robustness_summary,
        forecast_metrics=forecast_metrics_all,
        table_dir=table_dir,
        paper_table_dir=paper_table_dir,
    )
    write_walmart_figures(
        context_summary=context_summary,
        context_comparison=context_comparison,
        window_stress=window_stress,
        cadence_constraints=cadence_constraints,
        figure_dir=figure_dir,
        paper_figure_dir=paper_figure_dir,
    )
    logger.info("Walmart robustness pipeline completed successfully.")


def normalize_run_mode(run_mode: str) -> str:
    """Normalize run-mode aliases."""
    value = str(run_mode).strip().lower()
    aliases = {"quick_mode": "quick", "medium_mode": "medium", "full_mode": "full"}
    return aliases.get(value, value)


def resolve_walmart_max_series(run_mode: str, config: Mapping[str, object]) -> Optional[int]:
    """Return the configured Walmart sample size."""
    walmart_config = config.get("walmart_pipeline", {})
    if run_mode == "quick":
        return int(walmart_config.get("quick_mode_max_series", 200))
    if run_mode == "medium":
        return int(walmart_config.get("medium_mode_max_series", 750))
    value = walmart_config.get("full_mode_max_series")
    return None if value is None else int(value)


def write_missing_data_instructions(table_dir: Path, raw_data_dir: Path, reason: str) -> None:
    """Write a missing-data audit table instead of failing the repository workflow."""
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "dataset_name": "walmart",
                "status": "missing_required_data",
                "raw_data_dir": str(raw_data_dir),
                "reason": reason,
                "required_files": "train.csv, features.csv, stores.csv",
            }
        ]
    ).to_csv(table_dir / "walmart_missing_data_instructions.csv", index=False)


def assign_walmart_splits(modeling_table: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    """Assign chronological weekly train, validation, and test splits."""
    frame = modeling_table.copy()
    validation_horizon = int(config.get("walmart_pipeline", {}).get("validation_horizon", 13))
    test_horizon = int(config.get("walmart_pipeline", {}).get("test_horizon", 13))
    dates = sorted(pd.to_datetime(frame["date"]).unique())
    if len(dates) <= validation_horizon + test_horizon:
        raise ValueError("Walmart modeling table does not contain enough weekly dates for validation and test splits.")
    validation_dates = set(dates[-(validation_horizon + test_horizon) : -test_horizon])
    test_dates = set(dates[-test_horizon:])
    frame["split"] = "train"
    frame.loc[frame["date"].isin(validation_dates), "split"] = "validation"
    frame.loc[frame["date"].isin(test_dates), "split"] = "test"
    frame["horizon"] = frame.groupby(["series_id", "split"]).cumcount() + 1
    return frame


def build_walmart_candidate_forecasts(
    modeling_table: pd.DataFrame,
    feature_set: str,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Build transparent weekly forecast candidates for validation and test rows."""
    prediction_frame = modeling_table[modeling_table["split"].isin(["validation", "test"])].copy()
    rows = []
    base_columns = [
        "date",
        "series_id",
        "split",
        "horizon",
        "demand",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "demand_rolling_std_28",
        "zero_demand_rate_28",
    ]
    for model_name, feature_column in BASELINE_FORECAST_FEATURES.items():
        frame = prediction_frame[base_columns].copy()
        frame["model_name"] = model_name
        frame["forecast"] = pd.to_numeric(prediction_frame[feature_column], errors="coerce").fillna(0.0).clip(lower=0.0)
        frame = frame.rename(columns={"demand": "actual"})
        rows.append(frame)

    ml_predictions = build_global_sklearn_forecast(modeling_table, feature_set, config)
    if not ml_predictions.empty:
        rows.append(ml_predictions)
    forecasts = pd.concat(rows, ignore_index=True)
    forecasts["date"] = pd.to_datetime(forecasts["date"])
    return forecasts.sort_values(["split", "series_id", "date", "model_name"]).reset_index(drop=True)


def build_global_sklearn_forecast(
    modeling_table: pd.DataFrame,
    feature_set: str,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Fit a simple global sklearn regressor using the selected feature set."""
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        return pd.DataFrame()

    feature_columns = [column for column in FEATURE_SETS[feature_set] if column in modeling_table.columns]
    if not feature_columns:
        return pd.DataFrame()

    train_frame = modeling_table[modeling_table["split"] == "train"].copy()
    prediction_frame = modeling_table[modeling_table["split"].isin(["validation", "test"])].copy()
    max_rows = int(config.get("walmart_pipeline", {}).get("max_ml_training_rows", 150000))
    if len(train_frame) > max_rows:
        train_frame = train_frame.sample(n=max_rows, random_state=int(config.get("project", {}).get("random_seed", 42)))

    train_features, prediction_features = _prepare_feature_matrices(train_frame, prediction_frame, feature_columns)
    target = pd.to_numeric(train_frame["demand"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if train_features.empty or prediction_features.empty:
        return pd.DataFrame()

    model = RandomForestRegressor(
        n_estimators=80,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=int(config.get("project", {}).get("random_seed", 42)),
    )
    model.fit(train_features, target)
    forecast = np.maximum(model.predict(prediction_features), 0.0)
    output = prediction_frame[
        [
            "date",
            "series_id",
            "split",
            "horizon",
            "demand",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "demand_rolling_std_28",
            "zero_demand_rate_28",
        ]
    ].copy()
    output["model_name"] = "global_sklearn"
    output["forecast"] = forecast
    return output.rename(columns={"demand": "actual"})


def _prepare_feature_matrices(
    train_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return aligned train and prediction matrices without future outcomes."""
    train_features = train_frame[list(feature_columns)].copy()
    prediction_features = prediction_frame[list(feature_columns)].copy()
    combined = pd.concat([train_features, prediction_features], keys=["train", "prediction"], sort=False)
    categorical_columns = [column for column in combined.columns if combined[column].dtype == object]
    combined = pd.get_dummies(combined, columns=categorical_columns, dummy_na=True)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    train_matrix = combined.loc["train"].copy()
    prediction_matrix = combined.loc["prediction"].copy()
    medians = train_matrix.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_matrix = train_matrix.fillna(medians).fillna(0.0)
    prediction_matrix = prediction_matrix.fillna(medians).fillna(0.0)
    return train_matrix, prediction_matrix


def summarize_forecast_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return validation and test accuracy metrics by model."""
    records = []
    for (split, model_name), group in forecasts.groupby(["split", "model_name"]):
        records.append(
            {
                "split": split,
                "model_name": model_name,
                "row_count": len(group),
                "WAPE": weighted_absolute_percentage_error(group["actual"], group["forecast"]),
                "MAE": mean_absolute_error(group["actual"], group["forecast"]),
            }
        )
    return pd.DataFrame(records).sort_values(["split", "WAPE", "model_name"]).reset_index(drop=True)


def build_walmart_decision_outputs(
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Reuse common feasibility selectors for Walmart forecast candidates."""
    walmart_config = config.get("walmart_pipeline", {})
    scenario_config = copy.deepcopy(dict(config))
    scenario_config.setdefault("m5_pipeline", {})
    scenario_config["m5_pipeline"]["safety_stock_multiplier"] = float(walmart_config.get("safety_stock_multiplier", 0.5))
    decisions = build_common_decision_outputs(forecasts, modeling_table, forecast_metrics, scenario_config)
    decisions = decisions[decisions["strategy"].isin(WALMART_STRATEGY_ORDER)].copy()
    return decisions.sort_values(["strategy", "series_id", "date"]).reset_index(drop=True)


def summarize_walmart_decisions(
    decisions: pd.DataFrame,
    config: Mapping[str, object],
    run_mode: str,
    feature_set: str,
    experiment_name: str,
    window_type: str,
    scenario_name: str,
    lambda_execution: Optional[float] = None,
) -> pd.DataFrame:
    """Return a normalized Walmart strategy summary for one experiment slice."""
    scenario_config = copy.deepcopy(dict(config))
    weights = dict(scenario_config.get("planning_loss_weights", {}))
    if lambda_execution is not None:
        weights["lambda_execution"] = float(lambda_execution)
    scenario_config["planning_loss_weights"] = weights
    summary = summarize_planning_utility(decisions, weights)
    summary, _, _ = add_normalized_planning_loss(
        summary,
        weights,
        reference_strategy="global_best_model",
        dataset_name="walmart",
        run_mode=run_mode,
        split_name="test",
    )
    return finalize_walmart_summary(
        summary,
        dataset_name="walmart",
        run_mode=run_mode,
        feature_set=feature_set,
        experiment_name=experiment_name,
        window_type=window_type,
        scenario_name=scenario_name,
    )


def finalize_walmart_summary(
    summary: pd.DataFrame,
    dataset_name: str,
    run_mode: str,
    feature_set: str,
    experiment_name: str,
    window_type: str,
    scenario_name: str,
) -> pd.DataFrame:
    """Return paper-facing Walmart summary columns."""
    table = summary.copy()
    table["dataset_name"] = dataset_name
    table["run_mode"] = run_mode
    table["feature_set"] = feature_set
    table["experiment_name"] = experiment_name
    table["window_type"] = window_type
    table["scenario_name"] = scenario_name
    table["method_name"] = table["strategy"].map(short_strategy_label)
    table["WAPE"] = table["weighted_absolute_percentage_error"]
    table["inventory_cost"] = table["total_inventory_cost"]
    table["planning_volatility"] = table["planning_signal_volatility_total"]
    table["execution_penalty"] = table["execution_adaptation_penalty_total"]
    table = add_oracle_gap_columns(
        table,
        group_columns=[
            "dataset_name",
            "run_mode",
            "feature_set",
            "experiment_name",
            "window_type",
            "scenario_name",
        ],
    )
    table["rank_by_WAPE"] = table["WAPE"].rank(method="min").astype(int)
    table["rank_by_execution_penalty"] = table["execution_penalty"].rank(method="min").astype(int)
    table["rank_by_normalized_total_loss"] = table["normalized_total_loss"].rank(method="min").astype(int)
    return table[SUMMARY_COLUMNS + ["strategy"]].sort_values(
        ["feature_set", "experiment_name", "window_type", "scenario_name", "rank_by_normalized_total_loss", "method_name"]
    ).reset_index(drop=True)


def build_context_comparison(context_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare context-aware and context-free results within matching scenarios."""
    metrics = [
        "WAPE",
        "inventory_cost",
        "planning_volatility",
        "execution_penalty",
        "execution_violation_rate",
        "model_switch_count",
        "normalized_total_loss",
        "gap_to_dp_oracle",
        "gap_to_perfect_oracle",
    ]
    key_columns = ["dataset_name", "run_mode", "scenario_name", "strategy", "method_name"]
    subset = context_summary[context_summary["window_type"] == "all"].copy()
    pivot = subset.pivot_table(index=key_columns, columns="feature_set", values=metrics, aggfunc="first")
    rows = []
    for key, values in pivot.iterrows():
        record = dict(zip(key_columns, key if isinstance(key, tuple) else (key,)))
        for metric in metrics:
            history_value = values.get((metric, "history_only"), np.nan)
            context_value = values.get((metric, "history_plus_context"), np.nan)
            record["history_only_{}".format(metric)] = history_value
            record["history_plus_context_{}".format(metric)] = context_value
            record["context_minus_history_{}".format(metric)] = context_value - history_value
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["scenario_name", "method_name"]).reset_index(drop=True)


def run_holiday_markdown_window_stress(
    decisions: pd.DataFrame,
    modeling_table: pd.DataFrame,
    config: Mapping[str, object],
    run_mode: str,
) -> pd.DataFrame:
    """Summarize strategy behavior in holiday and markdown-heavy windows."""
    lookup = build_window_lookup(modeling_table, config)
    frame = decisions.merge(lookup, on=["series_id", "date"], how="left")
    rows = []
    for feature_set, feature_frame in frame.groupby("feature_set"):
        for window_type in ["normal", "holiday", "markdown_heavy", "holiday_or_markdown"]:
            mask = feature_frame[window_type].fillna(False).astype(bool)
            window_frame = feature_frame[mask].copy()
            if window_frame.empty:
                continue
            rows.append(
                summarize_walmart_decisions(
                    decisions=window_frame,
                    config=config,
                    run_mode=run_mode,
                    feature_set=str(feature_set),
                    experiment_name="holiday_markdown_stress",
                    window_type=window_type,
                    scenario_name="baseline",
                    lambda_execution=planning_loss_weight(config.get("planning_loss_weights", {}), "lambda_execution"),
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=SUMMARY_COLUMNS + ["strategy"])


def build_window_lookup(modeling_table: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    """Return normal, holiday, and markdown-heavy row flags."""
    walmart_config = config.get("walmart_pipeline", {})
    test_rows = modeling_table[modeling_table["split"] == "test"].copy()
    markdown_total = pd.to_numeric(test_rows["markdown_total"], errors="coerce").fillna(0.0)
    positive_markdowns = markdown_total[markdown_total > 0.0]
    if positive_markdowns.empty:
        threshold = np.inf
    else:
        threshold = float(positive_markdowns.quantile(float(walmart_config.get("markdown_heavy_percentile", 0.75))))
    test_rows["holiday"] = pd.to_numeric(test_rows["is_holiday"], errors="coerce").fillna(0).astype(int) == 1
    test_rows["markdown_heavy"] = markdown_total >= threshold
    test_rows["holiday_or_markdown"] = test_rows["holiday"] | test_rows["markdown_heavy"]
    test_rows["normal"] = ~test_rows["holiday_or_markdown"]
    return test_rows[["series_id", "date", "normal", "holiday", "markdown_heavy", "holiday_or_markdown"]].drop_duplicates()


def run_weekly_cadence_constraints(
    forecasts: pd.DataFrame,
    decisions: pd.DataFrame,
    modeling_table: pd.DataFrame,
    config: Mapping[str, object],
    run_mode: str,
) -> pd.DataFrame:
    """Evaluate weekly cadence constraints for switchable deployable selectors."""
    walmart_config = config.get("walmart_pipeline", {})
    cadence = walmart_config.get("weekly_cadence", {})
    constraint_config = copy.deepcopy(dict(config))
    constraint_config.setdefault("stability", {})["max_plan_change_rate"] = float(cadence.get("max_weekly_plan_change_rate", 0.20))
    constraint_config.setdefault("feasibility_analysis", {}).setdefault("dp_selector", {})["max_switches"] = int(cadence.get("max_switches_per_quarter", 2))

    adjusted_frames = []
    for feature_set, feature_decisions in decisions.groupby("feature_set"):
        feature_forecasts = forecasts[forecasts["feature_set"] == feature_set].copy()
        adjusted = apply_holiday_freeze(
            decisions=feature_decisions.copy(),
            forecasts=feature_forecasts,
            modeling_table=modeling_table,
            freeze_weeks=int(cadence.get("holiday_freeze_weeks", 1)),
        )
        adjusted = evaluate_selected_decisions(adjusted, constraint_config)
        adjusted["feature_set"] = feature_set
        adjusted_frames.append(adjusted)

    if not adjusted_frames:
        return pd.DataFrame(columns=SUMMARY_COLUMNS + ["strategy"])

    adjusted_decisions = pd.concat(adjusted_frames, ignore_index=True)
    rows = []
    for feature_set, feature_frame in adjusted_decisions.groupby("feature_set"):
        rows.append(
            summarize_walmart_decisions(
                decisions=feature_frame,
                config=constraint_config,
                run_mode=run_mode,
                feature_set=str(feature_set),
                experiment_name="weekly_cadence_constraints",
                window_type="all",
                scenario_name="weekly_cadence",
                lambda_execution=planning_loss_weight(constraint_config.get("planning_loss_weights", {}), "lambda_execution"),
            )
        )
    return pd.concat(rows, ignore_index=True)


def apply_holiday_freeze(
    decisions: pd.DataFrame,
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    freeze_weeks: int,
) -> pd.DataFrame:
    """Hold the incumbent selected model in a short pre-holiday freeze window."""
    if int(freeze_weeks) <= 0:
        return _strip_evaluation_columns(decisions)

    switchable_strategies = {
        "greedy_feasibility_selector",
        "dp_feasibility_selector",
        "budgeted_dp_feasibility_selector",
    }
    base = _strip_evaluation_columns(decisions)
    holiday_dates = (
        modeling_table[(modeling_table["split"] == "test") & (pd.to_numeric(modeling_table["is_holiday"], errors="coerce").fillna(0).astype(int) == 1)]
        .groupby("series_id")["date"]
        .apply(lambda values: sorted(pd.to_datetime(values).unique()))
        .to_dict()
    )
    candidate_lookup = {
        (row.series_id, pd.Timestamp(row.date), row.model_name): row
        for row in forecasts[forecasts["split"] == "test"].itertuples(index=False)
    }

    adjusted_rows = []
    for (_, _), group in base.sort_values(["strategy", "series_id", "date"]).groupby(["strategy", "series_id"], sort=False):
        previous_model = None
        for row in group.itertuples(index=False):
            row_dict = row._asdict()
            strategy = str(row_dict["strategy"])
            series_id = row_dict["series_id"]
            current_date = pd.Timestamp(row_dict["date"])
            frozen = strategy in switchable_strategies and previous_model is not None and _within_holiday_freeze(
                current_date,
                holiday_dates.get(series_id, []),
                freeze_weeks=freeze_weeks,
            )
            if frozen:
                candidate = candidate_lookup.get((series_id, current_date, previous_model))
                if candidate is not None:
                    candidate_dict = candidate._asdict()
                    row_dict["model_name"] = candidate_dict["model_name"]
                    row_dict["selected_model"] = candidate_dict["model_name"]
                    row_dict["forecast"] = candidate_dict["forecast"]
                    row_dict["fallback_used"] = bool(row_dict.get("fallback_used", False))
                    row_dict["fallback_type"] = str(row_dict.get("fallback_type", "none"))
                    row_dict["fallback_reason"] = str(row_dict.get("fallback_reason", "none"))
            previous_model = str(row_dict["selected_model"])
            adjusted_rows.append(row_dict)
    return pd.DataFrame(adjusted_rows)


def _within_holiday_freeze(current_date: pd.Timestamp, holiday_dates: Sequence[pd.Timestamp], freeze_weeks: int) -> bool:
    """Return whether current_date falls inside the pre-holiday freeze window."""
    for holiday_date in holiday_dates:
        holiday = pd.Timestamp(holiday_date)
        lower = holiday - pd.Timedelta(days=7 * int(freeze_weeks))
        if lower <= current_date <= holiday:
            return True
    return False


def _strip_evaluation_columns(decisions: pd.DataFrame) -> pd.DataFrame:
    """Remove evaluation columns before re-running the evaluation layer."""
    drop_columns = [
        "planning_signal",
        "inventory_target",
        "holding_cost",
        "shortage_cost",
        "total_inventory_cost",
        "service_level_hit",
        "absolute_plan_change",
        "plan_change_pct",
        "model_switch_flag",
        "execution_adaptation_penalty",
        "execution_violation",
        "total_planning_loss",
    ]
    return decisions.drop(columns=[column for column in drop_columns if column in decisions.columns]).copy()


def write_walmart_outputs(
    context_summary: pd.DataFrame,
    context_comparison: pd.DataFrame,
    window_stress: pd.DataFrame,
    cadence_constraints: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    table_dir: Path,
    paper_table_dir: Path,
) -> None:
    """Write Walmart CSV and LaTeX table outputs."""
    outputs = {
        "walmart_context_robustness_summary": context_summary,
        "walmart_context_aware_vs_context_free_comparison": context_comparison,
        "walmart_holiday_markdown_stress_by_window": window_stress,
        "walmart_weekly_cadence_constraints": cadence_constraints,
        "walmart_robustness_summary": robustness_summary,
        "walmart_forecast_metrics": forecast_metrics,
    }
    for stem, data in outputs.items():
        data.to_csv(table_dir / "{}.csv".format(stem), index=False)
        export_summary_table(
            data=data,
            table_name="{}_table".format(stem),
            output_dir=paper_table_dir,
            caption=walmart_table_caption(stem),
            label="tab:{}".format(stem.replace("_", "-")),
            numeric_precision=3,
            resize_to_textwidth=True,
        )


def walmart_table_caption(stem: str) -> str:
    """Return a concise English caption for a Walmart table."""
    captions = {
        "walmart_context_robustness_summary": "Walmart context-robustness summary by feature set and execution scenario.",
        "walmart_context_aware_vs_context_free_comparison": "Walmart context-aware versus context-free method comparison.",
        "walmart_holiday_markdown_stress_by_window": "Walmart holiday and markdown-heavy stress-window summary.",
        "walmart_weekly_cadence_constraints": "Walmart weekly cadence constraint summary.",
        "walmart_robustness_summary": "Walmart robustness summary across context, stress windows, and cadence constraints.",
        "walmart_forecast_metrics": "Walmart forecast accuracy diagnostics by feature set and candidate model.",
    }
    return captions.get(stem, stem.replace("_", " ").title())


def write_walmart_figures(
    context_summary: pd.DataFrame,
    context_comparison: pd.DataFrame,
    window_stress: pd.DataFrame,
    cadence_constraints: pd.DataFrame,
    figure_dir: Path,
    paper_figure_dir: Path,
) -> None:
    """Write Walmart figures as PNG for review and PDF for LaTeX."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    paper_figure_dir.mkdir(parents=True, exist_ok=True)
    save_context_tradeoff(
        context_summary,
        png_path=figure_dir / "walmart_context_aware_vs_context_free_tradeoff.png",
        pdf_path=paper_figure_dir / "walmart_context_aware_vs_context_free_tradeoff.pdf",
    )
    save_ranking_shift(
        context_summary,
        png_path=figure_dir / "walmart_context_ranking_shift.png",
        pdf_path=paper_figure_dir / "walmart_context_ranking_shift.pdf",
    )
    save_window_metric(
        window_stress,
        metric="normalized_total_loss",
        y_label="Normalized Total Loss",
        png_path=figure_dir / "walmart_window_type_normalized_loss.png",
        pdf_path=paper_figure_dir / "walmart_window_type_normalized_loss.pdf",
    )
    save_window_metric(
        window_stress,
        metric="execution_penalty",
        y_label="Execution Penalty",
        png_path=figure_dir / "walmart_window_type_execution_penalty.png",
        pdf_path=paper_figure_dir / "walmart_window_type_execution_penalty.pdf",
    )
    save_cadence_metric(
        cadence_constraints,
        metric="normalized_total_loss",
        y_label="Normalized Total Loss",
        png_path=figure_dir / "walmart_weekly_cadence_constraints.png",
        pdf_path=paper_figure_dir / "walmart_weekly_cadence_constraints.pdf",
    )
    save_cadence_metric(
        cadence_constraints,
        metric="model_switch_count",
        y_label="Model Switch Count",
        png_path=figure_dir / "walmart_budgeted_dp_switching_behavior.png",
        pdf_path=paper_figure_dir / "walmart_budgeted_dp_switching_behavior.pdf",
    )


def save_context_tradeoff(data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save accuracy-versus-execution tradeoff for Walmart feature sets."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.35, 4.45))
    plot_data = data[(data["experiment_name"] == "context_robustness") & (data["scenario_name"] == "baseline")].dropna(
        subset=["WAPE", "execution_penalty"]
    )
    for index, row in enumerate(plot_data.itertuples(index=False)):
        marker = "o" if row.feature_set == "history_only" else "s"
        ax.scatter(
            row.WAPE,
            row.execution_penalty,
            marker=marker,
            color=strategy_color(row.strategy, index),
            edgecolor="white",
            linewidth=0.6,
            s=70,
            zorder=3,
        )
        ax.annotate(
            "{}\n{}".format(row.method_name, row.feature_set.replace("_", " ").title()),
            (row.WAPE, row.execution_penalty),
            xytext=(5, 5 if index % 2 == 0 else -9),
            textcoords="offset points",
            fontsize=6.8,
        )
    format_axis(ax, x_label="Weighted Absolute Percentage Error", y_label="Execution Penalty", grid_axis="both")
    save_paper_figure(fig, png_path, pdf_path)


def save_ranking_shift(data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save rank changes between history-only and context-aware feature sets."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.35, 4.45))
    plot_data = data[(data["experiment_name"] == "context_robustness") & (data["scenario_name"] == "baseline")].copy()
    x_order = ["history_only", "history_plus_context"]
    positions = {value: index for index, value in enumerate(x_order)}
    for index, strategy in enumerate(WALMART_STRATEGY_ORDER):
        strategy_data = plot_data[plot_data["strategy"] == strategy].copy()
        if strategy_data.empty:
            continue
        strategy_data["x_position"] = strategy_data["feature_set"].map(positions)
        strategy_data = strategy_data.sort_values("x_position")
        ax.plot(
            strategy_data["x_position"],
            strategy_data["rank_by_normalized_total_loss"],
            marker=strategy_marker(strategy),
            color=strategy_color(strategy, index),
            linestyle=strategy_linestyle(strategy),
            linewidth=1.7,
            label=short_strategy_label(strategy),
        )
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(["History Only", "History Plus Context"])
    ax.invert_yaxis()
    format_axis(ax, x_label="Feature Set", y_label="Rank by Normalized Total Loss")
    place_legend(ax, columns=3)
    save_paper_figure(fig, png_path, pdf_path)


def save_window_metric(data: pd.DataFrame, metric: str, y_label: str, png_path: Path, pdf_path: Path) -> None:
    """Save Walmart stress-window line figure by method."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.35, 4.45))
    plot_data = data[data["feature_set"] == "history_plus_context"].copy()
    x_order = ["normal", "holiday", "markdown_heavy", "holiday_or_markdown"]
    positions = {value: index for index, value in enumerate(x_order)}
    for index, strategy in enumerate(WALMART_STRATEGY_ORDER):
        strategy_data = plot_data[plot_data["strategy"] == strategy].copy()
        if strategy_data.empty:
            continue
        strategy_data["x_position"] = strategy_data["window_type"].map(positions)
        strategy_data = strategy_data.sort_values("x_position")
        ax.plot(
            strategy_data["x_position"],
            strategy_data[metric],
            marker=strategy_marker(strategy),
            color=strategy_color(strategy, index),
            linestyle=strategy_linestyle(strategy),
            linewidth=1.7,
            label=short_strategy_label(strategy),
        )
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(["Normal", "Holiday", "Markdown\nHeavy", "Holiday or\nMarkdown"])
    format_axis(ax, x_label="Window Type", y_label=y_label)
    place_legend(ax, columns=3)
    save_paper_figure(fig, png_path, pdf_path)


def save_cadence_metric(data: pd.DataFrame, metric: str, y_label: str, png_path: Path, pdf_path: Path) -> None:
    """Save Walmart weekly-cadence metric bar figure."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.35, 4.45))
    plot_data = data[data["feature_set"] == "history_plus_context"].copy()
    plot_data = plot_data.sort_values(metric)
    labels = plot_data["method_name"].astype(str).tolist()
    colors = [strategy_color(strategy, index) for index, strategy in enumerate(plot_data["strategy"].astype(str))]
    ax.barh(range(len(plot_data)), plot_data[metric], color=colors, alpha=0.88)
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(labels)
    format_axis(ax, x_label=y_label, y_label=None, grid_axis="x")
    save_paper_figure(fig, png_path, pdf_path)


if __name__ == "__main__":
    main()
