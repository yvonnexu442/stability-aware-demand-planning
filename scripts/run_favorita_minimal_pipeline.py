"""Run the minimal real-data Favorita planning-stability pipeline."""

import argparse
import copy
import logging
import os
import sys
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except ImportError:
    sns = None

from data_loaders.favorita_loader import load_favorita_modeling_table
from data_loaders.dataco_loader import load_dataco_orders
from decision_layer.feasibility_dp_selector import (
    BudgetedDPFeasibilitySelector,
    DPFeasibilitySelector,
    GreedyFeasibilitySelector,
)
from decision_layer.no_leakage import attach_actuals_for_evaluation, drop_future_outcomes, require_no_future_outcomes
from evaluation.forecast_metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)
from evaluation.inventory_metrics import compute_holding_cost, compute_service_level, compute_shortage_cost
from evaluation.planning_utility import add_normalized_planning_loss, compute_total_planning_loss
from evaluation.stability_metrics import compute_absolute_plan_change, compute_percentage_plan_change
from planning_environment.execution_capacity import compute_execution_capacity, compute_execution_violation
from planning_environment.planning_actions import forecast_to_inventory_target
from reporting.latex_export import export_summary_table, write_asset_manifest
from utils.logging_utils import setup_logger
from visualization.plots import (
    apply_paper_style,
    format_axis,
    palette_for_strategies,
    place_legend,
    save_paper_figure,
    strategy_color,
    strategy_linestyle,
    strategy_marker,
)


MODEL_FEATURE_MAP = {
    "naive_last_value": "demand_lag_1",
    "seasonal_naive": "demand_lag_7",
    "moving_average": "demand_rolling_mean_28",
    "exponential_smoothing": "demand_ewm_alpha_0_3",
}

PAPER_STRATEGY_ORDER = [
    "global_best_model",
    "family_best_model",
    "feasibility_aware_selector",
    "greedy_feasibility_selector",
    "dp_feasibility_selector",
    "budgeted_dp_feasibility_selector",
    "feasibility_aware_smoothed_utility_alpha",
    "feasibility_aware_ensemble_constrained",
    "stability_aware_selector",
    "simple_ensemble",
    "best_inventory_cost_model",
    "best_stability_model",
    "individual_global_lightgbm",
    "individual_global_xgboost",
    "individual_moving_average",
    "oracle_realized_demand",
]

BASELINE_COMPARISON_STRATEGIES = [
    "global_best_model",
    "family_best_model",
    "feasibility_aware_selector",
    "greedy_feasibility_selector",
    "dp_feasibility_selector",
    "budgeted_dp_feasibility_selector",
    "stability_aware_selector",
    "simple_ensemble",
    "best_inventory_cost_model",
    "best_stability_model",
    "oracle_realized_demand",
]

IMPROVED_METHOD_COMPARISON_STRATEGIES = [
    "global_best_model",
    "family_best_model",
    "simple_ensemble",
    "stability_aware_selector",
    "feasibility_aware_selector",
    "greedy_feasibility_selector",
    "dp_feasibility_selector",
    "budgeted_dp_feasibility_selector",
    "feasibility_aware_smoothed_alpha_0_25",
    "feasibility_aware_smoothed_alpha_0_50",
    "feasibility_aware_smoothed_alpha_0_75",
    "feasibility_aware_smoothed_scenario_alpha",
    "feasibility_aware_smoothed_utility_alpha",
    "feasibility_aware_ensemble_inverse_accuracy",
    "feasibility_aware_ensemble_inverse_operational_loss",
    "feasibility_aware_ensemble_constrained",
    "best_inventory_cost_model",
    "best_stability_model",
    "oracle_realized_demand",
]

RANKING_OBJECTIVES = [
    ("WAPE", "WAPE", True),
    ("inventory_cost", "inventory_cost", True),
    ("planning_volatility", "planning_volatility", True),
    ("execution_penalty", "execution_penalty", True),
    ("execution_violation_rate", "execution_violation_rate", True),
    ("normalized_total_loss", "normalized_total_loss", True),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Favorita pipeline."""
    parser = argparse.ArgumentParser(description="Run the minimal Favorita planning-stability pipeline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the experiment config file.")
    parser.add_argument("--raw-data-dir", default=None, help="Directory containing local Favorita CSV files.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated experiment outputs.")
    parser.add_argument(
        "--run-mode",
        choices=["quick", "medium", "full", "quick_mode", "medium_mode", "full_mode"],
        default=None,
        help="Override the configured run mode.",
    )
    parser.add_argument("--max-series", type=int, default=None, help="Override the number of store-family series to use.")
    parser.add_argument("--max-ml-training-rows", type=int, default=None, help="Override the global ML training row cap.")
    parser.add_argument("--paper-table-dir", default="paper/tables", help="Directory for LaTeX-ready paper tables.")
    parser.add_argument("--paper-figure-dir", default="paper/figures", help="Directory for LaTeX-ready paper figures.")
    parser.add_argument("--asset-manifest-path", default="paper/asset_manifest.md", help="Path for the paper asset manifest.")
    parser.add_argument("--skip-ml", action="store_true", help="Skip the global machine-learning forecast candidate.")
    parser.add_argument(
        "--reuse-forecast-table",
        default=None,
        help="Reuse an existing standardized forecast CSV instead of rebuilding forecast candidates.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Favorita loading, forecasting, planning evaluation, and paper export."""
    from utils.config import load_config, save_config_snapshot

    args = parse_args()
    config = load_config(args.config)
    if args.max_ml_training_rows is not None:
        config.setdefault("favorita_pipeline", {})["max_ml_training_rows"] = int(args.max_ml_training_rows)

    run_mode = _normalize_run_mode(args.run_mode or config.get("project", {}).get("run_mode", "quick"))
    config.setdefault("project", {})["run_mode"] = run_mode
    output_dir = Path(args.output_dir or config.get("project", {}).get("output_dir", "outputs"))
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    log_dir = output_dir / "logs"
    config_dir = output_dir / "configs"
    paper_table_dir = Path(args.paper_table_dir)
    paper_figure_dir = Path(args.paper_figure_dir)
    for path in [table_dir, figure_dir, log_dir, config_dir, paper_table_dir, paper_figure_dir]:
        path.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("favorita_minimal_pipeline", log_file=log_dir / "favorita_minimal_pipeline.log")
    logger.info("Starting Favorita minimal pipeline.")
    logger.info("Run mode: %s", run_mode)

    if config.get("experiments", {}).get("save_config_snapshot", True):
        snapshot = save_config_snapshot(config, config_dir, file_name="favorita_minimal_pipeline_config.yaml")
        logger.info("Saved config snapshot to %s", snapshot)

    raw_data_dir = Path(
        args.raw_data_dir
        or config.get("favorita_pipeline", {}).get("raw_data_dir")
        or Path(config.get("data", {}).get("raw_data_dir", "data/raw")) / "favorita"
    )
    max_series = _resolve_max_series(args.max_series, run_mode, config)
    logger.info("Raw data directory: %s", raw_data_dir)
    logger.info("Maximum series: %s", "all eligible series" if max_series is None else max_series)

    modeling_table, quality_report = load_favorita_modeling_table(
        raw_data_dir=raw_data_dir,
        max_series=max_series,
        min_history_length=int(config.get("data", {}).get("min_history_length", 90)),
        min_nonzero_observations=int(config.get("data", {}).get("min_nonzero_observations", 30)),
        output_table_dir=table_dir,
        output_log_dir=log_dir,
    )
    logger.info("Loaded modeling table with %s rows and %s series.", len(modeling_table), modeling_table["series_id"].nunique())

    modeling_table, split_dates = _assign_chronological_splits(modeling_table, config)
    logger.info("Train dates end at %s.", split_dates["train_end_date"].date())
    logger.info(
        "Validation window: %s to %s.",
        split_dates["validation_start_date"].date(),
        split_dates["validation_end_date"].date(),
    )
    logger.info("Test window: %s to %s.", split_dates["test_start_date"].date(), split_dates["test_end_date"].date())

    if args.reuse_forecast_table:
        forecasts = pd.read_csv(args.reuse_forecast_table, parse_dates=["date"])
        logger.info("Reused standardized forecast table from %s.", args.reuse_forecast_table)
    else:
        forecasts = _build_candidate_forecasts(modeling_table, config, logger, skip_ml=args.skip_ml)
    forecast_metrics = _summarize_forecast_metrics(forecasts)
    forecast_metrics.to_csv(table_dir / "favorita_forecast_metrics.csv", index=False)
    forecasts.to_csv(table_dir / "favorita_forecasts.csv", index=False)
    logger.info("Saved forecast metrics and standardized forecast outputs.")

    decisions = _build_decision_outputs(forecasts, modeling_table, forecast_metrics, config, logger)
    decisions.to_csv(table_dir / "favorita_decision_records.csv", index=False)
    inventory_metrics = _summarize_inventory_metrics(decisions)
    stability_metrics = _summarize_stability_metrics(decisions)
    planning_utility = _summarize_planning_utility(decisions, config.get("planning_loss_weights", {}))

    inventory_metrics.to_csv(table_dir / "favorita_inventory_metrics.csv", index=False)
    stability_metrics.to_csv(table_dir / "favorita_stability_metrics.csv", index=False)
    planning_utility.to_csv(table_dir / "favorita_planning_utility.csv", index=False)
    quality_report.to_csv(table_dir / "favorita_data_quality_report.csv", index=False)
    logger.info("Saved planning utility tables to %s.", table_dir)

    table_assets = _export_latex_tables(forecast_metrics, inventory_metrics, stability_metrics, planning_utility, paper_table_dir)
    figure_assets = _make_figures(planning_utility, stability_metrics, decisions, figure_dir, paper_figure_dir)
    baseline_tables, baseline_figures = _run_baseline_and_execution_risk_outputs(
        decisions=decisions,
        forecasts=forecasts,
        modeling_table=modeling_table,
        forecast_metrics=forecast_metrics,
        config=config,
        run_mode=run_mode,
        output_table_dir=table_dir,
        output_figure_dir=figure_dir,
        paper_table_dir=paper_table_dir,
        paper_figure_dir=paper_figure_dir,
        logger=logger,
    )
    table_assets.extend(baseline_tables)
    figure_assets.extend(baseline_figures)
    feasibility_tables, feasibility_figures = _run_feasibility_analyses(
        forecasts=forecasts,
        modeling_table=modeling_table,
        forecast_metrics=forecast_metrics,
        config=config,
        output_table_dir=table_dir,
        output_figure_dir=figure_dir,
        paper_table_dir=paper_table_dir,
        paper_figure_dir=paper_figure_dir,
        logger=logger,
    )
    table_assets.extend(feasibility_tables)
    figure_assets.extend(feasibility_figures)
    manifest_path = write_asset_manifest(args.asset_manifest_path, table_assets, figure_assets)
    logger.info("Saved LaTeX-ready tables to %s.", paper_table_dir)
    logger.info("Saved LaTeX-ready PDF figures to %s.", paper_figure_dir)
    logger.info("Updated paper asset manifest at %s.", manifest_path)
    logger.info("Favorita minimal pipeline completed successfully.")


def _resolve_max_series(max_series_arg: Optional[int], run_mode: str, config: Mapping[str, object]) -> Optional[int]:
    """Return the configured number of series for quick, medium, or full mode."""
    if max_series_arg is not None:
        return int(max_series_arg)
    data_config = config.get("data", {})
    normalized_mode = _normalize_run_mode(run_mode)
    if normalized_mode == "quick":
        return data_config.get("quick_mode_max_series", 100)
    if normalized_mode == "medium":
        return data_config.get("medium_mode_max_series", 500)
    return data_config.get("full_mode_max_series", 1000)


def _normalize_run_mode(run_mode: str) -> str:
    """Normalize command-line and config run-mode names."""
    value = str(run_mode).strip().lower()
    aliases = {
        "quick": "quick",
        "quick_mode": "quick",
        "medium": "medium",
        "medium_mode": "medium",
        "full": "full",
        "full_mode": "full",
    }
    if value not in aliases:
        raise ValueError("Unsupported run mode: {}".format(run_mode))
    return aliases[value]


def _assign_chronological_splits(data: pd.DataFrame, config: Mapping[str, object]) -> Tuple[pd.DataFrame, Dict[str, pd.Timestamp]]:
    """Assign train, validation, and test splits using the final dates."""
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    dates = pd.Series(frame["date"].dropna().unique()).sort_values().reset_index(drop=True)

    horizon = int(config.get("forecasting", {}).get("horizon", 28))
    favorita_config = config.get("favorita_pipeline", {})
    validation_horizon = int(favorita_config.get("validation_horizon", horizon))
    test_horizon = int(favorita_config.get("test_horizon", horizon))
    minimum_train_dates = int(favorita_config.get("minimum_train_dates", 120))
    required_dates = minimum_train_dates + validation_horizon + test_horizon
    if len(dates) < required_dates:
        raise ValueError(
            "Favorita data does not contain enough dates for the configured split. "
            "Need at least {} dates, found {}.".format(required_dates, len(dates))
        )

    test_start_position = len(dates) - test_horizon
    validation_start_position = test_start_position - validation_horizon
    train_end_position = validation_start_position - 1

    boundaries = {
        "train_end_date": dates.iloc[train_end_position],
        "validation_start_date": dates.iloc[validation_start_position],
        "validation_end_date": dates.iloc[test_start_position - 1],
        "test_start_date": dates.iloc[test_start_position],
        "test_end_date": dates.iloc[-1],
    }

    frame["split"] = "train"
    frame.loc[
        (frame["date"] >= boundaries["validation_start_date"]) & (frame["date"] <= boundaries["validation_end_date"]),
        "split",
    ] = "validation"
    frame.loc[(frame["date"] >= boundaries["test_start_date"]) & (frame["date"] <= boundaries["test_end_date"]), "split"] = "test"
    return frame, boundaries


def _build_candidate_forecasts(
    modeling_table: pd.DataFrame,
    config: Mapping[str, object],
    logger: logging.Logger,
    skip_ml: bool = False,
) -> pd.DataFrame:
    """Build standardized candidate forecasts for validation and test splits."""
    evaluation_frame = modeling_table[modeling_table["split"].isin(["validation", "test"])].copy()
    evaluation_frame = evaluation_frame.sort_values(["series_id", "date"]).reset_index(drop=True)
    evaluation_frame["horizon"] = _horizon_numbers(evaluation_frame)

    records: List[pd.DataFrame] = []
    for model_name, feature_column in MODEL_FEATURE_MAP.items():
        if feature_column not in evaluation_frame.columns:
            logger.warning("Skipping %s because feature column %s is unavailable.", model_name, feature_column)
            continue
        records.append(_forecast_from_feature(evaluation_frame, model_name, feature_column))

    if not skip_ml:
        records.extend(_fit_global_ml_forecasts(modeling_table, evaluation_frame, config, logger))
    else:
        logger.info("Skipping global machine-learning candidate because --skip-ml was provided.")

    if not records:
        raise ValueError("No forecast candidates were generated.")
    forecasts = pd.concat(records, ignore_index=True)
    required_columns = ["date", "series_id", "model_name", "forecast", "actual", "split", "horizon"]
    return forecasts[required_columns + ["family", "store_nbr"]].sort_values(["split", "model_name", "series_id", "date"])


def _horizon_numbers(evaluation_frame: pd.DataFrame) -> pd.Series:
    """Return split-relative horizon numbers for validation and test dates."""
    dates = evaluation_frame[["split", "date"]].drop_duplicates().sort_values(["split", "date"]).copy()
    dates["horizon"] = dates.groupby("split").cumcount() + 1
    return evaluation_frame.merge(dates, on=["split", "date"], how="left")["horizon"].astype(int)


def _forecast_from_feature(evaluation_frame: pd.DataFrame, model_name: str, feature_column: str) -> pd.DataFrame:
    """Create a forecast output frame from a leakage-aware feature column."""
    frame = evaluation_frame[["date", "series_id", "family", "store_nbr", "demand", "split", "horizon", feature_column]].copy()
    fallback = evaluation_frame.groupby("series_id")["demand_lag_1"].transform("mean").fillna(evaluation_frame["demand"].mean())
    frame["forecast"] = pd.to_numeric(frame[feature_column], errors="coerce").fillna(fallback).clip(lower=0.0)
    frame["actual"] = frame["demand"]
    frame["model_name"] = model_name
    return frame.drop(columns=[feature_column, "demand"])


def _fit_global_ml_forecasts(
    modeling_table: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    config: Mapping[str, object],
    logger: logging.Logger,
) -> List[pd.DataFrame]:
    """Fit available global ML candidates with robust fallbacks."""
    train_frame = modeling_table[modeling_table["split"] == "train"].copy()
    train_frame = train_frame.dropna(subset=["demand_lag_28", "demand_rolling_mean_28"]).copy()
    if train_frame.empty:
        logger.warning("Skipping global ML candidate because no training rows have enough lag history.")
        return []

    favorita_config = config.get("favorita_pipeline", {})
    max_training_rows = int(favorita_config.get("max_ml_training_rows", 250000))
    random_seed = int(config.get("project", {}).get("random_seed", 42))
    if len(train_frame) > max_training_rows:
        train_frame = train_frame.sample(n=max_training_rows, random_state=random_seed).sort_values(["series_id", "date"])
        logger.info("Sampled %s rows for global ML training.", max_training_rows)

    x_train, y_train, x_predict = _prepare_ml_feature_matrices(train_frame, evaluation_frame)
    candidate_estimators = _make_ml_estimators(random_seed, logger)
    outputs: List[pd.DataFrame] = []
    for model_name, estimator, estimator_label in candidate_estimators:
        try:
            estimator.fit(x_train, y_train)
            predictions = np.maximum(estimator.predict(x_predict), 0.0)
        except Exception as error:
            logger.warning("Skipping %s because model fitting failed: %s", estimator_label, error)
            continue
        output = evaluation_frame[["date", "series_id", "family", "store_nbr", "demand", "split", "horizon"]].copy()
        output["forecast"] = predictions
        output["actual"] = output["demand"]
        output["model_name"] = model_name
        outputs.append(output.drop(columns=["demand"]))
        logger.info("Generated global ML forecasts with %s.", estimator_label)

    if not outputs:
        sklearn_model = _make_sklearn_fallback_estimator(random_seed, logger)
        fallback_name, fallback_estimator, fallback_label = sklearn_model
        fallback_estimator.fit(x_train, y_train)
        predictions = np.maximum(fallback_estimator.predict(x_predict), 0.0)
        output = evaluation_frame[["date", "series_id", "family", "store_nbr", "demand", "split", "horizon"]].copy()
        output["forecast"] = predictions
        output["actual"] = output["demand"]
        output["model_name"] = fallback_name
        outputs.append(output.drop(columns=["demand"]))
        logger.info("Generated global ML forecasts with %s.", fallback_label)
    return outputs


def _make_ml_estimators(random_seed: int, logger: logging.Logger) -> List[Tuple[str, object, str]]:
    """Create all available global ML estimators.

    LightGBM and XGBoost are kept as separate candidates when both are
    installed. The scikit-learn model is a fallback only when neither boosting
    package can be used.
    """
    estimators: List[Tuple[str, object, str]] = []
    try:
        lgb = _import_lightgbm_without_optional_dask()
        estimators.append(
            (
                "global_lightgbm",
                lgb.LGBMRegressor(
                    n_estimators=120,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=random_seed,
                    n_jobs=-1,
                    verbose=-1,
                ),
                "LightGBM",
            )
        )
    except Exception as error:
        logger.warning("LightGBM is unavailable and will be skipped: %s", error)

    try:
        import xgboost as xgb

        estimators.append(
            (
                "global_xgboost",
                xgb.XGBRegressor(
                    n_estimators=80,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=random_seed,
                    n_jobs=-1,
                    verbosity=0,
                ),
                "XGBoost",
            )
        )
    except Exception as error:
        logger.warning("XGBoost is unavailable and will be skipped: %s", error)

    if estimators:
        return estimators
    return [_make_sklearn_fallback_estimator(random_seed, logger)]


def _import_lightgbm_without_optional_dask():
    """Import LightGBM while disabling optional dask/distributed integration.

    Some local environments have dask installed but not usable in sandboxed or
    offline runs. The Favorita pipeline only needs the standard sklearn-style
    LightGBM estimator, so optional distributed integrations are hidden during
    import.
    """
    blocked_modules = ["dask", "dask.dataframe", "dask.distributed", "distributed"]
    sentinel = object()
    previous_modules = {name: sys.modules.get(name, sentinel) for name in blocked_modules}
    for module_name in blocked_modules:
        sys.modules[module_name] = None
    try:
        import sklearn.utils.validation as sklearn_validation
        import lightgbm as lgb
        import lightgbm.sklearn as lgb_sklearn

        lgb_sklearn._LGBMCpuCount = lambda only_physical_cores=True: os.cpu_count() or 1
        if not hasattr(sklearn_validation, "_num_features"):
            sklearn_validation._num_features = _sklearn_num_features_compat
    finally:
        for module_name, previous_module in previous_modules.items():
            if previous_module is sentinel:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module
    return lgb


def _sklearn_num_features_compat(data) -> int:
    """Return the number of columns for older scikit-learn versions."""
    if hasattr(data, "shape") and len(data.shape) >= 2:
        return int(data.shape[1])
    if hasattr(data, "__array__"):
        array = np.asarray(data)
        if array.ndim >= 2:
            return int(array.shape[1])
    raise TypeError("Unable to determine the number of features for LightGBM compatibility.")


def _make_sklearn_fallback_estimator(random_seed: int, logger: logging.Logger) -> Tuple[str, object, str]:
    """Create the scikit-learn fallback estimator."""
    try:
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
        from sklearn.ensemble import HistGradientBoostingRegressor

        return (
            "global_sklearn",
            HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.06,
                max_leaf_nodes=31,
                random_state=random_seed,
            ),
            "scikit-learn HistGradientBoostingRegressor",
        )
    except Exception as error:
        logger.warning("HistGradientBoostingRegressor is unavailable (%s). Falling back to RandomForestRegressor.", error)

    from sklearn.ensemble import RandomForestRegressor

    return (
        "global_sklearn",
        RandomForestRegressor(
            n_estimators=60,
            max_depth=14,
            min_samples_leaf=5,
            random_state=random_seed,
            n_jobs=-1,
        ),
        "scikit-learn RandomForestRegressor",
    )


def _prepare_ml_feature_matrices(train_frame: pd.DataFrame, prediction_frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return numeric ML feature matrices aligned across train and prediction rows."""
    numeric_columns = [
        "day_of_week",
        "week_of_year",
        "month",
        "year",
        "promotion_count",
        "is_promotion",
        "is_holiday_event",
        "national_holiday_count",
        "regional_holiday_count",
        "local_holiday_count",
        "holiday_event_count",
        "dcoilwtico",
        "oil_lag_1",
        "oil_lag_7",
        "oil_rolling_mean_7",
        "transactions_lag_1",
        "transactions_lag_7",
        "transactions_rolling_mean_7",
        "demand_lag_1",
        "demand_lag_7",
        "demand_lag_14",
        "demand_lag_28",
        "demand_rolling_mean_7",
        "demand_rolling_mean_14",
        "demand_rolling_mean_28",
        "demand_rolling_mean_56",
        "demand_rolling_std_7",
        "demand_rolling_std_28",
        "demand_rolling_std_56",
        "demand_ewm_alpha_0_3",
        "zero_demand_rate_7",
        "zero_demand_rate_28",
        "store_cluster",
        "known_context_available",
    ]
    categorical_columns = ["family", "city", "state", "store_type"]
    available_numeric = [column for column in numeric_columns if column in train_frame.columns and column in prediction_frame.columns]
    available_categorical = [
        column for column in categorical_columns if column in train_frame.columns and column in prediction_frame.columns
    ]

    train_features = train_frame[available_numeric + available_categorical].copy()
    predict_features = prediction_frame[available_numeric + available_categorical].copy()
    combined = pd.concat([train_features, predict_features], keys=["train", "predict"], names=["role", "row"])

    for column in available_numeric:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
        median_value = combined.loc["train", column].median()
        if pd.isna(median_value):
            median_value = 0.0
        combined[column] = combined[column].fillna(float(median_value))
    for column in available_categorical:
        combined[column] = combined[column].fillna("missing").astype(str)

    encoded = pd.get_dummies(combined, columns=available_categorical, dummy_na=False)
    x_train = encoded.loc["train"].to_numpy(dtype=float)
    x_predict = encoded.loc["predict"].to_numpy(dtype=float)
    y_train = train_frame["demand"].to_numpy(dtype=float)
    return x_train, y_train, x_predict


def _summarize_forecast_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return split-model forecast accuracy metrics."""
    records: List[Dict[str, object]] = []
    for (split, model_name), group in forecasts.groupby(["split", "model_name"]):
        actual = group["actual"].to_numpy(dtype=float)
        forecast = group["forecast"].to_numpy(dtype=float)
        records.append(
            {
                "split": split,
                "model_name": model_name,
                "row_count": len(group),
                "series_count": group["series_id"].nunique(),
                "mean_actual": float(np.mean(actual)),
                "mean_forecast": float(np.mean(forecast)),
                "mean_absolute_error": mean_absolute_error(actual, forecast),
                "root_mean_squared_error": root_mean_squared_error(actual, forecast),
                "weighted_absolute_percentage_error": weighted_absolute_percentage_error(actual, forecast),
            }
        )
    return pd.DataFrame(records).sort_values(["split", "weighted_absolute_percentage_error", "model_name"])


def _build_decision_outputs(
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Build decision strategy records and evaluate planning outcomes."""
    test_forecasts = forecasts[forecasts["split"] == "test"].copy()
    safety_lookup = modeling_table[modeling_table["split"] == "test"][
        ["date", "series_id", "demand_rolling_std_28"]
    ].copy()
    safety_multiplier = float(config.get("favorita_pipeline", {}).get("safety_stock_multiplier", 0.5))
    safety_lookup["safety_stock"] = (
        pd.to_numeric(safety_lookup["demand_rolling_std_28"], errors="coerce").fillna(0.0) * safety_multiplier
    ).clip(lower=0.0)
    test_forecasts = test_forecasts.merge(
        safety_lookup[["date", "series_id", "safety_stock"]],
        on=["date", "series_id"],
        how="left",
    )
    test_forecasts["safety_stock"] = test_forecasts["safety_stock"].fillna(0.0)
    deployable_test_forecasts = drop_future_outcomes(test_forecasts)

    validation_metrics = forecast_metrics[forecast_metrics["split"] == "validation"].copy()
    global_best_model = validation_metrics.sort_values("weighted_absolute_percentage_error").iloc[0]["model_name"]
    family_best = _family_best_models(forecasts, global_best_model)
    logger.info("Global best validation model: %s", global_best_model)

    selected_frames: List[pd.DataFrame] = []
    for model_name in sorted(deployable_test_forecasts["model_name"].unique()):
        selected = deployable_test_forecasts[deployable_test_forecasts["model_name"] == model_name].copy()
        selected["strategy"] = "individual_{}".format(model_name)
        selected["selected_model"] = model_name
        selected_frames.append(selected)

    global_selected = deployable_test_forecasts[deployable_test_forecasts["model_name"] == global_best_model].copy()
    global_selected["strategy"] = "global_best_model"
    global_selected["selected_model"] = global_best_model
    selected_frames.append(global_selected)

    family_selected = deployable_test_forecasts.copy()
    family_selected["family_best_model"] = family_selected["family"].map(family_best).fillna(global_best_model)
    family_selected = family_selected[family_selected["model_name"] == family_selected["family_best_model"]].copy()
    family_selected["strategy"] = "family_best_model"
    family_selected["selected_model"] = family_selected["family_best_model"]
    selected_frames.append(family_selected.drop(columns=["family_best_model"]))

    selected_frames.append(_simple_ensemble_selection(deployable_test_forecasts))
    selected_frames.append(_best_inventory_cost_selection(deployable_test_forecasts, forecasts, modeling_table, forecast_metrics, config))
    selected_frames.append(_best_stability_selection(deployable_test_forecasts, forecasts, modeling_table, global_best_model, config))

    stability_selected = _stability_aware_selection(deployable_test_forecasts, forecasts, forecast_metrics, config)
    selected_frames.append(stability_selected)

    feasibility_selected = _feasibility_aware_selection(deployable_test_forecasts, forecasts, modeling_table, forecast_metrics, config)
    selected_frames.append(feasibility_selected)
    selected_frames.append(_greedy_feasibility_selection(deployable_test_forecasts, forecasts, modeling_table, forecast_metrics, config))
    selected_frames.append(_dp_feasibility_selection(deployable_test_forecasts, forecasts, modeling_table, forecast_metrics, config))
    selected_frames.append(_budgeted_dp_feasibility_selection(deployable_test_forecasts, forecasts, modeling_table, forecast_metrics, config))

    deployable_decisions = attach_actuals_for_evaluation(
        pd.concat(selected_frames, ignore_index=True),
        test_forecasts,
        key_columns=("date", "series_id"),
    )
    oracle_decisions = _oracle_realized_demand_selection(test_forecasts)
    selected_decisions = pd.concat([deployable_decisions, oracle_decisions], ignore_index=True)
    return _evaluate_selected_decisions(selected_decisions, config)


def _simple_ensemble_selection(test_forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return an equal-weight average of candidate forecasts."""
    require_no_future_outcomes(test_forecasts, "_simple_ensemble_selection")
    base_columns = ["date", "series_id", "family", "store_nbr", "split", "horizon", "safety_stock"]
    ensemble = (
        test_forecasts.groupby(base_columns, dropna=False)
        .agg(forecast=("forecast", "mean"))
        .reset_index()
    )
    ensemble["model_name"] = "simple_ensemble"
    ensemble["selected_model"] = "simple_ensemble"
    ensemble["strategy"] = "simple_ensemble"
    return ensemble


def _best_inventory_cost_selection(
    test_forecasts: pd.DataFrame,
    all_forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Select the validation model with the lowest inventory cost per demand unit."""
    require_no_future_outcomes(test_forecasts, "_best_inventory_cost_selection")
    expected_losses = _validation_expected_planning_losses(all_forecasts, modeling_table, config)
    global_losses = _global_validation_expected_planning_losses(expected_losses, forecast_metrics)
    if global_losses:
        best_model = min(
            global_losses.items(),
            key=lambda item: (float(item[1].get("inventory_cost_per_demand_unit", np.inf)), item[0]),
        )[0]
    else:
        validation_metrics = forecast_metrics[forecast_metrics["split"] == "validation"].copy()
        best_model = validation_metrics.sort_values("weighted_absolute_percentage_error").iloc[0]["model_name"]
    selected = test_forecasts[test_forecasts["model_name"] == best_model].copy()
    selected["strategy"] = "best_inventory_cost_model"
    selected["selected_model"] = best_model
    return selected


def _best_stability_selection(
    test_forecasts: pd.DataFrame,
    all_forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    fallback_model: str,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Select the validation model with the lowest planning-signal volatility."""
    require_no_future_outcomes(test_forecasts, "_best_stability_selection")
    stability = _validation_stability_by_model(all_forecasts, modeling_table, config)
    if stability.empty:
        best_model = fallback_model
    else:
        best_model = stability.sort_values(["planning_signal_volatility_total", "model_name"]).iloc[0]["model_name"]
    selected = test_forecasts[test_forecasts["model_name"] == best_model].copy()
    selected["strategy"] = "best_stability_model"
    selected["selected_model"] = best_model
    return selected


def _oracle_realized_demand_selection(test_forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return a non-deployable realized-demand upper bound for comparison."""
    base_columns = ["date", "series_id", "family", "store_nbr", "actual", "split", "horizon", "safety_stock"]
    oracle = test_forecasts[base_columns].drop_duplicates(["date", "series_id"]).copy()
    oracle["forecast"] = oracle["actual"]
    oracle["model_name"] = "oracle_realized_demand"
    oracle["selected_model"] = "oracle_realized_demand"
    oracle["strategy"] = "oracle_realized_demand"
    return oracle


def _family_best_models(forecasts: pd.DataFrame, global_best_model: str) -> Dict[str, str]:
    """Return the validation-best model for each product family."""
    validation = forecasts[forecasts["split"] == "validation"].copy()
    records = []
    for (family, model_name), group in validation.groupby(["family", "model_name"]):
        records.append(
            {
                "family": family,
                "model_name": model_name,
                "wape": weighted_absolute_percentage_error(group["actual"], group["forecast"]),
            }
        )
    if not records:
        return {}
    family_metrics = pd.DataFrame(records)
    best = family_metrics.sort_values(["family", "wape", "model_name"]).groupby("family").first().reset_index()
    mapping = dict(zip(best["family"], best["model_name"]))
    for family in validation["family"].unique():
        mapping.setdefault(family, global_best_model)
    return mapping


def _stability_aware_selection(
    test_forecasts: pd.DataFrame,
    all_forecasts: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Select models with validation loss, switching cost, and plan-change burden."""
    require_no_future_outcomes(test_forecasts, "_stability_aware_selection")
    validation_family_losses = _family_model_validation_losses(all_forecasts)
    validation_global_losses = (
        forecast_metrics[forecast_metrics["split"] == "validation"]
        .set_index("model_name")["weighted_absolute_percentage_error"]
        .to_dict()
    )
    weights = config.get("planning_loss_weights", {})
    stability_config = config.get("stability", {})
    favorita_config = config.get("favorita_pipeline", {})
    switch_penalty = float(favorita_config.get("stability_switch_penalty", 0.02))
    max_plan_change_rate = float(stability_config.get("max_plan_change_rate", 0.20))
    lambda_switch = float(weights.get("lambda_switch", 0.5))
    lambda_volatility = float(weights.get("lambda_volatility", 0.5))

    candidate_groups = {
        key: group.copy()
        for key, group in test_forecasts.groupby(["series_id", "date"], sort=False)
    }
    base_rows = (
        test_forecasts[["date", "series_id", "family", "store_nbr", "split", "horizon", "safety_stock"]]
        .drop_duplicates(["date", "series_id"])
        .sort_values(["series_id", "date"])
    )

    states: Dict[str, Dict[str, object]] = {}
    selected_records: List[Mapping[str, object]] = []
    for row in base_rows.itertuples(index=False):
        candidates = candidate_groups[(row.series_id, row.date)]
        previous_state = states.get(row.series_id, {})
        previous_model = previous_state.get("selected_model")
        previous_plan = previous_state.get("planning_signal")

        best_score = None
        best_candidate = None
        best_planning_signal = None
        for candidate in candidates.itertuples(index=False):
            expected_loss = validation_family_losses.get(
                (row.family, candidate.model_name),
                validation_global_losses.get(candidate.model_name, 1.0),
            )
            planning_signal = float(forecast_to_inventory_target([candidate.forecast], [row.safety_stock])[0])
            if previous_plan is None:
                plan_change_pct = 0.0
            else:
                plan_change_pct = abs(planning_signal - float(previous_plan)) / max(abs(float(previous_plan)), 1e-8)
            switch_cost = 0.0 if previous_model is None or previous_model == candidate.model_name else switch_penalty
            volatility_cost = max(plan_change_pct - max_plan_change_rate, 0.0)
            adjusted_score = float(expected_loss) + lambda_switch * switch_cost + lambda_volatility * volatility_cost
            if best_score is None or adjusted_score < best_score:
                best_score = adjusted_score
                best_candidate = candidate
                best_planning_signal = planning_signal

        states[row.series_id] = {
            "selected_model": best_candidate.model_name,
            "planning_signal": best_planning_signal,
        }
        selected_records.append(
            {
                "date": row.date,
                "series_id": row.series_id,
                "family": row.family,
                "store_nbr": row.store_nbr,
                "model_name": best_candidate.model_name,
                "selected_model": best_candidate.model_name,
                "forecast": float(best_candidate.forecast),
                "split": row.split,
                "horizon": int(row.horizon),
                "safety_stock": float(row.safety_stock),
                "strategy": "stability_aware_selector",
            }
        )
    return pd.DataFrame(selected_records)


def _feasibility_aware_selection(
    test_forecasts: pd.DataFrame,
    all_forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Select forecasts with an interpretable expected feasibility objective.

    The initial stability-aware selector does not need to dominate under every
    weight setting. Operational planning is a multi-objective feasibility
    problem: a small forecast gain can be rationally rejected when it creates
    inventory exposure, large plan movement, switching burden, or execution
    violations that the operation cannot absorb.
    """
    require_no_future_outcomes(test_forecasts, "_feasibility_aware_selection")
    expected_losses = _validation_expected_planning_losses(all_forecasts, modeling_table, config)
    global_expected_losses = _global_validation_expected_planning_losses(expected_losses, forecast_metrics)
    weights = config.get("planning_loss_weights", {})
    planning_config = config.get("planning", {})
    stability_config = config.get("stability", {})
    feasibility_config = config.get("feasibility_analysis", {}).get("feasibility_selector", {})
    switch_penalty = float(feasibility_config.get("switch_penalty", config.get("favorita_pipeline", {}).get("stability_switch_penalty", 0.02)))
    minimum_utility_gain = float(feasibility_config.get("minimum_utility_gain", 0.0))
    max_plan_change_rate = float(stability_config.get("max_plan_change_rate", 0.20))

    candidate_groups = {
        key: group.copy()
        for key, group in test_forecasts.groupby(["series_id", "date"], sort=False)
    }
    base_rows = (
        test_forecasts[["date", "series_id", "family", "store_nbr", "split", "horizon", "safety_stock"]]
        .drop_duplicates(["date", "series_id"])
        .sort_values(["series_id", "date"])
    )

    states: Dict[str, Dict[str, object]] = {}
    selected_records: List[Mapping[str, object]] = []
    for row in base_rows.itertuples(index=False):
        candidates = candidate_groups[(row.series_id, row.date)]
        previous_state = states.get(row.series_id, {})
        previous_model = previous_state.get("selected_model")
        previous_plan = previous_state.get("planning_signal")

        scored_candidates = []
        for candidate in candidates.itertuples(index=False):
            planning_signal = float(forecast_to_inventory_target([candidate.forecast], [row.safety_stock])[0])
            plan_change_abs, plan_change_pct, execution_violation_units = _expected_plan_burden(
                planning_signal=planning_signal,
                previous_plan=previous_plan,
                max_plan_change_rate=max_plan_change_rate,
            )
            calibrated_loss = expected_losses.get(
                (row.family, candidate.model_name),
                global_expected_losses.get(candidate.model_name, {}),
            )
            expected_forecast_loss = float(calibrated_loss.get("wape", 1.0))
            expected_inventory_loss = float(calibrated_loss.get("inventory_cost_per_demand_unit", 0.0))
            switch_cost = 0.0 if previous_model is None or previous_model == candidate.model_name else switch_penalty
            normalized_execution_violation = execution_violation_units / max(abs(float(previous_plan or planning_signal)), 1e-8)
            score = (
                float(weights.get("alpha_forecast", 1.0)) * expected_forecast_loss
                + float(weights.get("beta_inventory", 1.0)) * expected_inventory_loss
                + float(weights.get("lambda_volatility", 0.5)) * plan_change_pct
                + float(weights.get("lambda_switch", 0.5)) * switch_cost
                + float(weights.get("lambda_execution", 1.0)) * normalized_execution_violation
            )
            scored_candidates.append(
                {
                    "candidate": candidate,
                    "planning_signal": planning_signal,
                    "score": score,
                    "plan_change_abs": plan_change_abs,
                    "plan_change_pct": plan_change_pct,
                }
            )

        best = min(scored_candidates, key=lambda item: item["score"])
        if previous_model is not None:
            incumbent_matches = [item for item in scored_candidates if item["candidate"].model_name == previous_model]
            if incumbent_matches:
                incumbent = incumbent_matches[0]
                if best["candidate"].model_name != previous_model and best["score"] > incumbent["score"] - minimum_utility_gain:
                    best = incumbent

        best_candidate = best["candidate"]
        states[row.series_id] = {
            "selected_model": best_candidate.model_name,
            "planning_signal": best["planning_signal"],
        }
        selected_records.append(
            {
                "date": row.date,
                "series_id": row.series_id,
                "family": row.family,
                "store_nbr": row.store_nbr,
                "model_name": best_candidate.model_name,
                "selected_model": best_candidate.model_name,
                "forecast": float(best_candidate.forecast),
                "split": row.split,
                "horizon": int(row.horizon),
                "safety_stock": float(row.safety_stock),
                "strategy": "feasibility_aware_selector",
            }
        )
    return pd.DataFrame(selected_records)


def _greedy_feasibility_selection(
    test_forecasts: pd.DataFrame,
    all_forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Select one-step minimum expected operational-cost forecasts."""
    require_no_future_outcomes(test_forecasts, "_greedy_feasibility_selection")
    expected_losses, global_expected_losses = _favorita_dp_expected_losses(all_forecasts, modeling_table, forecast_metrics, config)
    selector = GreedyFeasibilitySelector(
        expected_losses=expected_losses,
        global_expected_losses=global_expected_losses,
        weights=config.get("planning_loss_weights", {}),
        switch_penalty=_selector_switch_penalty(config),
        max_plan_change_rate=_selector_max_plan_change_rate(config),
        calibration_group_column="family",
        strategy_name="greedy_feasibility_selector",
    )
    return selector.select(test_forecasts)


def _dp_feasibility_selection(
    test_forecasts: pd.DataFrame,
    all_forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Select finite-horizon DP minimum expected cumulative operational loss."""
    require_no_future_outcomes(test_forecasts, "_dp_feasibility_selection")
    expected_losses, global_expected_losses = _favorita_dp_expected_losses(all_forecasts, modeling_table, forecast_metrics, config)
    selector = DPFeasibilitySelector(
        expected_losses=expected_losses,
        global_expected_losses=global_expected_losses,
        weights=config.get("planning_loss_weights", {}),
        switch_penalty=_selector_switch_penalty(config),
        max_plan_change_rate=_selector_max_plan_change_rate(config),
        calibration_group_column="family",
        strategy_name="dp_feasibility_selector",
    )
    return selector.select(test_forecasts)


def _budgeted_dp_feasibility_selection(
    test_forecasts: pd.DataFrame,
    all_forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Select finite-horizon DP forecasts under a hard switch-count budget."""
    require_no_future_outcomes(test_forecasts, "_budgeted_dp_feasibility_selection")
    expected_losses, global_expected_losses = _favorita_dp_expected_losses(all_forecasts, modeling_table, forecast_metrics, config)
    selector = BudgetedDPFeasibilitySelector(
        expected_losses=expected_losses,
        global_expected_losses=global_expected_losses,
        weights=config.get("planning_loss_weights", {}),
        switch_penalty=_selector_switch_penalty(config),
        max_plan_change_rate=_selector_max_plan_change_rate(config),
        max_switches=_selector_switch_budget(config),
        calibration_group_column="family",
        strategy_name="budgeted_dp_feasibility_selector",
    )
    return selector.select(test_forecasts)


def _favorita_dp_expected_losses(
    all_forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Return validation-derived cost estimates for deployable DP selectors."""
    expected_losses = _validation_expected_planning_losses(all_forecasts, modeling_table, config)
    global_expected_losses = _global_validation_expected_planning_losses(expected_losses, forecast_metrics)
    return expected_losses, global_expected_losses


def _selector_switch_penalty(config: Mapping[str, object]) -> float:
    """Return the configured soft switch penalty for selector scoring."""
    feasibility_config = config.get("feasibility_analysis", {}).get("feasibility_selector", {})
    return float(feasibility_config.get("switch_penalty", config.get("favorita_pipeline", {}).get("stability_switch_penalty", 0.02)))


def _selector_max_plan_change_rate(config: Mapping[str, object]) -> float:
    """Return the execution-capacity plan-change rate used by selectors."""
    return float(config.get("stability", {}).get("max_plan_change_rate", 0.20))


def _selector_switch_budget(config: Mapping[str, object]) -> int:
    """Return the hard switch budget for budgeted DP selectors."""
    dp_config = config.get("feasibility_analysis", {}).get("dp_selector", {})
    return int(dp_config.get("max_switches", config.get("stability", {}).get("max_model_switches_per_window", 2)))


def _expected_plan_burden(
    planning_signal: float,
    previous_plan: Optional[object],
    max_plan_change_rate: float,
) -> Tuple[float, float, float]:
    """Return expected plan movement and capacity violation for one candidate."""
    if previous_plan is None:
        return 0.0, 0.0, 0.0
    previous_value = float(previous_plan)
    plan_change_abs = abs(float(planning_signal) - previous_value)
    plan_change_pct = plan_change_abs / max(abs(previous_value), 1e-8)
    execution_capacity = abs(previous_value) * float(max_plan_change_rate)
    execution_violation_units = max(plan_change_abs - execution_capacity, 0.0)
    return plan_change_abs, plan_change_pct, execution_violation_units


def _validation_expected_planning_losses(
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    config: Mapping[str, object],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Estimate normalized forecast and inventory losses on validation data."""
    validation = forecasts[forecasts["split"] == "validation"].copy()
    safety_lookup = modeling_table[modeling_table["split"] == "validation"][
        ["date", "series_id", "demand_rolling_std_28"]
    ].copy()
    safety_multiplier = float(config.get("favorita_pipeline", {}).get("safety_stock_multiplier", 0.5))
    safety_lookup["safety_stock"] = (
        pd.to_numeric(safety_lookup["demand_rolling_std_28"], errors="coerce").fillna(0.0) * safety_multiplier
    ).clip(lower=0.0)
    validation = validation.merge(safety_lookup[["date", "series_id", "safety_stock"]], on=["date", "series_id"], how="left")
    validation["safety_stock"] = validation["safety_stock"].fillna(0.0)

    planning_config = config.get("planning", {})
    losses: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (family, model_name), group in validation.groupby(["family", "model_name"]):
        actual = group["actual"].to_numpy(dtype=float)
        forecast = group["forecast"].to_numpy(dtype=float)
        safety_stock = group["safety_stock"].to_numpy(dtype=float)
        planning_signal = forecast_to_inventory_target(forecast, safety_stock)
        inventory_cost = compute_holding_cost(
            planning_signal,
            actual,
            holding_cost_rate=float(planning_config.get("holding_cost_rate", 1.0)),
        ) + compute_shortage_cost(
            planning_signal,
            actual,
            shortage_cost_rate=float(planning_config.get("shortage_cost_rate", 5.0)),
        )
        demand_total = max(float(np.sum(np.abs(actual))), 1e-8)
        losses[(family, model_name)] = {
            "wape": weighted_absolute_percentage_error(actual, forecast),
            "inventory_cost_per_demand_unit": float(np.sum(inventory_cost) / demand_total),
        }
    return losses


def _global_validation_expected_planning_losses(
    family_losses: Mapping[Tuple[str, str], Mapping[str, float]],
    forecast_metrics: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Return global fallback losses when a family-model calibration is absent."""
    global_losses: Dict[str, Dict[str, float]] = {}
    for (_, model_name), metrics in family_losses.items():
        global_losses.setdefault(model_name, {"wape_values": [], "inventory_values": []})
        global_losses[model_name]["wape_values"].append(float(metrics["wape"]))
        global_losses[model_name]["inventory_values"].append(float(metrics["inventory_cost_per_demand_unit"]))
    output = {
        model_name: {
            "wape": float(np.mean(values["wape_values"])),
            "inventory_cost_per_demand_unit": float(np.mean(values["inventory_values"])),
        }
        for model_name, values in global_losses.items()
    }
    for row in forecast_metrics[forecast_metrics["split"] == "validation"].itertuples(index=False):
        output.setdefault(
            row.model_name,
            {
                "wape": float(row.weighted_absolute_percentage_error),
                "inventory_cost_per_demand_unit": 0.0,
            },
        )
    return output


def _validation_stability_by_model(
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Return validation planning-signal volatility by model."""
    validation = forecasts[forecasts["split"] == "validation"].copy()
    if validation.empty:
        return pd.DataFrame(columns=["model_name", "planning_signal_volatility_total"])
    safety_lookup = modeling_table[modeling_table["split"] == "validation"][
        ["date", "series_id", "demand_rolling_std_28"]
    ].copy()
    safety_multiplier = float(config.get("favorita_pipeline", {}).get("safety_stock_multiplier", 0.5))
    safety_lookup["safety_stock"] = (
        pd.to_numeric(safety_lookup["demand_rolling_std_28"], errors="coerce").fillna(0.0) * safety_multiplier
    ).clip(lower=0.0)
    validation = validation.merge(
        safety_lookup[["date", "series_id", "safety_stock"]],
        on=["date", "series_id"],
        how="left",
    )
    validation["safety_stock"] = validation["safety_stock"].fillna(0.0)
    validation["planning_signal"] = forecast_to_inventory_target(validation["forecast"], validation["safety_stock"])
    records = []
    for model_name, model_group in validation.groupby("model_name"):
        total_volatility = 0.0
        for _, series_group in model_group.sort_values(["series_id", "date"]).groupby("series_id"):
            total_volatility += float(np.sum(compute_percentage_plan_change(series_group["planning_signal"])))
        records.append(
            {
                "model_name": model_name,
                "planning_signal_volatility_total": total_volatility,
            }
        )
    return pd.DataFrame(records)


def _family_model_validation_losses(forecasts: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Return validation WAPE by family and model."""
    validation = forecasts[forecasts["split"] == "validation"].copy()
    losses: Dict[Tuple[str, str], float] = {}
    for (family, model_name), group in validation.groupby(["family", "model_name"]):
        losses[(family, model_name)] = weighted_absolute_percentage_error(group["actual"], group["forecast"])
    return losses


def _evaluate_selected_decisions(selected_decisions: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    """Add planning signal, cost, stability, and loss columns to decisions."""
    frame = selected_decisions.copy().sort_values(["strategy", "series_id", "date"]).reset_index(drop=True)
    planning_config = config.get("planning", {})
    stability_config = config.get("stability", {})
    weights = config.get("planning_loss_weights", {})

    default_planning_signal = forecast_to_inventory_target(frame["forecast"], frame["safety_stock"])
    if "planning_signal_override" in frame.columns:
        override_signal = pd.to_numeric(frame["planning_signal_override"], errors="coerce")
        frame["planning_signal"] = override_signal.where(override_signal.notna(), default_planning_signal)
    else:
        frame["planning_signal"] = default_planning_signal
    frame["inventory_target"] = frame["planning_signal"]
    frame["holding_cost"] = compute_holding_cost(
        frame["inventory_target"],
        frame["actual"],
        holding_cost_rate=float(planning_config.get("holding_cost_rate", 1.0)),
    )
    frame["shortage_cost"] = compute_shortage_cost(
        frame["inventory_target"],
        frame["actual"],
        shortage_cost_rate=float(planning_config.get("shortage_cost_rate", 5.0)),
    )
    frame["total_inventory_cost"] = frame["holding_cost"] + frame["shortage_cost"]
    frame["service_level_hit"] = (frame["inventory_target"] >= frame["actual"]).astype(int)
    frame["absolute_plan_change"] = 0.0
    frame["plan_change_pct"] = 0.0
    frame["large_jump_flag"] = 0
    frame["model_switch_flag"] = 0
    frame["execution_capacity"] = 0.0
    frame["execution_adaptation_penalty"] = 0.0
    frame["execution_violation"] = 0

    max_plan_change_rate = float(stability_config.get("max_plan_change_rate", 0.20))
    jump_threshold = float(stability_config.get("jump_threshold", 0.25))
    for (_, _), index in frame.groupby(["strategy", "series_id"]).groups.items():
        signal = frame.loc[index, "planning_signal"].to_numpy(dtype=float)
        abs_change = compute_absolute_plan_change(signal)
        pct_change = compute_percentage_plan_change(signal)
        execution_capacity = compute_execution_capacity(signal, max_plan_change_rate=max_plan_change_rate)
        execution_penalty = compute_execution_violation(signal, execution_capacity)
        models = frame.loc[index, "selected_model"].astype(str).tolist()
        switch_flags = np.array([0] + [int(previous != current) for previous, current in zip(models[:-1], models[1:])])

        frame.loc[index, "absolute_plan_change"] = abs_change
        frame.loc[index, "plan_change_pct"] = pct_change
        frame.loc[index, "large_jump_flag"] = (pct_change > jump_threshold).astype(int)
        frame.loc[index, "model_switch_flag"] = switch_flags
        frame.loc[index, "execution_capacity"] = execution_capacity
        frame.loc[index, "execution_adaptation_penalty"] = execution_penalty
        frame.loc[index, "execution_violation"] = (execution_penalty > 0.0).astype(int)

    forecast_error = (frame["actual"] - frame["forecast"]).abs()
    frame["total_planning_loss"] = (
        float(weights.get("alpha_forecast", 1.0)) * forecast_error
        + float(weights.get("beta_inventory", 1.0)) * frame["total_inventory_cost"]
        + float(weights.get("lambda_volatility", 0.5)) * frame["plan_change_pct"]
        + float(weights.get("lambda_switch", 0.5)) * frame["model_switch_flag"]
        + float(weights.get("lambda_execution", 1.0)) * frame["execution_adaptation_penalty"]
    )
    return frame


def _summarize_inventory_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    """Return strategy-level inventory and service summaries."""
    records = []
    for strategy, group in decisions.groupby("strategy"):
        actual = group["actual"].to_numpy(dtype=float)
        target = group["inventory_target"].to_numpy(dtype=float)
        total_demand = float(np.sum(actual))
        records.append(
            {
                "strategy": strategy,
                "row_count": len(group),
                "series_count": group["series_id"].nunique(),
                "total_demand": total_demand,
                "holding_cost_total": float(group["holding_cost"].sum()),
                "shortage_cost_total": float(group["shortage_cost"].sum()),
                "total_inventory_cost": float(group["total_inventory_cost"].sum()),
                "total_inventory_cost_per_demand_unit": float(group["total_inventory_cost"].sum() / max(total_demand, 1e-8)),
                "service_level": compute_service_level(target, actual),
                "service_level_hit_rate": float(group["service_level_hit"].mean()),
            }
        )
    return pd.DataFrame(records).sort_values("total_inventory_cost")


def _summarize_stability_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    """Return strategy-level planning signal stability summaries."""
    records = []
    for strategy, group in decisions.groupby("strategy"):
        transition_count = max(len(group) - group["series_id"].nunique(), 1)
        records.append(
            {
                "strategy": strategy,
                "row_count": len(group),
                "mean_plan_change_pct": float(group["plan_change_pct"].mean()),
                "median_plan_change_pct": float(group["plan_change_pct"].median()),
                "planning_signal_volatility_total": float(group["plan_change_pct"].sum()),
                "large_jump_count": int(group["large_jump_flag"].sum()),
                "large_jump_rate": float(group["large_jump_flag"].sum() / transition_count),
                "model_switch_count": int(group["model_switch_flag"].sum()),
                "model_switch_rate": float(group["model_switch_flag"].sum() / transition_count),
                "execution_violation_count": int(group["execution_violation"].sum()),
                "execution_violation_rate": float(group["execution_violation"].sum() / transition_count),
                "execution_adaptation_penalty_total": float(group["execution_adaptation_penalty"].sum()),
            }
        )
    return pd.DataFrame(records).sort_values(["execution_violation_rate", "mean_plan_change_pct"])


def _summarize_planning_utility(decisions: pd.DataFrame, loss_weights: Mapping[str, float]) -> pd.DataFrame:
    """Return strategy-level multi-objective planning utility summaries."""
    records = []
    for strategy, group in decisions.groupby("strategy"):
        actual = group["actual"].to_numpy(dtype=float)
        forecast = group["forecast"].to_numpy(dtype=float)
        total_loss = compute_total_planning_loss(
            forecast_error=np.abs(actual - forecast),
            inventory_cost=group["total_inventory_cost"].to_numpy(dtype=float),
            planning_signal_volatility=group["plan_change_pct"].to_numpy(dtype=float),
            model_switching_cost=group["model_switch_flag"].to_numpy(dtype=float),
            execution_adaptation_penalty=group["execution_adaptation_penalty"].to_numpy(dtype=float),
            weights=loss_weights,
        )
        records.append(
            {
                "strategy": strategy,
                "selected_model_count": group["selected_model"].nunique(),
                "mean_absolute_error": mean_absolute_error(actual, forecast),
                "weighted_absolute_percentage_error": weighted_absolute_percentage_error(actual, forecast),
                "total_inventory_cost": float(group["total_inventory_cost"].sum()),
                "planning_signal_volatility_total": float(group["plan_change_pct"].sum()),
                "model_switching_cost_total": float(group["model_switch_flag"].sum()),
                "model_switch_count": int(group["model_switch_flag"].sum()),
                "execution_adaptation_penalty_total": float(group["execution_adaptation_penalty"].sum()),
                "total_planning_loss": total_loss,
                "service_level": compute_service_level(group["inventory_target"], actual),
                "large_jump_rate": float(group["large_jump_flag"].mean()),
                "execution_violation_rate": float(group["execution_violation"].mean()),
                "max_period_plan_change_pct": float(group["plan_change_pct"].max()),
            }
        )
    return pd.DataFrame(records).sort_values("total_planning_loss")


def _run_baseline_and_execution_risk_outputs(
    decisions: pd.DataFrame,
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
    run_mode: str,
    output_table_dir: Path,
    output_figure_dir: Path,
    paper_table_dir: Path,
    paper_figure_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[Path], List[Path]]:
    """Export baseline comparisons, normalized loss audits, and risk scenarios."""
    table_assets: List[Path] = []
    figure_assets: List[Path] = []
    weights = config.get("planning_loss_weights", {})
    baseline_summary, reference_values, scale_audit = _baseline_normalized_summary(decisions, weights, run_mode)
    baseline_comparison = _baseline_comparison_table(baseline_summary)

    table_assets.append(
        _save_output_and_paper_table(
            baseline_comparison,
            output_table_dir,
            paper_table_dir,
            output_stem="favorita_baseline_weights_comparison",
            paper_stem="favorita_baseline_weights_comparison_table",
            caption="Favorita baseline-weight comparison with normalized planning loss.",
            label="tab:favorita-baseline-weights-comparison",
            resize_to_textwidth=True,
        )
    )
    table_assets.append(
        _save_output_and_paper_table(
            scale_audit,
            output_table_dir,
            paper_table_dir,
            output_stem="favorita_loss_component_scale_audit",
            paper_stem="favorita_loss_component_scale_audit_table",
            caption="Loss-component scale audit for normalized planning loss.",
            label="tab:favorita-loss-component-scale-audit",
            resize_to_textwidth=True,
        )
    )
    table_assets.append(
        _save_output_and_paper_table(
            reference_values,
            output_table_dir,
            paper_table_dir,
            output_stem="normalization_reference_values",
            paper_stem="normalization_reference_values_table",
            caption="Normalization reference values for the Favorita comparison split.",
            label="tab:normalization-reference-values",
            resize_to_textwidth=True,
        )
    )
    _save_baseline_weights_tradeoff_plot(
        baseline_comparison,
        output_figure_dir / "favorita_baseline_weights_tradeoff.png",
        paper_figure_dir / "favorita_baseline_weights_tradeoff.pdf",
    )
    figure_assets.append(paper_figure_dir / "favorita_baseline_weights_tradeoff.pdf")

    dataco_percentiles, dataco_calibration, dataco_context_rates = _compute_dataco_execution_risk_tables(config, logger)
    table_assets.append(
        _save_output_and_paper_table(
            dataco_percentiles,
            output_table_dir,
            paper_table_dir,
            output_stem="dataco_execution_risk_percentiles",
            paper_stem="dataco_execution_risk_percentiles_table",
            caption="DataCo late-delivery risk percentiles used as execution-risk anchors.",
            label="tab:dataco-execution-risk-percentiles",
            resize_to_textwidth=True,
        )
    )
    table_assets.append(
        _save_output_and_paper_table(
            dataco_calibration,
            output_table_dir,
            paper_table_dir,
            output_stem="dataco_execution_risk_calibration",
            paper_stem="dataco_execution_risk_calibration_table",
            caption="DataCo execution-risk calibration audit.",
            label="tab:dataco-execution-risk-calibration",
            resize_to_textwidth=True,
        )
    )
    _save_dataco_context_risk_plot(
        dataco_context_rates,
        output_figure_dir / "dataco_late_delivery_risk_by_context.png",
        paper_figure_dir / "dataco_late_delivery_risk_by_context.pdf",
    )
    figure_assets.append(paper_figure_dir / "dataco_late_delivery_risk_by_context.pdf")

    scenario_table = _generate_execution_risk_scenario_table(dataco_percentiles, config, logger)
    table_assets.append(
        _save_output_and_paper_table(
            scenario_table,
            output_table_dir,
            paper_table_dir,
            output_stem="generated_execution_risk_scenarios",
            paper_stem="generated_execution_risk_scenarios_table",
            caption="Generated execution-risk scenarios for DataCo-informed re-evaluation.",
            label="tab:generated-execution-risk-scenarios",
            resize_to_textwidth=True,
        )
    )
    _save_lambda_scenarios_plot(
        scenario_table,
        output_figure_dir / "dataco_execution_lambda_scenarios.png",
        paper_figure_dir / "dataco_execution_lambda_scenarios.pdf",
    )
    figure_assets.append(paper_figure_dir / "dataco_execution_lambda_scenarios.pdf")

    scenario_results = _evaluate_dataco_informed_scenarios(decisions, scenario_table, config, run_mode)
    rank_reordering = _build_strategy_rank_reordering_by_weight(scenario_results)
    objective_ranking = _build_model_ranking_by_objective(scenario_results)
    table_assets.append(
        _save_output_and_paper_table(
            scenario_results,
            output_table_dir,
            paper_table_dir,
            output_stem="favorita_dataco_informed_weight_sensitivity",
            paper_stem="favorita_dataco_informed_weight_sensitivity_table",
            caption="Favorita re-evaluation under generated DataCo-informed execution-risk scenarios.",
            label="tab:favorita-dataco-informed-weight-sensitivity",
            resize_to_textwidth=True,
        )
    )
    table_assets.append(
        _save_output_and_paper_table(
            rank_reordering,
            output_table_dir,
            paper_table_dir,
            output_stem="favorita_strategy_rank_reordering_by_weight",
            paper_stem="favorita_strategy_rank_reordering_by_weight_table",
            caption="Strategy rank reordering as execution-risk sensitivity changes.",
            label="tab:favorita-strategy-rank-reordering-by-weight",
            resize_to_textwidth=True,
        )
    )
    table_assets.append(
        _save_output_and_paper_table(
            objective_ranking,
            output_table_dir,
            paper_table_dir,
            output_stem="favorita_model_ranking_by_objective",
            paper_stem="favorita_model_ranking_by_objective_table",
            caption="Favorita model ranking by objective and execution-risk scenario.",
            label="tab:favorita-model-ranking-by-objective",
            paper_data=_compact_objective_ranking_table(objective_ranking),
            resize_to_textwidth=True,
        )
    )

    _save_strategy_rank_by_execution_weight_plot(
        scenario_results,
        output_figure_dir / "favorita_strategy_rank_by_execution_weight.png",
        paper_figure_dir / "favorita_strategy_rank_by_execution_weight.pdf",
    )
    _save_scenario_metric_plot(
        scenario_results,
        metric_column="normalized_total_loss",
        y_label="Normalized Total Loss",
        png_path=output_figure_dir / "favorita_normalized_loss_by_execution_scenario.png",
        pdf_path=paper_figure_dir / "favorita_normalized_loss_by_execution_scenario.pdf",
    )
    _save_scenario_metric_plot(
        scenario_results,
        metric_column="execution_penalty",
        y_label="Execution Penalty",
        png_path=output_figure_dir / "favorita_execution_penalty_by_execution_scenario.png",
        pdf_path=paper_figure_dir / "favorita_execution_penalty_by_execution_scenario.pdf",
    )
    _save_rank_reordering_by_execution_weight_plot(
        rank_reordering,
        output_figure_dir / "favorita_rank_reordering_by_execution_weight.png",
        paper_figure_dir / "favorita_rank_reordering_by_execution_weight.pdf",
    )
    _save_model_rank_reordering_plot(
        objective_ranking,
        output_figure_dir / "favorita_model_rank_reordering.png",
        paper_figure_dir / "favorita_model_rank_reordering.pdf",
    )
    figure_assets.extend(
        [
            paper_figure_dir / "favorita_strategy_rank_by_execution_weight.pdf",
            paper_figure_dir / "favorita_normalized_loss_by_execution_scenario.pdf",
            paper_figure_dir / "favorita_execution_penalty_by_execution_scenario.pdf",
            paper_figure_dir / "favorita_rank_reordering_by_execution_weight.pdf",
            paper_figure_dir / "favorita_model_rank_reordering.pdf",
        ]
    )
    improved_tables, improved_figures = _run_improved_feasibility_method_outputs(
        base_decisions=decisions,
        forecasts=forecasts,
        modeling_table=modeling_table,
        forecast_metrics=forecast_metrics,
        scenario_table=scenario_table,
        config=config,
        run_mode=run_mode,
        output_table_dir=output_table_dir,
        output_figure_dir=output_figure_dir,
        paper_table_dir=paper_table_dir,
        paper_figure_dir=paper_figure_dir,
        logger=logger,
    )
    table_assets.extend(improved_tables)
    figure_assets.extend(improved_figures)
    return table_assets, figure_assets


def _baseline_normalized_summary(
    decisions: pd.DataFrame,
    weights: Mapping[str, float],
    run_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return baseline strategy summary with normalized loss and audit tables."""
    summary = _summarize_planning_utility(decisions, weights)
    summary, reference_values, scale_audit = add_normalized_planning_loss(
        summary,
        weights,
        dataset_name="favorita",
        run_mode=run_mode,
        split_name="test",
    )
    return summary, reference_values, scale_audit


def _baseline_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Build the requested baseline comparison table."""
    table = summary[summary["strategy"].isin(BASELINE_COMPARISON_STRATEGIES)].copy()
    table["method_name"] = table["strategy"].map(_short_strategy_label)
    table["WAPE"] = table["weighted_absolute_percentage_error"]
    table["inventory_cost"] = table["total_inventory_cost"]
    table["planning_volatility"] = table["planning_signal_volatility_total"]
    table["execution_penalty"] = table["execution_adaptation_penalty_total"]
    table["model_switch_count"] = table["model_switch_count"].astype(int)
    oracle_loss = table.loc[table["strategy"] == "oracle_realized_demand", "normalized_total_loss"]
    oracle_value = float(oracle_loss.iloc[0]) if not oracle_loss.empty else np.nan
    table["gap_to_oracle"] = table["normalized_total_loss"] - oracle_value
    table["non_deployable_upper_bound"] = table["strategy"] == "oracle_realized_demand"
    output_columns = [
        "method_name",
        "strategy",
        "non_deployable_upper_bound",
        "WAPE",
        "inventory_cost",
        "planning_volatility",
        "execution_penalty",
        "execution_violation_rate",
        "model_switch_count",
        "max_period_plan_change_pct",
        "raw_total_planning_loss",
        "normalized_inventory_component",
        "normalized_volatility_component",
        "normalized_execution_component",
        "normalized_switch_component",
        "normalized_total_loss",
        "gap_to_oracle",
    ]
    return table[output_columns].sort_values(["normalized_total_loss", "method_name"]).reset_index(drop=True)


def _compute_dataco_execution_risk_tables(
    config: Mapping[str, object],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute DataCo late-delivery risk percentiles and calibration tables."""
    dataco_dir = Path(config.get("data", {}).get("raw_data_dir", "data/raw")) / "dataco"
    try:
        orders = load_dataco_orders(raw_data_dir=dataco_dir)
    except Exception as exc:  # pragma: no cover - exercised when local raw data is absent.
        logger.warning(
            "DataCo-derived execution scenarios could not be computed. Falling back to configured default scenario values."
        )
        reason = "DataCo orders could not be loaded from {}: {}".format(dataco_dir, exc)
        return _fallback_dataco_risk_tables(reason)

    if "late_delivery_risk" not in orders.columns or orders["late_delivery_risk"].dropna().empty:
        logger.warning(
            "DataCo-derived execution scenarios could not be computed. Falling back to configured default scenario values."
        )
        return _fallback_dataco_risk_tables("The late_delivery_risk field is unavailable or empty.")

    orders["late_delivery_risk"] = pd.to_numeric(orders["late_delivery_risk"], errors="coerce")
    context_rates = _dataco_context_late_delivery_rates(orders)
    if context_rates.empty:
        logger.warning(
            "DataCo-derived execution scenarios could not be computed. Falling back to configured default scenario values."
        )
        return _fallback_dataco_risk_tables("No context-level late delivery rates could be computed.")

    percentiles = []
    for percentile in [30, 50, 75, 95]:
        percentiles.append(
            {
                "percentile_name": "p{}_late_delivery_rate".format(percentile),
                "percentile": percentile,
                "late_delivery_rate": float(np.percentile(context_rates["late_delivery_rate"], percentile)),
                "source": "dataco_derived",
                "fallback_used": False,
                "fallback_reason": "",
            }
        )
    percentile_table = pd.DataFrame(percentiles)
    calibration = _dataco_calibration_table(orders, context_rates, percentile_table)
    return percentile_table, calibration, context_rates


def _fallback_dataco_risk_tables(reason: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return explicit fallback audit tables when DataCo risk cannot be computed."""
    percentile_table = pd.DataFrame(
        [
            {
                "percentile_name": "p{}_late_delivery_rate".format(percentile),
                "percentile": percentile,
                "late_delivery_rate": np.nan,
                "source": "config_fallback",
                "fallback_used": True,
                "fallback_reason": reason,
            }
            for percentile in [30, 50, 75, 95]
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "metric_name": "dataco_execution_risk_status",
                "metric_value": np.nan,
                "metric_unit": "not_available",
                "context_type": "dataset",
                "source": "config_fallback",
                "fallback_used": True,
                "notes": reason,
            }
        ]
    )
    context_rates = pd.DataFrame(
        columns=["context_type", "context_value", "row_count", "late_delivery_rate"]
    )
    return percentile_table, calibration, context_rates


def _dataco_context_late_delivery_rates(orders: pd.DataFrame) -> pd.DataFrame:
    """Return context-level late-delivery rates for available DataCo fields."""
    context_columns = [
        ("shipping_mode", "shipping_mode"),
        ("market", "market"),
        ("region", "order_region"),
        ("country", "order_country"),
        ("product_category", "category_name" if "category_name" in orders.columns else "category_id"),
    ]
    frames = []
    for context_type, column in context_columns:
        if column not in orders.columns:
            continue
        grouped = (
            orders.groupby(column, dropna=False)
            .agg(row_count=("late_delivery_risk", "size"), late_delivery_rate=("late_delivery_risk", "mean"))
            .reset_index()
            .rename(columns={column: "context_value"})
        )
        grouped["context_type"] = context_type
        frames.append(grouped[["context_type", "context_value", "row_count", "late_delivery_rate"]])
    if not frames:
        return pd.DataFrame(columns=["context_type", "context_value", "row_count", "late_delivery_rate"])
    return pd.concat(frames, ignore_index=True).dropna(subset=["late_delivery_rate"])


def _dataco_calibration_table(
    orders: pd.DataFrame,
    context_rates: pd.DataFrame,
    percentile_table: pd.DataFrame,
) -> pd.DataFrame:
    """Return a compact audit table for DataCo execution-risk calibration."""
    records = [
        {
            "metric_name": "global_late_delivery_rate",
            "metric_value": float(orders["late_delivery_risk"].mean()),
            "metric_unit": "rate",
            "context_type": "dataset",
            "source": "dataco_derived",
            "fallback_used": False,
            "notes": "Computed from the local DataCo late_delivery_risk field.",
        },
        {
            "metric_name": "context_rate_count",
            "metric_value": float(len(context_rates)),
            "metric_unit": "count",
            "context_type": "all_contexts",
            "source": "dataco_derived",
            "fallback_used": False,
            "notes": "Number of context-level late-delivery rates used for percentile anchors.",
        },
    ]
    for row in percentile_table.itertuples(index=False):
        records.append(
            {
                "metric_name": row.percentile_name,
                "metric_value": float(row.late_delivery_rate),
                "metric_unit": "rate",
                "context_type": "context_distribution",
                "source": "dataco_derived",
                "fallback_used": False,
                "notes": "Percentile of late-delivery rates across available DataCo contexts.",
            }
        )
    if "shipment_delay_days" in orders.columns:
        delay = pd.to_numeric(orders["shipment_delay_days"], errors="coerce").dropna()
        if not delay.empty:
            records.extend(
                [
                    {
                        "metric_name": "mean_delay_days",
                        "metric_value": float(delay.mean()),
                        "metric_unit": "days",
                        "context_type": "dataset",
                        "source": "dataco_derived",
                        "fallback_used": False,
                        "notes": "Computed from actual minus scheduled shipping days.",
                    },
                    {
                        "metric_name": "p95_delay_days",
                        "metric_value": float(np.percentile(delay, 95)),
                        "metric_unit": "days",
                        "context_type": "dataset",
                        "source": "dataco_derived",
                        "fallback_used": False,
                        "notes": "Computed from actual minus scheduled shipping days.",
                    },
                ]
            )
    order_value_column = "sales" if "sales" in orders.columns else None
    if order_value_column is not None:
        values = pd.to_numeric(orders[order_value_column], errors="coerce").fillna(0.0)
        late_mask = orders["late_delivery_risk"].fillna(0.0) > 0
        records.append(
            {
                "metric_name": "order_value_at_risk",
                "metric_value": float(values[late_mask].sum()),
                "metric_unit": "sales_value",
                "context_type": "dataset",
                "source": "dataco_derived",
                "fallback_used": False,
                "notes": "Total sales value attached to rows marked as late-delivery risk.",
            }
        )
    return pd.DataFrame(records)


def build_execution_risk_scenarios(
    risk_percentiles: Mapping[int, float],
    base_lambda: float,
    risk_sensitivity: float,
    min_lambda: float,
    max_lambda: float,
    percentile_anchors: Optional[Mapping[str, int]] = None,
) -> Dict[str, Dict[str, object]]:
    """Generate execution-risk scenarios from late-delivery percentile anchors."""
    anchors = percentile_anchors or {
        "dataco_low": 30,
        "dataco_median": 50,
        "dataco_high": 75,
        "dataco_severe": 95,
    }
    formula = "lambda_execution = clip(base_lambda + risk_sensitivity * late_delivery_rate_anchor, min_lambda, max_lambda)"
    scenarios: Dict[str, Dict[str, object]] = {
        "baseline": {
            "scenario_name": "baseline",
            "percentile_anchor": 0,
            "late_delivery_rate_anchor": 0.0,
            "lambda_execution": float(np.clip(base_lambda, min_lambda, max_lambda)),
            "mapping_formula": formula,
            "source": "dataco_derived",
            "fallback_used": False,
        }
    }
    for scenario_name, percentile in anchors.items():
        anchor_value = float(risk_percentiles[int(percentile)])
        scenarios[scenario_name] = {
            "scenario_name": scenario_name,
            "percentile_anchor": int(percentile),
            "late_delivery_rate_anchor": anchor_value,
            "lambda_execution": float(np.clip(base_lambda + risk_sensitivity * anchor_value, min_lambda, max_lambda)),
            "mapping_formula": formula,
            "source": "dataco_derived",
            "fallback_used": False,
        }
    return scenarios


def _generate_execution_risk_scenario_table(
    percentile_table: pd.DataFrame,
    config: Mapping[str, object],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Return generated DataCo-informed or fallback execution-risk scenarios."""
    mapping_config = config.get("execution_risk_scenario_mapping", {})
    fallback_config = config.get("execution_risk_scenarios_fallback", {})
    percentile_anchors = mapping_config.get(
        "percentile_anchors",
        {"dataco_low": 30, "dataco_median": 50, "dataco_high": 75, "dataco_severe": 95},
    )
    usable_percentiles = {
        int(row.percentile): float(row.late_delivery_rate)
        for row in percentile_table.itertuples(index=False)
        if not bool(row.fallback_used) and pd.notna(row.late_delivery_rate)
    }
    if usable_percentiles and all(int(value) in usable_percentiles for value in percentile_anchors.values()):
        scenarios = build_execution_risk_scenarios(
            usable_percentiles,
            base_lambda=float(mapping_config.get("base_lambda", 0.05)),
            risk_sensitivity=float(mapping_config.get("risk_sensitivity", 0.50)),
            min_lambda=float(mapping_config.get("min_lambda", 0.05)),
            max_lambda=float(mapping_config.get("max_lambda", 0.60)),
            percentile_anchors=percentile_anchors,
        )
        return pd.DataFrame([scenarios[name] for name in ["baseline"] + list(percentile_anchors.keys())])

    logger.warning(
        "DataCo-derived execution scenarios could not be computed. Falling back to configured default scenario values."
    )
    records = []
    formula = "Configured fallback lambda_execution values were used because DataCo anchors were unavailable."
    for scenario_name in ["baseline", "dataco_low", "dataco_median", "dataco_high", "dataco_severe"]:
        records.append(
            {
                "scenario_name": scenario_name,
                "percentile_anchor": percentile_anchors.get(scenario_name, 0),
                "late_delivery_rate_anchor": np.nan,
                "lambda_execution": float(fallback_config.get(scenario_name, {}).get("lambda_execution", 0.10)),
                "mapping_formula": formula,
                "source": "config_fallback",
                "fallback_used": True,
            }
        )
    return pd.DataFrame(records)


def _evaluate_dataco_informed_scenarios(
    decisions: pd.DataFrame,
    scenario_table: pd.DataFrame,
    config: Mapping[str, object],
    run_mode: str,
) -> pd.DataFrame:
    """Re-evaluate baseline strategies under generated execution-risk weights."""
    rows = []
    base_weights = config.get("planning_loss_weights", {})
    for scenario in scenario_table.itertuples(index=False):
        scenario_weights = dict(base_weights)
        scenario_weights["lambda_execution"] = float(scenario.lambda_execution)
        summary = _summarize_planning_utility(decisions, scenario_weights)
        summary, _, _ = add_normalized_planning_loss(
            summary,
            scenario_weights,
            dataset_name="favorita",
            run_mode=run_mode,
            split_name="test",
        )
        summary = summary[summary["strategy"].isin(BASELINE_COMPARISON_STRATEGIES)].copy()
        summary["scenario_name"] = scenario.scenario_name
        summary["run_mode"] = run_mode
        summary["lambda_execution"] = float(scenario.lambda_execution)
        summary["late_delivery_rate_anchor"] = scenario.late_delivery_rate_anchor
        summary["source"] = scenario.source
        summary["fallback_used"] = bool(scenario.fallback_used)
        summary["method_name"] = summary["strategy"].map(_short_strategy_label)
        summary["WAPE"] = summary["weighted_absolute_percentage_error"]
        summary["inventory_cost"] = summary["total_inventory_cost"]
        summary["planning_volatility"] = summary["planning_signal_volatility_total"]
        summary["execution_penalty"] = summary["execution_adaptation_penalty_total"]
        summary["rank_by_normalized_total_loss"] = summary["normalized_total_loss"].rank(method="min").astype(int)
        summary["rank_by_WAPE"] = summary["WAPE"].rank(method="min").astype(int)
        summary["rank_by_execution_penalty"] = summary["execution_penalty"].rank(method="min").astype(int)
        rows.append(summary)
    result = pd.concat(rows, ignore_index=True)
    output_columns = [
        "run_mode",
        "scenario_name",
        "lambda_execution",
        "late_delivery_rate_anchor",
        "source",
        "fallback_used",
        "method_name",
        "strategy",
        "WAPE",
        "inventory_cost",
        "planning_volatility",
        "execution_penalty",
        "execution_violation_rate",
        "model_switch_count",
        "normalized_inventory_component",
        "normalized_volatility_component",
        "normalized_execution_component",
        "normalized_switch_component",
        "normalized_total_loss",
        "raw_total_planning_loss",
        "rank_by_normalized_total_loss",
        "rank_by_WAPE",
        "rank_by_execution_penalty",
    ]
    return result[output_columns].sort_values(["scenario_name", "rank_by_normalized_total_loss", "method_name"])


def _run_improved_feasibility_method_outputs(
    base_decisions: pd.DataFrame,
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    scenario_table: pd.DataFrame,
    config: Mapping[str, object],
    run_mode: str,
    output_table_dir: Path,
    output_figure_dir: Path,
    paper_table_dir: Path,
    paper_figure_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[Path], List[Path]]:
    """Run improved feasibility-aware smoothing and ensemble comparisons."""
    logger.info("Running improved feasibility-aware smoothing and ensemble comparisons.")
    rows: List[pd.DataFrame] = []
    metadata_rows: List[pd.DataFrame] = []
    base_weights = config.get("planning_loss_weights", {})

    for scenario in scenario_table.itertuples(index=False):
        scenario_weights = dict(base_weights)
        scenario_weights["lambda_execution"] = float(scenario.lambda_execution)
        scenario_config = _copy_config_with_updates(config, loss_weights=scenario_weights)
        improved_selected, method_metadata = _build_improved_feasibility_method_selections(
            forecasts=forecasts,
            modeling_table=modeling_table,
            forecast_metrics=forecast_metrics,
            config=scenario_config,
            scenario_name=str(scenario.scenario_name),
            scenario_weights=scenario_weights,
        )
        improved_decisions = _evaluate_selected_decisions(improved_selected, scenario_config)
        combined_decisions = pd.concat([base_decisions, improved_decisions], ignore_index=True, sort=False)
        summary = _summarize_planning_utility(combined_decisions, scenario_weights)
        summary, _, _ = add_normalized_planning_loss(
            summary,
            scenario_weights,
            dataset_name="favorita",
            run_mode=run_mode,
            split_name="test",
        )
        summary = summary[summary["strategy"].isin(IMPROVED_METHOD_COMPARISON_STRATEGIES)].copy()
        summary["scenario_name"] = scenario.scenario_name
        summary["run_mode"] = run_mode
        summary["lambda_execution"] = float(scenario.lambda_execution)
        summary["late_delivery_rate_anchor"] = scenario.late_delivery_rate_anchor
        summary["source"] = scenario.source
        summary["fallback_used"] = bool(scenario.fallback_used)
        summary["method_name"] = summary["strategy"].map(_short_strategy_label)
        summary["WAPE"] = summary["weighted_absolute_percentage_error"]
        summary["inventory_cost"] = summary["total_inventory_cost"]
        summary["planning_volatility"] = summary["planning_signal_volatility_total"]
        summary["execution_penalty"] = summary["execution_adaptation_penalty_total"]
        summary["rank_by_normalized_total_loss"] = summary["normalized_total_loss"].rank(method="min").astype(int)
        summary["rank_by_WAPE"] = summary["WAPE"].rank(method="min").astype(int)
        summary["rank_by_execution_penalty"] = summary["execution_penalty"].rank(method="min").astype(int)
        oracle_loss = summary.loc[summary["strategy"] == "oracle_realized_demand", "normalized_total_loss"]
        oracle_value = float(oracle_loss.iloc[0]) if not oracle_loss.empty else np.nan
        summary["gap_to_oracle"] = summary["normalized_total_loss"] - oracle_value
        rows.append(summary)

        method_metadata["scenario_name"] = scenario.scenario_name
        method_metadata["run_mode"] = run_mode
        metadata_rows.append(method_metadata)

    result = pd.concat(rows, ignore_index=True)
    metadata = pd.concat(metadata_rows, ignore_index=True) if metadata_rows else pd.DataFrame()
    if not metadata.empty:
        result = result.merge(metadata, on=["run_mode", "scenario_name", "strategy"], how="left")
    result["method_family"] = result["method_family"].fillna(result["strategy"].map(_method_family_from_strategy))

    improved_methods = _improved_method_output_columns(result)
    smoothing_alpha = improved_methods[improved_methods["method_family"] == "FeasibilityAwareSmoothed"].copy()
    ensemble_comparison = improved_methods[
        improved_methods["method_family"].isin(["FeasibilityAwareEnsemble", "ReferenceEnsemble"])
    ].copy()
    rankings = _build_improved_method_rankings(improved_methods)

    table_assets = []
    table_assets.append(
        _save_output_and_paper_table(
            improved_methods,
            output_table_dir,
            paper_table_dir,
            output_stem="favorita_improved_feasibility_methods",
            paper_stem="favorita_improved_feasibility_methods_table",
            caption="Improved feasibility-aware method comparison under DataCo-informed scenarios.",
            label="tab:favorita-improved-feasibility-methods",
            resize_to_textwidth=True,
        )
    )
    table_assets.append(
        _save_output_and_paper_table(
            smoothing_alpha,
            output_table_dir,
            paper_table_dir,
            output_stem="favorita_smoothing_alpha_sensitivity",
            paper_stem="favorita_smoothing_alpha_sensitivity_table",
            caption="Smoothing alpha sensitivity for feasibility-aware gradual adaptation.",
            label="tab:favorita-smoothing-alpha-sensitivity",
            resize_to_textwidth=True,
        )
    )
    table_assets.append(
        _save_output_and_paper_table(
            ensemble_comparison,
            output_table_dir,
            paper_table_dir,
            output_stem="favorita_feasibility_ensemble_comparison",
            paper_stem="favorita_feasibility_ensemble_comparison_table",
            caption="Feasibility-aware ensemble comparison under DataCo-informed scenarios.",
            label="tab:favorita-feasibility-ensemble-comparison",
            resize_to_textwidth=True,
        )
    )
    table_assets.append(
        _save_output_and_paper_table(
            rankings,
            output_table_dir,
            paper_table_dir,
            output_stem="favorita_improved_method_rankings",
            paper_stem="favorita_improved_method_rankings_table",
            caption="Objective-specific rankings for improved feasibility-aware methods.",
            label="tab:favorita-improved-method-rankings",
            resize_to_textwidth=True,
        )
    )

    figure_assets = _make_improved_feasibility_figures(
        improved_methods=improved_methods,
        smoothing_alpha=smoothing_alpha,
        ensemble_comparison=ensemble_comparison,
        output_figure_dir=output_figure_dir,
        paper_figure_dir=paper_figure_dir,
    )
    logger.info("Saved improved feasibility-aware method outputs.")
    return table_assets, figure_assets


def _build_improved_feasibility_method_selections(
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
    scenario_name: str,
    scenario_weights: Mapping[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build scenario-specific improved strategy decisions and metadata."""
    test_forecasts_with_actual = _prepare_forecasts_with_safety_stock(forecasts, modeling_table, config, split_name="test")
    test_forecasts = drop_future_outcomes(test_forecasts_with_actual)
    expected_losses = _validation_expected_planning_losses(forecasts, modeling_table, config)
    global_expected_losses = _global_validation_expected_planning_losses(expected_losses, forecast_metrics)
    method_config = config.get("improved_feasibility_methods", {})
    smoothing_config = method_config.get("smoothing", {})
    ensemble_config = method_config.get("ensemble", {})

    selected_frames: List[pd.DataFrame] = []
    metadata_records: List[Dict[str, object]] = []

    alpha_grid = [float(value) for value in smoothing_config.get("utility_based_alpha_grid", [0.10, 0.25, 0.50, 0.75, 1.00])]
    alpha_cv_scores = _cross_validate_smoothing_alpha(
        forecasts=forecasts,
        modeling_table=modeling_table,
        forecast_metrics=forecast_metrics,
        config=config,
        scenario_weights=scenario_weights,
        alpha_grid=alpha_grid,
        fold_count=int(smoothing_config.get("utility_validation_folds", 3)),
    )
    cv_lookup = alpha_cv_scores.set_index("alpha")["validation_cv_normalized_loss"].to_dict() if not alpha_cv_scores.empty else {}

    for alpha in [float(value) for value in smoothing_config.get("fixed_alpha_values", [0.25, 0.50, 0.75])]:
        strategy = "feasibility_aware_smoothed_alpha_{}".format(_alpha_strategy_suffix(alpha))
        selected_frames.append(
            _feasibility_aware_smoothed_selection(
                test_forecasts,
                expected_losses,
                global_expected_losses,
                config,
                alpha=alpha,
                strategy=strategy,
            )
        )
        metadata_records.append(
            {
                "strategy": strategy,
                "method_family": "FeasibilityAwareSmoothed",
                "method_variant": "fixed_alpha_smoothing",
                "smoothing_alpha": alpha,
                "validation_cv_normalized_loss": cv_lookup.get(alpha, np.nan),
            }
        )

    scenario_alpha_map = smoothing_config.get("scenario_based_alpha", {})
    scenario_alpha = float(scenario_alpha_map.get(scenario_name, scenario_alpha_map.get("default", 0.50)))
    selected_frames.append(
        _feasibility_aware_smoothed_selection(
            test_forecasts,
            expected_losses,
            global_expected_losses,
            config,
            alpha=scenario_alpha,
            strategy="feasibility_aware_smoothed_scenario_alpha",
        )
    )
    metadata_records.append(
        {
            "strategy": "feasibility_aware_smoothed_scenario_alpha",
            "method_family": "FeasibilityAwareSmoothed",
            "method_variant": "scenario_based_alpha",
            "smoothing_alpha": scenario_alpha,
            "validation_cv_normalized_loss": cv_lookup.get(scenario_alpha, np.nan),
        }
    )

    utility_alpha = _select_alpha_from_cv(alpha_cv_scores)
    selected_frames.append(
        _feasibility_aware_smoothed_selection(
            test_forecasts,
            expected_losses,
            global_expected_losses,
            config,
            alpha=utility_alpha,
            strategy="feasibility_aware_smoothed_utility_alpha",
        )
    )
    metadata_records.append(
        {
            "strategy": "feasibility_aware_smoothed_utility_alpha",
            "method_family": "FeasibilityAwareSmoothed",
            "method_variant": "utility_based_alpha",
            "smoothing_alpha": utility_alpha,
            "validation_cv_normalized_loss": cv_lookup.get(utility_alpha, np.nan),
        }
    )

    accuracy_weights = _inverse_accuracy_model_weights(forecast_metrics, test_forecasts["model_name"].unique(), ensemble_config)
    operational_losses = _validation_operational_loss_by_model(
        forecasts=forecasts,
        modeling_table=modeling_table,
        forecast_metrics=forecast_metrics,
        config=config,
        scenario_weights=scenario_weights,
    )
    operational_weights = _inverse_metric_model_weights(operational_losses, ensemble_config)
    constrained_weights, constrained_metadata = _select_constrained_ensemble_weights(
        forecasts=forecasts,
        modeling_table=modeling_table,
        forecast_metrics=forecast_metrics,
        config=config,
        scenario_weights=scenario_weights,
        accuracy_weights=accuracy_weights,
        operational_weights=operational_weights,
        ensemble_config=ensemble_config,
    )

    ensemble_specs = [
        (
            "feasibility_aware_ensemble_inverse_accuracy",
            "inverse_accuracy_weighted_ensemble",
            accuracy_weights,
            "validation_wape",
            np.nan,
        ),
        (
            "feasibility_aware_ensemble_inverse_operational_loss",
            "inverse_operational_loss_weighted_ensemble",
            operational_weights,
            "validation_normalized_operational_loss",
            np.nan,
        ),
        (
            "feasibility_aware_ensemble_constrained",
            "constrained_weighted_ensemble",
            constrained_weights,
            "blocked_cv_normalized_planning_loss",
            constrained_metadata.get("validation_cv_normalized_loss", np.nan),
        ),
    ]
    for strategy, variant, weights, weight_basis, cv_loss in ensemble_specs:
        selected_frames.append(_weighted_ensemble_selection(test_forecasts, weights, strategy))
        metadata_records.append(
            {
                "strategy": strategy,
                "method_family": "FeasibilityAwareEnsemble",
                "method_variant": variant,
                "ensemble_weight_basis": weight_basis,
                "ensemble_weight_entropy": _weight_entropy(weights),
                "selected_blend": constrained_metadata.get("selected_blend", np.nan) if strategy.endswith("constrained") else np.nan,
                "validation_cv_normalized_loss": cv_loss,
            }
        )

    selected = attach_actuals_for_evaluation(
        pd.concat(selected_frames, ignore_index=True),
        test_forecasts_with_actual,
        key_columns=("date", "series_id"),
    )
    return selected, pd.DataFrame(metadata_records)


def _feasibility_aware_smoothed_selection(
    test_forecasts: pd.DataFrame,
    expected_losses: Mapping[Tuple[str, str], Mapping[str, float]],
    global_expected_losses: Mapping[str, Mapping[str, float]],
    config: Mapping[str, object],
    alpha: float,
    strategy: str,
) -> pd.DataFrame:
    """Select forecast candidates and gradually adapt the executable plan."""
    require_no_future_outcomes(test_forecasts, "_feasibility_aware_smoothed_selection")
    weights = config.get("planning_loss_weights", {})
    stability_config = config.get("stability", {})
    feasibility_config = config.get("feasibility_analysis", {}).get("feasibility_selector", {})
    switch_penalty = float(feasibility_config.get("switch_penalty", config.get("favorita_pipeline", {}).get("stability_switch_penalty", 0.02)))
    minimum_utility_gain = float(feasibility_config.get("minimum_utility_gain", 0.0))
    max_plan_change_rate = float(stability_config.get("max_plan_change_rate", 0.20))
    bounded_alpha = float(np.clip(alpha, 0.0, 1.0))

    candidate_groups = {
        key: group.copy()
        for key, group in test_forecasts.groupby(["series_id", "date"], sort=False)
    }
    base_rows = (
        test_forecasts[["date", "series_id", "family", "store_nbr", "split", "horizon", "safety_stock"]]
        .drop_duplicates(["date", "series_id"])
        .sort_values(["series_id", "date"])
    )

    states: Dict[str, Dict[str, object]] = {}
    selected_records: List[Mapping[str, object]] = []
    for row in base_rows.itertuples(index=False):
        candidates = candidate_groups[(row.series_id, row.date)]
        previous_state = states.get(row.series_id, {})
        previous_model = previous_state.get("selected_model")
        previous_plan = previous_state.get("planning_signal")

        scored_candidates = []
        for candidate in candidates.itertuples(index=False):
            candidate_plan = float(forecast_to_inventory_target([candidate.forecast], [row.safety_stock])[0])
            if previous_plan is None:
                final_plan = candidate_plan
            else:
                final_plan = bounded_alpha * candidate_plan + (1.0 - bounded_alpha) * float(previous_plan)
            plan_change_abs, plan_change_pct, execution_violation_units = _expected_plan_burden(
                planning_signal=final_plan,
                previous_plan=previous_plan,
                max_plan_change_rate=max_plan_change_rate,
            )
            calibrated_loss = expected_losses.get(
                (row.family, candidate.model_name),
                global_expected_losses.get(candidate.model_name, {}),
            )
            expected_forecast_loss = float(calibrated_loss.get("wape", 1.0))
            expected_inventory_loss = float(calibrated_loss.get("inventory_cost_per_demand_unit", 0.0))
            switch_cost = 0.0 if previous_model is None or previous_model == candidate.model_name else switch_penalty
            normalized_execution_violation = execution_violation_units / max(abs(float(previous_plan or final_plan)), 1e-8)
            score = (
                float(weights.get("alpha_forecast", 1.0)) * expected_forecast_loss
                + float(weights.get("beta_inventory", 1.0)) * expected_inventory_loss
                + float(weights.get("lambda_volatility", 0.5)) * plan_change_pct
                + float(weights.get("lambda_switch", 0.5)) * switch_cost
                + float(weights.get("lambda_execution", 1.0)) * normalized_execution_violation
            )
            scored_candidates.append(
                {
                    "candidate": candidate,
                    "candidate_plan": candidate_plan,
                    "final_plan": final_plan,
                    "score": score,
                }
            )

        best = min(scored_candidates, key=lambda item: item["score"])
        if previous_model is not None:
            incumbent_matches = [item for item in scored_candidates if item["candidate"].model_name == previous_model]
            if incumbent_matches:
                incumbent = incumbent_matches[0]
                if best["candidate"].model_name != previous_model and best["score"] > incumbent["score"] - minimum_utility_gain:
                    best = incumbent

        best_candidate = best["candidate"]
        states[row.series_id] = {
            "selected_model": best_candidate.model_name,
            "planning_signal": best["final_plan"],
        }
        selected_records.append(
            {
                "date": row.date,
                "series_id": row.series_id,
                "family": row.family,
                "store_nbr": row.store_nbr,
                "model_name": best_candidate.model_name,
                "selected_model": best_candidate.model_name,
                "forecast": float(best_candidate.forecast),
                "split": row.split,
                "horizon": int(row.horizon),
                "safety_stock": float(row.safety_stock),
                "planning_signal_override": float(best["final_plan"]),
                "strategy": strategy,
            }
        )
    return pd.DataFrame(selected_records)


def _prepare_forecasts_with_safety_stock(
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    config: Mapping[str, object],
    split_name: str,
) -> pd.DataFrame:
    """Return forecast rows for one split with executable safety stock attached."""
    frame = forecasts[forecasts["split"] == split_name].copy()
    if "safety_stock" in frame.columns:
        frame = frame.drop(columns=["safety_stock"])
    safety_lookup = modeling_table[modeling_table["split"] == split_name][
        ["date", "series_id", "demand_rolling_std_28"]
    ].copy()
    safety_multiplier = float(config.get("favorita_pipeline", {}).get("safety_stock_multiplier", 0.5))
    safety_lookup["safety_stock"] = (
        pd.to_numeric(safety_lookup["demand_rolling_std_28"], errors="coerce").fillna(0.0) * safety_multiplier
    ).clip(lower=0.0)
    frame = frame.merge(safety_lookup[["date", "series_id", "safety_stock"]], on=["date", "series_id"], how="left")
    frame["safety_stock"] = frame["safety_stock"].fillna(0.0)
    return frame


def _expected_losses_from_prepared_forecasts(
    prepared_forecasts: pd.DataFrame,
    config: Mapping[str, object],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Estimate family-model forecast and inventory losses from prepared rows."""
    planning_config = config.get("planning", {})
    losses: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (family, model_name), group in prepared_forecasts.groupby(["family", "model_name"]):
        actual = group["actual"].to_numpy(dtype=float)
        forecast = group["forecast"].to_numpy(dtype=float)
        safety_stock = group["safety_stock"].to_numpy(dtype=float)
        planning_signal = forecast_to_inventory_target(forecast, safety_stock)
        inventory_cost = compute_holding_cost(
            planning_signal,
            actual,
            holding_cost_rate=float(planning_config.get("holding_cost_rate", 1.0)),
        ) + compute_shortage_cost(
            planning_signal,
            actual,
            shortage_cost_rate=float(planning_config.get("shortage_cost_rate", 5.0)),
        )
        demand_total = max(float(np.sum(np.abs(actual))), 1e-8)
        losses[(family, model_name)] = {
            "wape": weighted_absolute_percentage_error(actual, forecast),
            "inventory_cost_per_demand_unit": float(np.sum(inventory_cost) / demand_total),
        }
    return losses


def _global_losses_from_expected_losses(
    family_losses: Mapping[Tuple[str, str], Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Return model-level fallback losses from family-level calibration."""
    global_losses: Dict[str, Dict[str, List[float]]] = {}
    for (_, model_name), metrics in family_losses.items():
        global_losses.setdefault(model_name, {"wape": [], "inventory": []})
        global_losses[model_name]["wape"].append(float(metrics.get("wape", 1.0)))
        global_losses[model_name]["inventory"].append(float(metrics.get("inventory_cost_per_demand_unit", 0.0)))
    return {
        model_name: {
            "wape": float(np.mean(values["wape"])),
            "inventory_cost_per_demand_unit": float(np.mean(values["inventory"])),
        }
        for model_name, values in global_losses.items()
    }


def _cross_validate_smoothing_alpha(
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
    scenario_weights: Mapping[str, float],
    alpha_grid: Sequence[float],
    fold_count: int,
) -> pd.DataFrame:
    """Choose smoothing speed with blocked validation folds to limit overfitting."""
    validation = _prepare_forecasts_with_safety_stock(forecasts, modeling_table, config, split_name="validation")
    if validation.empty:
        return pd.DataFrame(columns=["alpha", "validation_cv_normalized_loss", "validation_cv_folds"])
    folds = _chronological_date_folds(validation["date"], max(int(fold_count), 2))
    records = []
    for alpha in alpha_grid:
        fold_losses = []
        for fold_dates in folds:
            if not fold_dates:
                continue
            fold_dates_set = set(fold_dates)
            fold_start = min(fold_dates_set)
            calibration = validation[validation["date"] < fold_start].copy()
            if calibration.empty:
                calibration = validation[~validation["date"].isin(fold_dates_set)].copy()
            evaluation = validation[validation["date"].isin(fold_dates_set)].copy()
            if calibration.empty or evaluation.empty:
                continue
            expected_losses = _expected_losses_from_prepared_forecasts(calibration, config)
            global_expected = _global_losses_from_expected_losses(expected_losses)
            best_reference_model = _best_model_from_global_losses(global_expected, forecast_metrics)
            deployable_evaluation = drop_future_outcomes(evaluation)
            reference = _fixed_model_selection(deployable_evaluation, best_reference_model, "global_best_model")
            selected = _feasibility_aware_smoothed_selection(
                deployable_evaluation,
                expected_losses,
                global_expected,
                config,
                alpha=float(alpha),
                strategy="feasibility_aware_smoothed_alpha_cv",
            )
            fold_decisions = attach_actuals_for_evaluation(
                pd.concat([reference, selected], ignore_index=True),
                evaluation,
                key_columns=("date", "series_id"),
            )
            decisions = _evaluate_selected_decisions(fold_decisions, config)
            summary = _summarize_planning_utility(decisions, scenario_weights)
            summary, _, _ = add_normalized_planning_loss(
                summary,
                scenario_weights,
                reference_strategy="global_best_model",
                dataset_name="favorita",
                run_mode=_normalize_run_mode(config.get("project", {}).get("run_mode", "quick")),
                split_name="validation_cv",
            )
            value = summary.loc[
                summary["strategy"] == "feasibility_aware_smoothed_alpha_cv",
                "normalized_total_loss",
            ]
            if not value.empty:
                fold_losses.append(float(value.iloc[0]))
        records.append(
            {
                "alpha": float(alpha),
                "validation_cv_normalized_loss": float(np.mean(fold_losses)) if fold_losses else np.nan,
                "validation_cv_folds": len(fold_losses),
            }
        )
    return pd.DataFrame(records)


def _chronological_date_folds(dates: Sequence[object], fold_count: int) -> List[List[pd.Timestamp]]:
    """Split unique dates into chronological blocked folds."""
    unique_dates = pd.to_datetime(pd.Series(dates).dropna().unique())
    unique_dates = sorted(pd.Timestamp(value) for value in unique_dates)
    if not unique_dates:
        return []
    arrays = np.array_split(np.array(unique_dates, dtype=object), min(max(fold_count, 1), len(unique_dates)))
    return [[pd.Timestamp(value) for value in values.tolist()] for values in arrays if len(values) > 0]


def _select_alpha_from_cv(alpha_cv_scores: pd.DataFrame) -> float:
    """Return the validation-CV alpha, preferring less smoothing on ties."""
    if alpha_cv_scores.empty or alpha_cv_scores["validation_cv_normalized_loss"].isna().all():
        return 0.50
    table = alpha_cv_scores.dropna(subset=["validation_cv_normalized_loss"]).copy()
    table = table.sort_values(["validation_cv_normalized_loss", "alpha"], ascending=[True, False])
    return float(table.iloc[0]["alpha"])


def _fixed_model_selection(prepared_forecasts: pd.DataFrame, model_name: str, strategy: str) -> pd.DataFrame:
    """Return prepared forecast rows for one fixed model and strategy name."""
    selected = prepared_forecasts[prepared_forecasts["model_name"] == model_name].copy()
    if selected.empty:
        fallback_model = str(prepared_forecasts["model_name"].iloc[0])
        selected = prepared_forecasts[prepared_forecasts["model_name"] == fallback_model].copy()
        model_name = fallback_model
    selected["selected_model"] = model_name
    selected["strategy"] = strategy
    return selected


def _best_model_from_global_losses(
    global_expected_losses: Mapping[str, Mapping[str, float]],
    forecast_metrics: pd.DataFrame,
) -> str:
    """Return the lowest validation-WAPE model available in calibration losses."""
    if global_expected_losses:
        return min(global_expected_losses.items(), key=lambda item: (float(item[1].get("wape", np.inf)), item[0]))[0]
    validation_metrics = forecast_metrics[forecast_metrics["split"] == "validation"].copy()
    return str(validation_metrics.sort_values("weighted_absolute_percentage_error").iloc[0]["model_name"])


def _inverse_accuracy_model_weights(
    forecast_metrics: pd.DataFrame,
    available_models: Sequence[str],
    ensemble_config: Mapping[str, object],
) -> Dict[str, float]:
    """Return nonnegative inverse-validation-WAPE ensemble weights."""
    validation = forecast_metrics[forecast_metrics["split"] == "validation"].copy()
    available = {str(model) for model in available_models}
    losses = {
        str(row.model_name): float(row.weighted_absolute_percentage_error)
        for row in validation.itertuples(index=False)
        if str(row.model_name) in available
    }
    return _inverse_metric_model_weights(losses, ensemble_config)


def _validation_operational_loss_by_model(
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
    scenario_weights: Mapping[str, float],
) -> Dict[str, float]:
    """Return validation normalized planning loss by model for ensemble weighting."""
    validation = _prepare_forecasts_with_safety_stock(forecasts, modeling_table, config, split_name="validation")
    if validation.empty:
        return {}
    validation_metrics = forecast_metrics[forecast_metrics["split"] == "validation"].copy()
    reference_model = str(validation_metrics.sort_values("weighted_absolute_percentage_error").iloc[0]["model_name"])
    planning_config = config.get("planning", {})
    stability_config = config.get("stability", {})
    max_plan_change_rate = float(stability_config.get("max_plan_change_rate", 0.20))

    records = []
    for model_name, group in validation.groupby("model_name"):
        model_group = group.sort_values(["series_id", "date"]).copy()
        actual = model_group["actual"].to_numpy(dtype=float)
        forecast = model_group["forecast"].to_numpy(dtype=float)
        signal = forecast_to_inventory_target(forecast, model_group["safety_stock"].to_numpy(dtype=float))
        inventory_cost = compute_holding_cost(
            signal,
            actual,
            holding_cost_rate=float(planning_config.get("holding_cost_rate", 1.0)),
        ) + compute_shortage_cost(
            signal,
            actual,
            shortage_cost_rate=float(planning_config.get("shortage_cost_rate", 5.0)),
        )
        volatility_total = 0.0
        execution_total = 0.0
        for _, series_group in model_group.assign(planning_signal=signal).groupby("series_id"):
            series_signal = series_group["planning_signal"].to_numpy(dtype=float)
            volatility_total += float(np.sum(compute_percentage_plan_change(series_signal)))
            capacity = compute_execution_capacity(series_signal, max_plan_change_rate=max_plan_change_rate)
            execution_total += float(np.sum(compute_execution_violation(series_signal, capacity)))
        records.append(
            {
                "model_name": str(model_name),
                "wape": weighted_absolute_percentage_error(actual, forecast),
                "inventory_cost": float(np.sum(inventory_cost)),
                "planning_volatility": volatility_total,
                "execution_penalty": execution_total,
            }
        )
    metrics = pd.DataFrame(records).set_index("model_name")
    if reference_model not in metrics.index:
        reference_model = str(metrics["wape"].idxmin())
    references = metrics.loc[reference_model].replace(0.0, np.nan)
    references = references.fillna(metrics.replace(0.0, np.nan).median()).fillna(1.0)
    loss = (
        float(scenario_weights.get("alpha_forecast", 1.0)) * metrics["wape"] / max(float(references["wape"]), 1e-8)
        + float(scenario_weights.get("beta_inventory", 1.0))
        * metrics["inventory_cost"]
        / max(float(references["inventory_cost"]), 1e-8)
        + float(scenario_weights.get("lambda_volatility", 0.5))
        * metrics["planning_volatility"]
        / max(float(references["planning_volatility"]), 1e-8)
        + float(scenario_weights.get("lambda_execution", 1.0))
        * metrics["execution_penalty"]
        / max(float(references["execution_penalty"]), 1e-8)
    )
    return {str(model_name): float(value) for model_name, value in loss.to_dict().items()}


def _inverse_metric_model_weights(
    losses: Mapping[str, float],
    ensemble_config: Mapping[str, object],
) -> Dict[str, float]:
    """Convert model losses to normalized nonnegative inverse-loss weights."""
    epsilon = float(ensemble_config.get("inverse_loss_epsilon", 1e-6))
    available = {str(model): max(float(value), epsilon) for model, value in losses.items() if np.isfinite(float(value))}
    if not available:
        return {}
    raw = {model: 1.0 / value for model, value in available.items()}
    return _normalize_model_weights(raw)


def _normalize_model_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """Normalize nonnegative model weights to sum to one."""
    cleaned = {str(model): max(float(value), 0.0) for model, value in weights.items() if np.isfinite(float(value))}
    total = float(sum(cleaned.values()))
    if total <= 0.0 and cleaned:
        equal = 1.0 / float(len(cleaned))
        return {model: equal for model in cleaned}
    if total <= 0.0:
        return {}
    return {model: value / total for model, value in cleaned.items()}


def _select_constrained_ensemble_weights(
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
    scenario_weights: Mapping[str, float],
    accuracy_weights: Mapping[str, float],
    operational_weights: Mapping[str, float],
    ensemble_config: Mapping[str, object],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Select nonnegative ensemble weights using a transparent validation grid."""
    blends = [float(value) for value in ensemble_config.get("constrained_candidate_blends", [0.0, 0.25, 0.50, 0.75, 1.0])]
    validation = _prepare_forecasts_with_safety_stock(forecasts, modeling_table, config, split_name="validation")
    folds = _chronological_date_folds(validation["date"], int(config.get("improved_feasibility_methods", {}).get("smoothing", {}).get("utility_validation_folds", 3)))
    candidate_records = []
    best_weights = _normalize_model_weights(operational_weights or accuracy_weights)
    best_score = np.inf
    best_blend = np.nan
    for blend in blends:
        weights = _blend_model_weights(accuracy_weights, operational_weights, blend)
        fold_losses = []
        for fold_dates in folds:
            if not fold_dates:
                continue
            evaluation = validation[validation["date"].isin(set(fold_dates))].copy()
            if evaluation.empty:
                continue
            reference_model = _best_model_from_global_losses(_global_losses_from_expected_losses(_expected_losses_from_prepared_forecasts(validation, config)), forecast_metrics)
            deployable_evaluation = drop_future_outcomes(evaluation)
            reference = _fixed_model_selection(deployable_evaluation, reference_model, "global_best_model")
            selected = _weighted_ensemble_selection(deployable_evaluation, weights, "feasibility_aware_ensemble_cv")
            fold_decisions = attach_actuals_for_evaluation(
                pd.concat([reference, selected], ignore_index=True),
                evaluation,
                key_columns=("date", "series_id"),
            )
            decisions = _evaluate_selected_decisions(fold_decisions, config)
            summary = _summarize_planning_utility(decisions, scenario_weights)
            summary, _, _ = add_normalized_planning_loss(
                summary,
                scenario_weights,
                reference_strategy="global_best_model",
                dataset_name="favorita",
                run_mode=_normalize_run_mode(config.get("project", {}).get("run_mode", "quick")),
                split_name="validation_cv",
            )
            value = summary.loc[summary["strategy"] == "feasibility_aware_ensemble_cv", "normalized_total_loss"]
            if not value.empty:
                fold_losses.append(float(value.iloc[0]))
        score = float(np.mean(fold_losses)) if fold_losses else np.inf
        candidate_records.append({"blend": blend, "validation_cv_normalized_loss": score, "fold_count": len(fold_losses)})
        if score < best_score or (np.isclose(score, best_score) and blend < best_blend):
            best_score = score
            best_weights = weights
            best_blend = blend
    return best_weights, {
        "selected_blend": float(best_blend) if np.isfinite(best_blend) else np.nan,
        "validation_cv_normalized_loss": float(best_score) if np.isfinite(best_score) else np.nan,
    }


def _blend_model_weights(
    accuracy_weights: Mapping[str, float],
    operational_weights: Mapping[str, float],
    accuracy_blend: float,
) -> Dict[str, float]:
    """Blend accuracy and operational weights while preserving simplex constraints."""
    models = sorted(set(accuracy_weights).union(set(operational_weights)))
    blended = {
        model: float(accuracy_blend) * float(accuracy_weights.get(model, 0.0))
        + (1.0 - float(accuracy_blend)) * float(operational_weights.get(model, 0.0))
        for model in models
    }
    return _normalize_model_weights(blended)


def _weighted_ensemble_selection(
    prepared_forecasts: pd.DataFrame,
    weights: Mapping[str, float],
    strategy: str,
) -> pd.DataFrame:
    """Return a nonnegative weighted ensemble forecast selection."""
    require_no_future_outcomes(prepared_forecasts, "_weighted_ensemble_selection")
    frame = prepared_forecasts.copy()
    normalized_weights = _normalize_model_weights(weights)
    available_models = sorted(frame["model_name"].unique())
    if not normalized_weights:
        normalized_weights = {str(model): 1.0 / float(len(available_models)) for model in available_models}
    frame["ensemble_weight"] = frame["model_name"].astype(str).map(normalized_weights).fillna(0.0)
    base_columns = ["date", "series_id", "family", "store_nbr", "split", "horizon", "safety_stock"]
    weighted = frame.copy()
    weighted["weighted_forecast"] = weighted["forecast"] * weighted["ensemble_weight"]
    ensemble = (
        weighted.groupby(base_columns, dropna=False)
        .agg(forecast=("weighted_forecast", "sum"), weight_sum=("ensemble_weight", "sum"))
        .reset_index()
    )
    ensemble["forecast"] = ensemble["forecast"] / ensemble["weight_sum"].replace(0.0, np.nan)
    ensemble["forecast"] = ensemble["forecast"].fillna(0.0)
    ensemble = ensemble.drop(columns=["weight_sum"])
    ensemble["model_name"] = strategy
    ensemble["selected_model"] = strategy
    ensemble["strategy"] = strategy
    return ensemble


def _weight_entropy(weights: Mapping[str, float]) -> float:
    """Return normalized entropy for a model-weight vector."""
    values = np.asarray(list(_normalize_model_weights(weights).values()), dtype=float)
    values = values[values > 0.0]
    if values.size <= 1:
        return 0.0
    entropy = -float(np.sum(values * np.log(values)))
    return entropy / float(np.log(values.size))


def _alpha_strategy_suffix(alpha: float) -> str:
    """Return a stable strategy suffix for an alpha value."""
    return "{:0.2f}".format(float(alpha)).replace(".", "_")


def _method_family_from_strategy(strategy: str) -> str:
    """Return a compact method family for improved-method outputs."""
    if strategy.startswith("feasibility_aware_smoothed"):
        return "FeasibilityAwareSmoothed"
    if strategy.startswith("feasibility_aware_ensemble"):
        return "FeasibilityAwareEnsemble"
    if strategy == "simple_ensemble":
        return "ReferenceEnsemble"
    if strategy == "oracle_realized_demand":
        return "Oracle"
    return "Baseline"


def _improved_method_output_columns(result: pd.DataFrame) -> pd.DataFrame:
    """Return the requested improved-method output columns in a stable order."""
    optional_columns = [
        "method_family",
        "method_variant",
        "smoothing_alpha",
        "ensemble_weight_basis",
        "ensemble_weight_entropy",
        "selected_blend",
        "validation_cv_normalized_loss",
    ]
    output_columns = [
        "run_mode",
        "scenario_name",
        "lambda_execution",
        "late_delivery_rate_anchor",
        "source",
        "fallback_used",
        "method_name",
        "method_family",
        "method_variant",
        "strategy",
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
        "rank_by_normalized_total_loss",
        "rank_by_WAPE",
        "rank_by_execution_penalty",
        "gap_to_oracle",
    ]
    for column in optional_columns:
        if column not in result.columns:
            result[column] = np.nan
    extra_columns = [
        "smoothing_alpha",
        "ensemble_weight_basis",
        "ensemble_weight_entropy",
        "selected_blend",
        "validation_cv_normalized_loss",
    ]
    table = result[output_columns + extra_columns].copy()
    return table.sort_values(["scenario_name", "rank_by_normalized_total_loss", "method_name"]).reset_index(drop=True)


def _build_improved_method_rankings(improved_methods: pd.DataFrame) -> pd.DataFrame:
    """Return objective-level rankings for the improved method comparison."""
    records = []
    objectives = [
        ("WAPE", "WAPE", True),
        ("inventory_cost", "inventory_cost", True),
        ("planning_volatility", "planning_volatility", True),
        ("execution_penalty", "execution_penalty", True),
        ("execution_violation_rate", "execution_violation_rate", True),
        ("model_switch_count", "model_switch_count", True),
        ("max_period_plan_change_pct", "max_period_plan_change_pct", True),
        ("normalized_total_loss", "normalized_total_loss", True),
        ("gap_to_oracle", "gap_to_oracle", True),
    ]
    for scenario_name, group in improved_methods.groupby("scenario_name", sort=False):
        for objective_name, metric_column, ascending in objectives:
            ranks = group[metric_column].rank(method="min", ascending=ascending).astype(int)
            for (_, row), rank in zip(group.iterrows(), ranks):
                records.append(
                    {
                        "scenario_name": scenario_name,
                        "lambda_execution": float(row["lambda_execution"]),
                        "objective_name": objective_name,
                        "method_name": row["method_name"],
                        "method_family": row["method_family"],
                        "strategy": row["strategy"],
                        "metric_value": float(row[metric_column]),
                        "objective_rank": int(rank),
                    }
                )
    return pd.DataFrame(records).sort_values(["scenario_name", "objective_name", "objective_rank", "method_name"])


def _build_strategy_rank_reordering_by_weight(scenario_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize normalized-loss rank changes across execution-risk scenarios."""
    baseline = scenario_results[scenario_results["scenario_name"] == "baseline"][
        ["strategy", "rank_by_normalized_total_loss"]
    ].rename(columns={"rank_by_normalized_total_loss": "baseline_rank_by_normalized_total_loss"})
    table = scenario_results.merge(baseline, on="strategy", how="left")
    table["rank_change_vs_baseline"] = (
        table["rank_by_normalized_total_loss"] - table["baseline_rank_by_normalized_total_loss"]
    )
    return table[
        [
            "scenario_name",
            "lambda_execution",
            "method_name",
            "strategy",
            "rank_by_normalized_total_loss",
            "baseline_rank_by_normalized_total_loss",
            "rank_change_vs_baseline",
            "normalized_total_loss",
        ]
    ].sort_values(["scenario_name", "rank_by_normalized_total_loss", "method_name"])


def _build_model_ranking_by_objective(scenario_results: pd.DataFrame) -> pd.DataFrame:
    """Return ranks by objective for every execution-risk scenario."""
    records = []
    for scenario_name, group in scenario_results.groupby("scenario_name", sort=False):
        for objective_name, metric_column, ascending in RANKING_OBJECTIVES:
            ranks = group[metric_column].rank(method="min", ascending=ascending).astype(int)
            for (_, row), rank in zip(group.iterrows(), ranks):
                records.append(
                    {
                        "scenario_name": scenario_name,
                        "lambda_execution": float(row["lambda_execution"]),
                        "objective_name": objective_name,
                        "method_name": row["method_name"],
                        "strategy": row["strategy"],
                        "metric_value": float(row[metric_column]),
                        "objective_rank": int(rank),
                    }
                )
    return pd.DataFrame(records).sort_values(["scenario_name", "objective_name", "objective_rank", "method_name"])


def _compact_objective_ranking_table(objective_ranking: pd.DataFrame) -> pd.DataFrame:
    """Return a compact paper table with the top method per objective."""
    compact = objective_ranking[objective_ranking["objective_rank"] == 1].copy()
    return compact[
        [
            "scenario_name",
            "lambda_execution",
            "objective_name",
            "method_name",
            "metric_value",
            "objective_rank",
        ]
    ].sort_values(["scenario_name", "objective_name", "method_name"])


def _save_output_and_paper_table(
    data: pd.DataFrame,
    output_table_dir: Path,
    paper_table_dir: Path,
    output_stem: str,
    paper_stem: str,
    caption: str,
    label: str,
    paper_data: Optional[pd.DataFrame] = None,
    resize_to_textwidth: bool = False,
) -> Path:
    """Save a raw CSV table and a LaTeX-ready paper table."""
    output_table_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_table_dir / "{}.csv".format(output_stem), index=False)
    exported = export_summary_table(
        data=data if paper_data is None else paper_data,
        table_name=paper_stem,
        output_dir=paper_table_dir,
        caption=caption,
        label=label,
        numeric_precision=3,
        resize_to_textwidth=resize_to_textwidth,
    )
    return exported["tex"]


def _run_feasibility_analyses(
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
    output_table_dir: Path,
    output_figure_dir: Path,
    paper_table_dir: Path,
    paper_figure_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[Path], List[Path]]:
    """Run Favorita feasibility stress tests and Pareto analysis.

    These analyses intentionally report the tradeoff surface instead of tuning
    weights until one selector wins. If execution infrastructure is strong,
    accuracy-only planning can be rational. If execution infrastructure is
    weak, feasibility-aware planning becomes more important. The data should
    show where those regimes appear.
    """
    logger.info("Running Favorita feasibility stress tests and Pareto analysis.")
    decisions = _build_decision_outputs(forecasts, modeling_table, forecast_metrics, config, logger)
    weight_results = _run_weight_sensitivity_analysis(forecasts, modeling_table, forecast_metrics, config, logger)
    capacity_results = _run_execution_capacity_stress_test(forecasts, modeling_table, forecast_metrics, config, logger)
    pareto_summary = _build_pareto_summary(decisions, config)

    weight_results.to_csv(output_table_dir / "weight_sensitivity_results.csv", index=False)
    capacity_results.to_csv(output_table_dir / "execution_capacity_stress_test.csv", index=False)
    pareto_summary.to_csv(output_table_dir / "pareto_summary.csv", index=False)

    table_assets = _export_feasibility_latex_tables(
        weight_results=weight_results,
        capacity_results=capacity_results,
        pareto_summary=pareto_summary,
        paper_table_dir=paper_table_dir,
    )
    figure_assets = _make_feasibility_figures(
        weight_results=weight_results,
        capacity_results=capacity_results,
        pareto_summary=pareto_summary,
        output_figure_dir=output_figure_dir,
        paper_figure_dir=paper_figure_dir,
    )
    logger.info("Saved Favorita feasibility tables and figures.")
    return table_assets, figure_assets


def _run_weight_sensitivity_analysis(
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Evaluate planning utility under a grid of feasibility weights."""
    analysis_config = config.get("feasibility_analysis", {}).get("weight_sensitivity", {})
    lambda_volatility_values = analysis_config.get("lambda_volatility_values", [0.5, 25.0, 100.0])
    lambda_switch_values = analysis_config.get("lambda_switch_values", [0.5, 10.0])
    lambda_execution_values = analysis_config.get("lambda_execution_values", [0.5, 1.0, 2.0, 5.0, 10.0])
    base_weights = config.get("planning_loss_weights", {})

    rows: List[pd.DataFrame] = []
    for scenario_index, (lambda_volatility, lambda_switch, lambda_execution) in enumerate(
        product(lambda_volatility_values, lambda_switch_values, lambda_execution_values),
        start=1,
    ):
        scenario_id = "weights_{:02d}".format(scenario_index)
        scenario_weights = dict(base_weights)
        scenario_weights.update(
            {
                "lambda_volatility": float(lambda_volatility),
                "lambda_switch": float(lambda_switch),
                "lambda_execution": float(lambda_execution),
            }
        )
        scenario_config = _copy_config_with_updates(config, loss_weights=scenario_weights)
        decisions = _build_decision_outputs(forecasts, modeling_table, forecast_metrics, scenario_config, logger)
        summary = _summarize_planning_utility(decisions, scenario_weights)
        summary, _, _ = add_normalized_planning_loss(
            summary,
            scenario_weights,
            dataset_name="favorita",
            run_mode=_normalize_run_mode(config.get("project", {}).get("run_mode", "quick")),
            split_name="test",
        )
        summary["scenario_id"] = scenario_id
        summary["lambda_volatility"] = float(lambda_volatility)
        summary["lambda_switch"] = float(lambda_switch)
        summary["lambda_execution"] = float(lambda_execution)
        summary["alpha_forecast"] = float(scenario_weights.get("alpha_forecast", 1.0))
        summary["beta_inventory"] = float(scenario_weights.get("beta_inventory", 1.0))
        rows.append(summary)
    result = pd.concat(rows, ignore_index=True)
    result["is_best_total_loss"] = result["normalized_total_loss"] == result.groupby("scenario_id")["normalized_total_loss"].transform("min")
    global_losses = result[result["strategy"] == "global_best_model"][
        ["scenario_id", "normalized_total_loss"]
    ].rename(columns={"normalized_total_loss": "global_best_total_loss"})
    result = result.merge(global_losses, on="scenario_id", how="left")
    result["loss_gap_to_global_best"] = result["normalized_total_loss"] - result["global_best_total_loss"]
    return result.sort_values(["scenario_id", "normalized_total_loss", "strategy"])


def _run_execution_capacity_stress_test(
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Evaluate strategies when execution infrastructure becomes tighter.

    Execution capacity stress testing is central to the planning-infrastructure
    gap: the same forecast can be easy to execute in a high-capacity operation
    and infeasible when planners, suppliers, systems, or stores can absorb only
    small period-to-period changes.
    """
    scenario_rates = _execution_capacity_scenarios(config)
    rows: List[pd.DataFrame] = []
    for scenario_name, max_plan_change_rate in scenario_rates.items():
        scenario_config = _copy_config_with_updates(config, max_plan_change_rate=float(max_plan_change_rate))
        decisions = _build_decision_outputs(forecasts, modeling_table, forecast_metrics, scenario_config, logger)
        summary = _summarize_planning_utility(decisions, scenario_config.get("planning_loss_weights", {}))
        summary, _, _ = add_normalized_planning_loss(
            summary,
            scenario_config.get("planning_loss_weights", {}),
            dataset_name="favorita",
            run_mode=_normalize_run_mode(config.get("project", {}).get("run_mode", "quick")),
            split_name="test",
        )
        summary["capacity_scenario"] = scenario_name
        summary["max_plan_change_rate"] = float(max_plan_change_rate)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True).sort_values(["capacity_scenario", "normalized_total_loss", "strategy"])


def _build_pareto_summary(decisions: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    """Return a Pareto-style summary over accuracy, cost, stability, and execution."""
    summary = _summarize_planning_utility(decisions, config.get("planning_loss_weights", {}))
    summary, _, _ = add_normalized_planning_loss(
        summary,
        config.get("planning_loss_weights", {}),
        dataset_name="favorita",
        run_mode=_normalize_run_mode(config.get("project", {}).get("run_mode", "quick")),
        split_name="test",
    )
    objective_columns = [
        "weighted_absolute_percentage_error",
        "total_inventory_cost",
        "planning_signal_volatility_total",
        "execution_adaptation_penalty_total",
        "execution_violation_rate",
    ]
    summary["pareto_efficient"] = _mark_pareto_efficient(summary, objective_columns)
    summary["dominated"] = ~summary["pareto_efficient"]
    return summary.sort_values(["pareto_efficient", "normalized_total_loss", "strategy"], ascending=[False, True, True])


def _mark_pareto_efficient(data: pd.DataFrame, objective_columns: Sequence[str]) -> pd.Series:
    """Mark rows that are not dominated on all minimization objectives."""
    objectives = data[list(objective_columns)].to_numpy(dtype=float)
    efficient = np.ones(objectives.shape[0], dtype=bool)
    for index, candidate in enumerate(objectives):
        other = np.delete(objectives, index, axis=0)
        if other.size == 0:
            continue
        dominated = np.any(np.all(other <= candidate, axis=1) & np.any(other < candidate, axis=1))
        efficient[index] = not dominated
    return pd.Series(efficient, index=data.index)


def _copy_config_with_updates(
    config: Mapping[str, object],
    loss_weights: Optional[Mapping[str, float]] = None,
    max_plan_change_rate: Optional[float] = None,
) -> Dict[str, object]:
    """Return a deep-copied config with selected planning assumptions changed."""
    updated = copy.deepcopy(dict(config))
    if loss_weights is not None:
        updated.setdefault("planning_loss_weights", {}).update(dict(loss_weights))
    if max_plan_change_rate is not None:
        updated.setdefault("stability", {})["max_plan_change_rate"] = float(max_plan_change_rate)
    return updated


def _execution_capacity_scenarios(config: Mapping[str, object]) -> Dict[str, float]:
    """Return ordered execution capacity scenarios."""
    default_scenarios = {
        "high_capacity": 0.40,
        "medium_capacity": 0.20,
        "low_capacity": 0.10,
        "severe_constraint": 0.05,
    }
    configured = config.get("feasibility_analysis", {}).get("execution_capacity_scenarios", default_scenarios)
    return {name: float(value) for name, value in configured.items()}


def _export_feasibility_latex_tables(
    weight_results: pd.DataFrame,
    capacity_results: pd.DataFrame,
    pareto_summary: pd.DataFrame,
    paper_table_dir: Path,
) -> List[Path]:
    """Export compact feasibility-analysis tables for manuscript inclusion."""
    weight_table = _compact_weight_sensitivity_table(weight_results)
    capacity_table = _compact_strategy_table(
        capacity_results,
        group_column="capacity_scenario",
        include_strategies=PAPER_STRATEGY_ORDER,
    )
    pareto_table = _with_strategy_display_labels(
        pareto_summary[
            [
                "strategy",
                "weighted_absolute_percentage_error",
                "total_inventory_cost",
                "planning_signal_volatility_total",
                "execution_adaptation_penalty_total",
                "execution_violation_rate",
                "pareto_efficient",
            ]
        ].copy()
    )

    outputs = []
    outputs.append(
        export_summary_table(
            data=weight_table,
            table_name="weight_sensitivity_results",
            output_dir=paper_table_dir,
            caption="Favorita weight sensitivity summary for feasibility-aware planning.",
            label="tab:weight-sensitivity-results",
            numeric_precision=3,
            column_renames={
                "scenario_id": "Scenario",
                "lambda_volatility": "Volatility Weight",
                "lambda_switch": "Switch Weight",
                "lambda_execution": "Execution Weight",
                "best_strategy": "Best Strategy",
                "global_best_loss": "Global Best Normalized Loss",
                "feasibility_aware_loss": "Feasibility-Aware Normalized Loss",
                "stability_aware_loss": "Stability-Aware Normalized Loss",
                "moving_average_loss": "Moving Avg. Normalized Loss",
            },
            resize_to_textwidth=True,
        )["tex"]
    )
    outputs.append(
        export_summary_table(
            data=capacity_table,
            table_name="execution_capacity_stress_test",
            output_dir=paper_table_dir,
            caption="Favorita execution capacity stress-test summary.",
            label="tab:execution-capacity-stress-test",
            numeric_precision=3,
            column_renames={
                "capacity_scenario": "Capacity Scenario",
                "strategy": "Strategy",
                "weighted_absolute_percentage_error": "WAPE",
                "total_inventory_cost": "Inventory Cost",
                "planning_signal_volatility_total": "Volatility Total",
                "execution_adaptation_penalty_total": "Execution Penalty",
                "execution_violation_rate": "Violation Rate",
                "normalized_total_loss": "Normalized Total Loss",
                "total_planning_loss": "Raw Total Loss",
                "service_level": "Service Level",
            },
            resize_to_textwidth=True,
        )["tex"]
    )
    outputs.append(
        export_summary_table(
            data=pareto_table,
            table_name="pareto_summary",
            output_dir=paper_table_dir,
            caption="Favorita Pareto summary over accuracy, cost, stability, and execution.",
            label="tab:pareto-summary",
            numeric_precision=3,
            column_renames={
                "strategy": "Strategy",
                "weighted_absolute_percentage_error": "WAPE",
                "total_inventory_cost": "Inventory Cost",
                "planning_signal_volatility_total": "Volatility Total",
                "execution_adaptation_penalty_total": "Execution Penalty",
                "execution_violation_rate": "Violation Rate",
                "pareto_efficient": "Pareto Efficient",
            },
            resize_to_textwidth=True,
        )["tex"]
    )
    return outputs


def _compact_weight_sensitivity_table(weight_results: pd.DataFrame) -> pd.DataFrame:
    """Return one manuscript row per weight scenario."""
    records = []
    for scenario_id, group in weight_results.groupby("scenario_id"):
        by_strategy = group.set_index("strategy")
        best_row = group.sort_values("normalized_total_loss").iloc[0]
        records.append(
            {
                "scenario_id": scenario_id,
                "lambda_volatility": float(best_row["lambda_volatility"]),
                "lambda_switch": float(best_row["lambda_switch"]),
                "lambda_execution": float(best_row["lambda_execution"]),
                "best_strategy": _short_strategy_label(best_row["strategy"]),
                "global_best_loss": _strategy_metric(by_strategy, "global_best_model", "normalized_total_loss"),
                "feasibility_aware_loss": _strategy_metric(by_strategy, "feasibility_aware_selector", "normalized_total_loss"),
                "stability_aware_loss": _strategy_metric(by_strategy, "stability_aware_selector", "normalized_total_loss"),
                "moving_average_loss": _strategy_metric(by_strategy, "individual_moving_average", "normalized_total_loss"),
            }
        )
    return pd.DataFrame(records)


def _compact_strategy_table(data: pd.DataFrame, group_column: str, include_strategies: Sequence[str]) -> pd.DataFrame:
    """Return compact manuscript rows for key strategies."""
    table = data[data["strategy"].isin(include_strategies)].copy()
    table = table[
        [
            group_column,
            "strategy",
            "weighted_absolute_percentage_error",
            "total_inventory_cost",
            "planning_signal_volatility_total",
            "execution_adaptation_penalty_total",
            "execution_violation_rate",
            "normalized_total_loss",
            "total_planning_loss",
            "service_level",
        ]
    ]
    table = _with_strategy_display_labels(table)
    if group_column == "capacity_scenario":
        scenario_order = {name: index for index, name in enumerate(_execution_capacity_scenarios({}).keys())}
        table["_scenario_order"] = table[group_column].map(scenario_order)
        table = table.sort_values(["_scenario_order", "normalized_total_loss"]).drop(columns=["_scenario_order"])
        return table
    return table.sort_values([group_column, "normalized_total_loss"])


def _strategy_metric(by_strategy: pd.DataFrame, strategy: str, metric: str) -> float:
    """Return a strategy metric from an indexed summary table."""
    if strategy not in by_strategy.index:
        return float("nan")
    return float(by_strategy.loc[strategy, metric])


def _export_latex_tables(
    forecast_metrics: pd.DataFrame,
    inventory_metrics: pd.DataFrame,
    stability_metrics: pd.DataFrame,
    planning_utility: pd.DataFrame,
    paper_table_dir: Path = Path("paper/tables"),
) -> List[Path]:
    """Export core Favorita result tables into paper/tables."""
    outputs = []
    forecast_table = forecast_metrics[
        [
            "split",
            "model_name",
            "row_count",
            "mean_absolute_error",
            "root_mean_squared_error",
            "weighted_absolute_percentage_error",
        ]
    ].copy()
    forecast_table["model_name"] = forecast_table["model_name"].map(_model_display_label)

    inventory_table = _with_strategy_display_labels(
        inventory_metrics[
            [
                "strategy",
                "total_inventory_cost",
                "total_inventory_cost_per_demand_unit",
                "service_level",
                "service_level_hit_rate",
            ]
        ]
    )
    stability_table = _with_strategy_display_labels(
        stability_metrics[
            [
                "strategy",
                "mean_plan_change_pct",
                "large_jump_count",
                "model_switch_count",
                "execution_violation_count",
                "execution_violation_rate",
            ]
        ]
    )
    utility_table = _with_strategy_display_labels(
        planning_utility[
            [
                "strategy",
                "selected_model_count",
                "weighted_absolute_percentage_error",
                "total_inventory_cost",
                "planning_signal_volatility_total",
                "execution_adaptation_penalty_total",
                "total_planning_loss",
            ]
        ]
    )

    outputs.append(
        export_summary_table(
            data=forecast_table,
            table_name="favorita_forecast_metrics_table",
            output_dir=paper_table_dir,
            caption="Favorita forecast accuracy metrics from the minimal pipeline.",
            label="tab:favorita-forecast-metrics",
            numeric_precision=3,
            column_renames={
                "split": "Split",
                "model_name": "Model",
                "row_count": "Rows",
                "mean_absolute_error": "MAE",
                "root_mean_squared_error": "RMSE",
                "weighted_absolute_percentage_error": "WAPE",
            },
            resize_to_textwidth=True,
        )["tex"]
    )
    outputs.append(
        export_summary_table(
            data=inventory_table,
            table_name="favorita_inventory_metrics_table",
            output_dir=paper_table_dir,
            caption="Favorita inventory and service metrics from the minimal pipeline.",
            label="tab:favorita-inventory-metrics",
            numeric_precision=3,
            column_renames={
                "strategy": "Strategy",
                "total_inventory_cost": "Total Inventory Cost",
                "total_inventory_cost_per_demand_unit": "Inventory Cost Per Demand Unit",
                "service_level": "Service Level",
                "service_level_hit_rate": "Service Hit Rate",
            },
            resize_to_textwidth=True,
        )["tex"]
    )
    outputs.append(
        export_summary_table(
            data=stability_table,
            table_name="favorita_stability_metrics_table",
            output_dir=paper_table_dir,
            caption="Favorita planning stability metrics from the minimal pipeline.",
            label="tab:favorita-stability-metrics",
            numeric_precision=3,
            column_renames={
                "strategy": "Strategy",
                "mean_plan_change_pct": "Mean Plan Change",
                "large_jump_count": "Large Jumps",
                "model_switch_count": "Model Switches",
                "execution_violation_count": "Execution Violations",
                "execution_violation_rate": "Execution Violation Rate",
            },
            resize_to_textwidth=True,
        )["tex"]
    )
    outputs.append(
        export_summary_table(
            data=utility_table,
            table_name="favorita_planning_utility_table",
            output_dir=paper_table_dir,
            caption="Favorita planning utility metrics from the minimal pipeline.",
            label="tab:favorita-planning-utility",
            numeric_precision=3,
            column_renames={
                "strategy": "Strategy",
                "selected_model_count": "Selected Models",
                "mean_absolute_error": "MAE",
                "weighted_absolute_percentage_error": "WAPE",
                "total_inventory_cost": "Total Inventory Cost",
                "planning_signal_volatility_total": "Volatility Total",
                "execution_adaptation_penalty_total": "Execution Penalty",
                "total_planning_loss": "Total Planning Loss",
                "service_level": "Service Level",
            },
            resize_to_textwidth=True,
        )["tex"]
    )
    return outputs


def _with_strategy_display_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with compact English strategy labels for paper tables."""
    frame = data.copy()
    frame["strategy"] = frame["strategy"].map(_short_strategy_label)
    return frame


def _model_display_label(model_name: str) -> str:
    """Return a compact English model label for paper tables."""
    labels = {
        "naive_last_value": "Naive",
        "seasonal_naive": "Seasonal",
        "moving_average": "Moving Avg.",
        "exponential_smoothing": "Exp. Smooth",
        "global_lightgbm": "LightGBM",
        "global_xgboost": "XGBoost",
        "global_sklearn": "sklearn",
    }
    return labels.get(model_name, str(model_name).replace("_", " ").title())


def _make_figures(
    planning_utility: pd.DataFrame,
    stability_metrics: pd.DataFrame,
    decisions: pd.DataFrame,
    output_figure_dir: Path,
    paper_figure_dir: Path,
) -> List[Path]:
    """Create PNG figures for quick viewing and PDF figures for LaTeX."""
    apply_paper_style()
    output_figure_dir.mkdir(parents=True, exist_ok=True)
    paper_figure_dir.mkdir(parents=True, exist_ok=True)

    merged = planning_utility.merge(stability_metrics, on="strategy", how="left", suffixes=("", "_stability"))
    _save_scatter_figure(
        merged,
        x_column="weighted_absolute_percentage_error",
        y_column="total_inventory_cost",
        title="Accuracy vs. Inventory Cost",
        x_label="Weighted Absolute Percentage Error",
        y_label="Total Inventory Cost",
        png_path=output_figure_dir / "favorita_accuracy_vs_inventory_cost.png",
        pdf_path=paper_figure_dir / "favorita_accuracy_vs_inventory_cost.pdf",
    )
    _save_scatter_figure(
        merged,
        x_column="weighted_absolute_percentage_error",
        y_column="mean_plan_change_pct",
        title="Accuracy vs. Planning Volatility",
        x_label="Weighted Absolute Percentage Error",
        y_label="Mean Planning Signal Change",
        png_path=output_figure_dir / "favorita_accuracy_vs_volatility.png",
        pdf_path=paper_figure_dir / "favorita_accuracy_vs_volatility.pdf",
    )
    _save_example_planning_signal(
        decisions,
        png_path=output_figure_dir / "favorita_example_planning_signal.png",
        pdf_path=paper_figure_dir / "favorita_example_planning_signal.pdf",
    )
    return [
        paper_figure_dir / "favorita_accuracy_vs_inventory_cost.pdf",
        paper_figure_dir / "favorita_accuracy_vs_volatility.pdf",
        paper_figure_dir / "favorita_example_planning_signal.pdf",
    ]


def _make_feasibility_figures(
    weight_results: pd.DataFrame,
    capacity_results: pd.DataFrame,
    pareto_summary: pd.DataFrame,
    output_figure_dir: Path,
    paper_figure_dir: Path,
) -> List[Path]:
    """Create feasibility tradeoff figures for outputs and the LaTeX paper."""
    apply_paper_style()
    output_figure_dir.mkdir(parents=True, exist_ok=True)
    paper_figure_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = []
    base_weight_slice = _weight_plot_slice(weight_results)
    _save_weight_sensitivity_plot(
        base_weight_slice,
        metric_column="normalized_total_loss",
        y_label="Normalized Total Loss",
        title="Weight Sensitivity: Normalized Loss",
        png_path=output_figure_dir / "plot_weight_sensitivity_total_loss.png",
        pdf_path=paper_figure_dir / "plot_weight_sensitivity_total_loss.pdf",
    )
    figure_paths.append(paper_figure_dir / "plot_weight_sensitivity_total_loss.pdf")
    _save_weight_sensitivity_plot(
        base_weight_slice,
        metric_column="execution_adaptation_penalty_total",
        y_label="Execution Penalty",
        title="Weight Sensitivity: Execution Penalty",
        png_path=output_figure_dir / "plot_weight_sensitivity_execution_penalty.png",
        pdf_path=paper_figure_dir / "plot_weight_sensitivity_execution_penalty.pdf",
    )
    figure_paths.append(paper_figure_dir / "plot_weight_sensitivity_execution_penalty.pdf")

    _save_capacity_stress_plot(
        capacity_results,
        metric_column="normalized_total_loss",
        y_label="Normalized Total Loss",
        title="Execution Capacity vs. Normalized Loss",
        png_path=output_figure_dir / "execution_capacity_vs_total_loss.png",
        pdf_path=paper_figure_dir / "execution_capacity_vs_total_loss.pdf",
    )
    figure_paths.append(paper_figure_dir / "execution_capacity_vs_total_loss.pdf")
    _save_capacity_stress_plot(
        capacity_results,
        metric_column="execution_violation_rate",
        y_label="Execution Violation Rate",
        title="Execution Capacity vs. Violation Rate",
        png_path=output_figure_dir / "execution_capacity_vs_violation_rate.png",
        pdf_path=paper_figure_dir / "execution_capacity_vs_violation_rate.pdf",
    )
    figure_paths.append(paper_figure_dir / "execution_capacity_vs_violation_rate.pdf")
    _save_capacity_stress_plot(
        capacity_results,
        metric_column="total_inventory_cost",
        y_label="Inventory Cost",
        title="Execution Capacity vs. Inventory Cost",
        png_path=output_figure_dir / "execution_capacity_vs_inventory_cost.png",
        pdf_path=paper_figure_dir / "execution_capacity_vs_inventory_cost.pdf",
    )
    figure_paths.append(paper_figure_dir / "execution_capacity_vs_inventory_cost.pdf")

    _save_pareto_plot(
        pareto_summary,
        x_column="weighted_absolute_percentage_error",
        y_column="execution_adaptation_penalty_total",
        x_label="Weighted Absolute Percentage Error",
        y_label="Execution Penalty",
        title="Pareto Tradeoff: Accuracy and Execution",
        png_path=output_figure_dir / "pareto_accuracy_vs_execution_penalty.png",
        pdf_path=paper_figure_dir / "pareto_accuracy_vs_execution_penalty.pdf",
    )
    figure_paths.append(paper_figure_dir / "pareto_accuracy_vs_execution_penalty.pdf")
    _save_pareto_plot(
        pareto_summary,
        x_column="weighted_absolute_percentage_error",
        y_column="planning_signal_volatility_total",
        x_label="Weighted Absolute Percentage Error",
        y_label="Planning Volatility",
        title="Pareto Tradeoff: Accuracy and Volatility",
        png_path=output_figure_dir / "pareto_accuracy_vs_planning_volatility.png",
        pdf_path=paper_figure_dir / "pareto_accuracy_vs_planning_volatility.pdf",
    )
    figure_paths.append(paper_figure_dir / "pareto_accuracy_vs_planning_volatility.pdf")
    _save_pareto_plot(
        pareto_summary,
        x_column="total_inventory_cost",
        y_column="execution_adaptation_penalty_total",
        x_label="Inventory Cost",
        y_label="Execution Penalty",
        title="Pareto Tradeoff: Inventory and Execution",
        png_path=output_figure_dir / "pareto_inventory_cost_vs_execution_penalty.png",
        pdf_path=paper_figure_dir / "pareto_inventory_cost_vs_execution_penalty.pdf",
    )
    figure_paths.append(paper_figure_dir / "pareto_inventory_cost_vs_execution_penalty.pdf")
    return figure_paths


def _make_improved_feasibility_figures(
    improved_methods: pd.DataFrame,
    smoothing_alpha: pd.DataFrame,
    ensemble_comparison: pd.DataFrame,
    output_figure_dir: Path,
    paper_figure_dir: Path,
) -> List[Path]:
    """Create figures for improved feasibility-aware strategies."""
    apply_paper_style()
    output_figure_dir.mkdir(parents=True, exist_ok=True)
    paper_figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = []

    _save_improved_accuracy_execution_plot(
        improved_methods,
        output_figure_dir / "favorita_improved_methods_accuracy_vs_execution_penalty.png",
        paper_figure_dir / "favorita_improved_methods_accuracy_vs_execution_penalty.pdf",
    )
    figure_paths.append(paper_figure_dir / "favorita_improved_methods_accuracy_vs_execution_penalty.pdf")

    _save_smoothing_alpha_tradeoff_plot(
        smoothing_alpha,
        output_figure_dir / "favorita_smoothing_alpha_tradeoff.png",
        paper_figure_dir / "favorita_smoothing_alpha_tradeoff.pdf",
    )
    figure_paths.append(paper_figure_dir / "favorita_smoothing_alpha_tradeoff.pdf")

    _save_ensemble_tradeoff_plot(
        ensemble_comparison,
        output_figure_dir / "favorita_feasibility_ensemble_tradeoff.png",
        paper_figure_dir / "favorita_feasibility_ensemble_tradeoff.pdf",
    )
    figure_paths.append(paper_figure_dir / "favorita_feasibility_ensemble_tradeoff.pdf")

    _save_improved_rank_reordering_plot(
        improved_methods,
        output_figure_dir / "favorita_improved_method_rank_reordering.png",
        paper_figure_dir / "favorita_improved_method_rank_reordering.pdf",
    )
    figure_paths.append(paper_figure_dir / "favorita_improved_method_rank_reordering.pdf")

    _save_gap_to_oracle_plot(
        improved_methods,
        output_figure_dir / "favorita_gap_to_oracle_by_method.png",
        paper_figure_dir / "favorita_gap_to_oracle_by_method.pdf",
    )
    figure_paths.append(paper_figure_dir / "favorita_gap_to_oracle_by_method.pdf")
    return figure_paths


def _improved_plot_strategies() -> List[str]:
    """Return a compact strategy subset for readable improved-method figures."""
    return [
        "global_best_model",
        "simple_ensemble",
        "feasibility_aware_selector",
        "feasibility_aware_smoothed_utility_alpha",
        "feasibility_aware_ensemble_constrained",
        "best_stability_model",
        "oracle_realized_demand",
    ]


def _save_improved_accuracy_execution_plot(data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save improved-method accuracy versus execution-penalty tradeoff."""
    plot_data = data[data["strategy"].isin(_improved_plot_strategies())].copy()
    fig, ax = plt.subplots(figsize=(7.4, 4.45))
    for index, strategy in enumerate(_improved_plot_strategies()):
        strategy_data = plot_data[plot_data["strategy"] == strategy].sort_values("lambda_execution")
        if strategy_data.empty:
            continue
        ax.plot(
            strategy_data["WAPE"],
            strategy_data["execution_penalty"],
            marker=strategy_marker(strategy),
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.55,
            color=strategy_color(strategy, index),
            linestyle=strategy_linestyle(strategy),
            linewidth=1.6,
            label=_short_strategy_label(strategy),
        )
    format_axis(
        ax,
        x_label="Weighted Absolute Percentage Error",
        y_label="Execution Penalty",
        grid_axis="both",
    )
    place_legend(ax, columns=3)
    save_paper_figure(fig, png_path, pdf_path)


def _save_smoothing_alpha_tradeoff_plot(data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save alpha sensitivity for smoothed feasibility-aware planning."""
    plot_data = data.dropna(subset=["smoothing_alpha", "normalized_total_loss"]).copy()
    fixed = plot_data[plot_data["method_variant"].isin(["fixed_alpha_smoothing", "utility_based_alpha", "scenario_based_alpha"])].copy()
    fig, ax = plt.subplots(figsize=(7.4, 4.45))
    if fixed.empty:
        ax.text(0.5, 0.5, "Smoothing alpha results are unavailable.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        save_paper_figure(fig, png_path, pdf_path)
        return
    scenario_order = list(dict.fromkeys(fixed["scenario_name"].tolist()))
    for index, scenario_name in enumerate(scenario_order):
        scenario_data = fixed[fixed["scenario_name"] == scenario_name].sort_values("smoothing_alpha")
        ax.plot(
            scenario_data["smoothing_alpha"],
            scenario_data["normalized_total_loss"],
            marker="o",
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.55,
            linewidth=1.7,
            color=strategy_color(str(scenario_name), index),
            label=str(scenario_name).replace("_", " ").title(),
        )
    format_axis(ax, x_label="Smoothing Alpha", y_label="Normalized Total Loss", grid_axis="both")
    place_legend(ax, columns=3)
    save_paper_figure(fig, png_path, pdf_path)


def _save_ensemble_tradeoff_plot(data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save feasibility-aware ensemble tradeoff plot."""
    plot_data = data.copy()
    fig, ax = plt.subplots(figsize=(7.4, 4.45))
    if plot_data.empty:
        ax.text(0.5, 0.5, "Ensemble comparison results are unavailable.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        save_paper_figure(fig, png_path, pdf_path)
        return
    strategy_order = [
        "simple_ensemble",
        "feasibility_aware_ensemble_inverse_accuracy",
        "feasibility_aware_ensemble_inverse_operational_loss",
        "feasibility_aware_ensemble_constrained",
    ]
    for index, strategy in enumerate(strategy_order):
        strategy_data = plot_data[plot_data["strategy"] == strategy].sort_values("lambda_execution")
        if strategy_data.empty:
            continue
        ax.plot(
            strategy_data["WAPE"],
            strategy_data["execution_penalty"],
            marker=strategy_marker(strategy),
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.55,
            color=strategy_color(strategy, index),
            linestyle=strategy_linestyle(strategy),
            linewidth=1.7,
            label=_short_strategy_label(strategy),
        )
    format_axis(
        ax,
        x_label="Weighted Absolute Percentage Error",
        y_label="Execution Penalty",
        grid_axis="both",
    )
    place_legend(ax, columns=2)
    save_paper_figure(fig, png_path, pdf_path)


def _save_improved_rank_reordering_plot(data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save improved-method rank changes across DataCo-informed scenarios."""
    plot_data = data[data["strategy"].isin(_improved_plot_strategies())].copy()
    fig, ax = plt.subplots(figsize=(7.4, 4.45))
    for index, strategy in enumerate(_improved_plot_strategies()):
        strategy_data = plot_data[plot_data["strategy"] == strategy].sort_values("lambda_execution")
        if strategy_data.empty:
            continue
        ax.plot(
            strategy_data["lambda_execution"],
            strategy_data["rank_by_normalized_total_loss"],
            marker=strategy_marker(strategy),
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.55,
            color=strategy_color(strategy, index),
            linestyle=strategy_linestyle(strategy),
            linewidth=1.7,
            label=_short_strategy_label(strategy),
        )
    ax.invert_yaxis()
    format_axis(ax, x_label="Lambda Execution", y_label="Rank by Normalized Total Loss")
    place_legend(ax, columns=3)
    save_paper_figure(fig, png_path, pdf_path)


def _save_gap_to_oracle_plot(data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save gap to non-deployable oracle by method and scenario."""
    plot_data = data[data["strategy"].isin(_improved_plot_strategies())].copy()
    scenario_order = list(dict.fromkeys(plot_data["scenario_name"].tolist()))
    positions = {scenario: index for index, scenario in enumerate(scenario_order)}
    fig, ax = plt.subplots(figsize=(7.4, 4.45))
    for index, strategy in enumerate(_improved_plot_strategies()):
        strategy_data = plot_data[plot_data["strategy"] == strategy].copy()
        if strategy_data.empty:
            continue
        strategy_data["scenario_position"] = strategy_data["scenario_name"].map(positions)
        strategy_data = strategy_data.sort_values("scenario_position")
        ax.plot(
            strategy_data["scenario_position"],
            strategy_data["gap_to_oracle"],
            marker=strategy_marker(strategy),
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.55,
            color=strategy_color(strategy, index),
            linestyle=strategy_linestyle(strategy),
            linewidth=1.7,
            label=_short_strategy_label(strategy),
        )
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([scenario.replace("_", "\n").title() for scenario in scenario_order])
    format_axis(ax, x_label="Execution-Risk Scenario", y_label="Gap to Oracle")
    place_legend(ax, columns=3)
    save_paper_figure(fig, png_path, pdf_path)


def _save_baseline_weights_tradeoff_plot(data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save baseline WAPE versus normalized planning loss tradeoff."""
    apply_paper_style()
    plot_data = data.copy()
    fig, ax = plt.subplots(figsize=(7.25, 4.35))
    for index, row in enumerate(plot_data.itertuples(index=False)):
        strategy = row.strategy
        ax.scatter(
            row.WAPE,
            row.normalized_total_loss,
            s=46,
            color=strategy_color(strategy, index),
            marker=strategy_marker(strategy),
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        ax.annotate(
            row.method_name,
            (row.WAPE, row.normalized_total_loss),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#333333",
        )
    format_axis(
        ax,
        x_label="Weighted Absolute Percentage Error",
        y_label="Normalized Total Loss",
        grid_axis="both",
    )
    save_paper_figure(fig, png_path, pdf_path)


def _save_dataco_context_risk_plot(context_rates: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save a DataCo context late-delivery risk figure or an explicit fallback figure."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.25, 4.35))
    if context_rates.empty:
        ax.text(
            0.5,
            0.5,
            "DataCo late-delivery contexts were unavailable.\nConfigured fallback execution-risk scenarios were used.",
            ha="center",
            va="center",
            fontsize=9,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        save_paper_figure(fig, png_path, pdf_path)
        return
    plot_data = context_rates.sort_values("late_delivery_rate", ascending=False).head(20).copy()
    plot_data["context_label"] = plot_data["context_type"].astype(str) + ": " + plot_data["context_value"].astype(str)
    ax.barh(plot_data["context_label"], plot_data["late_delivery_rate"], color="#0072B2")
    ax.invert_yaxis()
    format_axis(ax, x_label="Late Delivery Rate", y_label="DataCo Context", grid_axis="x")
    save_paper_figure(fig, png_path, pdf_path)


def _save_lambda_scenarios_plot(scenario_table: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save generated execution lambda scenarios."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.25, 4.0))
    plot_data = scenario_table.copy()
    positions = np.arange(len(plot_data))
    colors = ["#0072B2" if source == "dataco_derived" else "#999999" for source in plot_data["source"]]
    ax.bar(positions, plot_data["lambda_execution"], color=colors)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(value).replace("_", "\n") for value in plot_data["scenario_name"]])
    format_axis(ax, x_label="Execution-Risk Scenario", y_label="Lambda Execution")
    save_paper_figure(fig, png_path, pdf_path)


def _save_strategy_rank_by_execution_weight_plot(data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save strategy rank by generated execution weight."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.25, 4.35))
    plot_data = data[data["strategy"].isin(BASELINE_COMPARISON_STRATEGIES)].copy()
    for index, strategy in enumerate(BASELINE_COMPARISON_STRATEGIES):
        strategy_data = plot_data[plot_data["strategy"] == strategy].sort_values("lambda_execution")
        if strategy_data.empty:
            continue
        ax.plot(
            strategy_data["lambda_execution"],
            strategy_data["rank_by_normalized_total_loss"],
            marker=strategy_marker(strategy),
            color=strategy_color(strategy, index),
            linestyle=strategy_linestyle(strategy),
            linewidth=1.7,
            label=_short_strategy_label(strategy),
        )
    ax.invert_yaxis()
    format_axis(ax, x_label="Lambda Execution", y_label="Rank by Normalized Total Loss")
    place_legend(ax, columns=4)
    save_paper_figure(fig, png_path, pdf_path)


def _save_scenario_metric_plot(
    data: pd.DataFrame,
    metric_column: str,
    y_label: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Save a scenario line plot for one metric."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.25, 4.35))
    plot_data = data[data["strategy"].isin(BASELINE_COMPARISON_STRATEGIES)].copy()
    scenario_order = list(dict.fromkeys(plot_data["scenario_name"].tolist()))
    positions = {scenario: index for index, scenario in enumerate(scenario_order)}
    for index, strategy in enumerate(BASELINE_COMPARISON_STRATEGIES):
        strategy_data = plot_data[plot_data["strategy"] == strategy].copy()
        if strategy_data.empty:
            continue
        strategy_data["scenario_position"] = strategy_data["scenario_name"].map(positions)
        strategy_data = strategy_data.sort_values("scenario_position")
        ax.plot(
            strategy_data["scenario_position"],
            strategy_data[metric_column],
            marker=strategy_marker(strategy),
            color=strategy_color(strategy, index),
            linestyle=strategy_linestyle(strategy),
            linewidth=1.7,
            label=_short_strategy_label(strategy),
        )
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([scenario.replace("_", "\n").title() for scenario in scenario_order])
    format_axis(ax, x_label="Execution-Risk Scenario", y_label=y_label)
    place_legend(ax, columns=4)
    save_paper_figure(fig, png_path, pdf_path)


def _save_rank_reordering_by_execution_weight_plot(data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save rank change versus baseline by execution weight."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.25, 4.35))
    for index, strategy in enumerate(BASELINE_COMPARISON_STRATEGIES):
        strategy_data = data[data["strategy"] == strategy].sort_values("lambda_execution")
        if strategy_data.empty:
            continue
        ax.plot(
            strategy_data["lambda_execution"],
            strategy_data["rank_change_vs_baseline"],
            marker=strategy_marker(strategy),
            color=strategy_color(strategy, index),
            linestyle=strategy_linestyle(strategy),
            linewidth=1.7,
            label=_short_strategy_label(strategy),
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    format_axis(ax, x_label="Lambda Execution", y_label="Rank Change versus Baseline")
    place_legend(ax, columns=4)
    save_paper_figure(fig, png_path, pdf_path)


def _save_model_rank_reordering_plot(data: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save a heatmap-style view of method ranks across objectives."""
    apply_paper_style()
    baseline = data[data["scenario_name"] == "baseline"].copy()
    if baseline.empty:
        baseline = data.copy()
    pivot = baseline.pivot_table(
        index="method_name",
        columns="objective_name",
        values="objective_rank",
        aggfunc="min",
    )
    fig, ax = plt.subplots(figsize=(7.25, 4.35))
    if sns is not None and not pivot.empty:
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis_r", cbar_kws={"label": "Rank"}, ax=ax)
        ax.set_xlabel("Objective")
        ax.set_ylabel("Method")
    else:
        ax.imshow(pivot.fillna(0).to_numpy(), aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Objective")
        ax.set_ylabel("Method")
    save_paper_figure(fig, png_path, pdf_path)


def _weight_plot_slice(weight_results: pd.DataFrame) -> pd.DataFrame:
    """Return a readable weight-sensitivity slice for plotting."""
    base_volatility = float(weight_results["lambda_volatility"].min())
    base_switch = float(weight_results["lambda_switch"].min())
    subset = weight_results[
        (weight_results["lambda_volatility"] == base_volatility)
        & (weight_results["lambda_switch"] == base_switch)
        & (weight_results["strategy"].isin(PAPER_STRATEGY_ORDER))
    ].copy()
    if subset.empty:
        subset = weight_results[weight_results["strategy"].isin(PAPER_STRATEGY_ORDER)].copy()
    return subset


def _save_weight_sensitivity_plot(
    data: pd.DataFrame,
    metric_column: str,
    y_label: str,
    title: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Save a line plot over execution weight for key strategies."""
    fig, ax = plt.subplots(figsize=(7.25, 4.35))
    for index, strategy in enumerate(PAPER_STRATEGY_ORDER):
        strategy_data = data[data["strategy"] == strategy].sort_values("lambda_execution")
        if strategy_data.empty:
            continue
        ax.plot(
            strategy_data["lambda_execution"],
            strategy_data[metric_column],
            marker=strategy_marker(strategy),
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.55,
            color=strategy_color(strategy, index),
            linestyle=strategy_linestyle(strategy),
            linewidth=1.9,
            label=_short_strategy_label(strategy),
        )
    ax.margins(x=0.03)
    format_axis(ax, x_label="Execution Weight", y_label=y_label)
    place_legend(ax, columns=3)
    save_paper_figure(fig, png_path, pdf_path)


def _save_capacity_stress_plot(
    data: pd.DataFrame,
    metric_column: str,
    y_label: str,
    title: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Save a scenario line plot for execution-capacity stress testing."""
    scenario_order = list(_execution_capacity_scenarios({}).keys())
    scenario_positions = {scenario: position for position, scenario in enumerate(scenario_order)}
    scenario_labels = {
        "high_capacity": "High",
        "medium_capacity": "Medium",
        "low_capacity": "Low",
        "severe_constraint": "Severe",
    }
    fig, ax = plt.subplots(figsize=(7.25, 4.35))
    for index, strategy in enumerate(PAPER_STRATEGY_ORDER):
        strategy_data = data[data["strategy"] == strategy].copy()
        if strategy_data.empty:
            continue
        strategy_data["scenario_position"] = strategy_data["capacity_scenario"].map(scenario_positions)
        strategy_data = strategy_data.sort_values("scenario_position")
        ax.plot(
            strategy_data["scenario_position"],
            strategy_data[metric_column],
            marker=strategy_marker(strategy),
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.55,
            color=strategy_color(strategy, index),
            linestyle=strategy_linestyle(strategy),
            linewidth=1.9,
            label=_short_strategy_label(strategy),
        )
    ax.set_xticks(list(scenario_positions.values()))
    ax.set_xticklabels([scenario_labels.get(label, label.replace("_", " ").title()) for label in scenario_order])
    if "rate" in metric_column:
        upper = max(0.05, float(data[metric_column].max()) * 1.15)
        ax.set_ylim(bottom=0.0, top=min(1.0, upper))
    ax.margins(x=0.03)
    format_axis(ax, x_label="Execution Capacity Scenario", y_label=y_label)
    place_legend(ax, columns=3)
    save_paper_figure(fig, png_path, pdf_path)


def _save_pareto_plot(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    title: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Save a Pareto scatter plot with dominated methods shown separately."""
    fig, ax = plt.subplots(figsize=(7.25, 4.35))
    plot_data = data.dropna(subset=[x_column, y_column]).copy()
    plot_data["pareto_status"] = np.where(plot_data["pareto_efficient"], "Pareto Efficient", "Dominated")
    if sns is not None:
        sns.scatterplot(
            data=plot_data,
            x=x_column,
            y=y_column,
            hue="pareto_status",
            style="pareto_status",
            hue_order=["Pareto Efficient", "Dominated"],
            style_order=["Pareto Efficient", "Dominated"],
            palette={"Pareto Efficient": "#0072B2", "Dominated": "#A6A6A6"},
            markers={"Pareto Efficient": "D", "Dominated": "o"},
            s=72,
            edgecolor="white",
            linewidth=0.6,
            ax=ax,
        )
    else:
        dominated = plot_data[~plot_data["pareto_efficient"]]
        efficient = plot_data[plot_data["pareto_efficient"]]
        ax.scatter(dominated[x_column], dominated[y_column], color="#A6A6A6", s=58, label="Dominated")
        ax.scatter(efficient[x_column], efficient[y_column], color="#0072B2", s=72, label="Pareto Efficient")
    _annotate_strategy_points(ax, plot_data, x_column=x_column, y_column=y_column)
    format_axis(ax, x_label=x_label, y_label=y_label, grid_axis="both")
    place_legend(ax, columns=2)
    save_paper_figure(fig, png_path, pdf_path)


def _save_scatter_figure(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Save a labeled scatter plot for strategy tradeoffs."""
    fig, ax = plt.subplots(figsize=(7.0, 4.25))
    plot_data = data.dropna(subset=[x_column, y_column]).copy()
    strategy_order = list(dict.fromkeys(plot_data["strategy"].tolist()))
    if sns is not None:
        sns.scatterplot(
            data=plot_data,
            x=x_column,
            y=y_column,
            hue="strategy",
            style="strategy",
            hue_order=strategy_order,
            style_order=strategy_order,
            palette=palette_for_strategies(strategy_order),
            markers={strategy: strategy_marker(strategy) for strategy in strategy_order},
            s=70,
            edgecolor="white",
            linewidth=0.6,
            legend=False,
            ax=ax,
        )
    else:
        for index, row in enumerate(plot_data.itertuples(index=False)):
            strategy = getattr(row, "strategy")
            ax.scatter(
                getattr(row, x_column),
                getattr(row, y_column),
                color=strategy_color(strategy, index),
                marker=strategy_marker(strategy),
                edgecolor="white",
                linewidth=0.6,
                s=70,
            )
    _annotate_strategy_points(ax, plot_data, x_column=x_column, y_column=y_column)
    format_axis(ax, x_label=x_label, y_label=y_label, grid_axis="both")
    save_paper_figure(fig, png_path, pdf_path)


def _save_example_planning_signal(decisions: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save an example actual-demand and planning-signal plot."""
    strategy_subset = ["global_best_model", "family_best_model", "stability_aware_selector", "feasibility_aware_selector"]
    candidate = decisions[decisions["strategy"].isin(strategy_subset)].copy()
    totals = candidate.groupby("series_id")["actual"].sum().sort_values(ascending=False)
    example_series = totals.index[0]
    plot_data = candidate[candidate["series_id"] == example_series].copy()

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    actual = plot_data[plot_data["strategy"] == strategy_subset[0]].sort_values("date")
    ax.plot(actual["date"], actual["actual"], color="#222222", linewidth=2.2, label="Actual Demand")
    for index, strategy in enumerate(strategy_subset):
        strategy_data = plot_data[plot_data["strategy"] == strategy].sort_values("date")
        if strategy_data.empty:
            continue
        ax.plot(
            strategy_data["date"],
            strategy_data["planning_signal"],
            linewidth=1.85,
            label=_short_strategy_label(strategy),
            color=strategy_color(strategy, index),
            linestyle=strategy_linestyle(strategy),
        )
    format_axis(ax, x_label="Date", y_label="Units")
    place_legend(ax, columns=3)
    fig.autofmt_xdate()
    save_paper_figure(fig, png_path, pdf_path)


def _annotate_strategy_points(ax, data: pd.DataFrame, x_column: str, y_column: str) -> None:
    """Add compact strategy labels to sparse tradeoff scatter plots."""
    offsets = {
        "individual_global_lightgbm": (8, 13, "left"),
        "individual_global_xgboost": (10, 10, "left"),
        "global_best_model": (10, -7, "left"),
        "family_best_model": (8, -14, "left"),
        "stability_aware_selector": (10, 5, "left"),
        "feasibility_aware_selector": (8, -12, "left"),
        "individual_moving_average": (-8, 17, "right"),
        "moving_average_baseline": (-8, 17, "right"),
        "individual_exponential_smoothing": (10, 18, "left"),
        "individual_naive_last_value": (-8, -12, "right"),
        "individual_seasonal_naive": (-12, 10, "right"),
    }
    for index, row in enumerate(data.itertuples(index=False)):
        strategy = getattr(row, "strategy")
        x_offset, y_offset, horizontal_alignment = offsets.get(
            strategy,
            (5 if index % 2 == 0 else -5, 4 if index % 3 != 0 else -8, "left" if index % 2 == 0 else "right"),
        )
        ax.annotate(
            _short_strategy_label(strategy),
            (getattr(row, x_column), getattr(row, y_column)),
            textcoords="offset points",
            xytext=(x_offset, y_offset),
            ha=horizontal_alignment,
            fontsize=7.2,
            color="#222222",
        )


def _short_strategy_label(strategy: str) -> str:
    """Return a compact English label for figure annotations."""
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
        "simple_ensemble": "Simple Ensemble",
        "best_inventory_cost_model": "Best Inventory",
        "best_stability_model": "Best Stability",
        "stability_aware_selector": "Stability-Aware",
        "feasibility_aware_selector": "Feasibility-Aware",
        "greedy_feasibility_selector": "Greedy Feasibility",
        "dp_feasibility_selector": "DP Feasibility",
        "budgeted_dp_feasibility_selector": "Budgeted DP",
        "feasibility_aware_smoothed_alpha_0_25": "Smoothed Alpha 0.25",
        "feasibility_aware_smoothed_alpha_0_50": "Smoothed Alpha 0.50",
        "feasibility_aware_smoothed_alpha_0_75": "Smoothed Alpha 0.75",
        "feasibility_aware_smoothed_scenario_alpha": "Scenario Alpha",
        "feasibility_aware_smoothed_utility_alpha": "Utility Alpha",
        "feasibility_aware_ensemble_inverse_accuracy": "Accuracy Ensemble",
        "feasibility_aware_ensemble_inverse_operational_loss": "Operational Ensemble",
        "feasibility_aware_ensemble_constrained": "Constrained Ensemble",
        "oracle_realized_demand": "Oracle",
    }
    return replacements.get(strategy, strategy.replace("_", " ").title())


if __name__ == "__main__":
    main()
