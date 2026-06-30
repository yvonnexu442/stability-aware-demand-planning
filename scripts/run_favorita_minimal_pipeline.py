"""Run the minimal real-data Favorita planning-stability pipeline."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loaders.favorita_loader import load_favorita_modeling_table
from evaluation.forecast_metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)
from evaluation.inventory_metrics import compute_holding_cost, compute_service_level, compute_shortage_cost
from evaluation.planning_utility import compute_total_planning_loss
from evaluation.stability_metrics import compute_absolute_plan_change, compute_percentage_plan_change
from planning_environment.execution_capacity import compute_execution_capacity, compute_execution_violation
from planning_environment.planning_actions import forecast_to_inventory_target
from reporting.latex_export import export_summary_table, write_asset_manifest
from utils.config import load_config, save_config_snapshot
from utils.logging_utils import setup_logger


MODEL_FEATURE_MAP = {
    "naive_last_value": "demand_lag_1",
    "seasonal_naive": "demand_lag_7",
    "moving_average": "demand_rolling_mean_28",
    "exponential_smoothing": "demand_ewm_alpha_0_3",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Favorita pipeline."""
    parser = argparse.ArgumentParser(description="Run the minimal Favorita planning-stability pipeline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the experiment config file.")
    parser.add_argument("--raw-data-dir", default=None, help="Directory containing local Favorita CSV files.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated experiment outputs.")
    parser.add_argument("--run-mode", choices=["quick", "full"], default=None, help="Override the configured run mode.")
    parser.add_argument("--max-series", type=int, default=None, help="Override the number of store-family series to use.")
    parser.add_argument("--skip-ml", action="store_true", help="Skip the global machine-learning forecast candidate.")
    return parser.parse_args()


def main() -> None:
    """Run Favorita loading, forecasting, planning evaluation, and paper export."""
    args = parse_args()
    config = load_config(args.config)

    run_mode = args.run_mode or config.get("project", {}).get("run_mode", "quick")
    output_dir = Path(args.output_dir or config.get("project", {}).get("output_dir", "outputs"))
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    log_dir = output_dir / "logs"
    config_dir = output_dir / "configs"
    for path in [table_dir, figure_dir, log_dir, config_dir, Path("paper/tables"), Path("paper/figures")]:
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

    table_assets = _export_latex_tables(forecast_metrics, inventory_metrics, stability_metrics, planning_utility)
    figure_assets = _make_figures(planning_utility, stability_metrics, decisions, figure_dir, Path("paper/figures"))
    manifest_path = write_asset_manifest("paper/asset_manifest.md", table_assets, figure_assets)
    logger.info("Saved LaTeX-ready tables to paper/tables.")
    logger.info("Saved LaTeX-ready PDF figures to paper/figures.")
    logger.info("Updated paper asset manifest at %s.", manifest_path)
    logger.info("Favorita minimal pipeline completed successfully.")


def _resolve_max_series(max_series_arg: Optional[int], run_mode: str, config: Mapping[str, object]) -> Optional[int]:
    """Return the configured number of series for quick or full mode."""
    if max_series_arg is not None:
        return int(max_series_arg)
    data_config = config.get("data", {})
    if run_mode == "quick":
        return data_config.get("quick_mode_max_series", 100)
    return data_config.get("full_mode_max_series")


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

    validation_metrics = forecast_metrics[forecast_metrics["split"] == "validation"].copy()
    global_best_model = validation_metrics.sort_values("weighted_absolute_percentage_error").iloc[0]["model_name"]
    family_best = _family_best_models(forecasts, global_best_model)
    logger.info("Global best validation model: %s", global_best_model)

    selected_frames: List[pd.DataFrame] = []
    for model_name in sorted(test_forecasts["model_name"].unique()):
        selected = test_forecasts[test_forecasts["model_name"] == model_name].copy()
        selected["strategy"] = "individual_{}".format(model_name)
        selected["selected_model"] = model_name
        selected_frames.append(selected)

    global_selected = test_forecasts[test_forecasts["model_name"] == global_best_model].copy()
    global_selected["strategy"] = "global_best_model"
    global_selected["selected_model"] = global_best_model
    selected_frames.append(global_selected)

    family_selected = test_forecasts.copy()
    family_selected["family_best_model"] = family_selected["family"].map(family_best).fillna(global_best_model)
    family_selected = family_selected[family_selected["model_name"] == family_selected["family_best_model"]].copy()
    family_selected["strategy"] = "family_best_model"
    family_selected["selected_model"] = family_selected["family_best_model"]
    selected_frames.append(family_selected.drop(columns=["family_best_model"]))

    stability_selected = _stability_aware_selection(test_forecasts, forecasts, forecast_metrics, config)
    selected_frames.append(stability_selected)

    selected_decisions = pd.concat(selected_frames, ignore_index=True)
    return _evaluate_selected_decisions(selected_decisions, config)


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
        test_forecasts[["date", "series_id", "family", "store_nbr", "actual", "split", "horizon", "safety_stock"]]
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
                "actual": float(row.actual),
                "split": row.split,
                "horizon": int(row.horizon),
                "safety_stock": float(row.safety_stock),
                "strategy": "stability_aware_selector",
            }
        )
    return pd.DataFrame(selected_records)


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

    frame["planning_signal"] = forecast_to_inventory_target(frame["forecast"], frame["safety_stock"])
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
                "execution_adaptation_penalty_total": float(group["execution_adaptation_penalty"].sum()),
                "total_planning_loss": total_loss,
                "service_level": compute_service_level(group["inventory_target"], actual),
                "large_jump_rate": float(group["large_jump_flag"].mean()),
                "execution_violation_rate": float(group["execution_violation"].mean()),
            }
        )
    return pd.DataFrame(records).sort_values("total_planning_loss")


def _export_latex_tables(
    forecast_metrics: pd.DataFrame,
    inventory_metrics: pd.DataFrame,
    stability_metrics: pd.DataFrame,
    planning_utility: pd.DataFrame,
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
            output_dir="paper/tables",
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
            output_dir="paper/tables",
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
            output_dir="paper/tables",
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
            output_dir="paper/tables",
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
    output_figure_dir.mkdir(parents=True, exist_ok=True)
    paper_figure_dir.mkdir(parents=True, exist_ok=True)

    merged = planning_utility.merge(stability_metrics, on="strategy", how="left", suffixes=("", "_stability"))
    _save_scatter_figure(
        merged,
        x_column="weighted_absolute_percentage_error",
        y_column="total_inventory_cost",
        title="Favorita Accuracy Versus Inventory Cost",
        x_label="Weighted Absolute Percentage Error",
        y_label="Total Inventory Cost",
        png_path=output_figure_dir / "favorita_accuracy_vs_inventory_cost.png",
        pdf_path=paper_figure_dir / "favorita_accuracy_vs_inventory_cost.pdf",
    )
    _save_scatter_figure(
        merged,
        x_column="weighted_absolute_percentage_error",
        y_column="mean_plan_change_pct",
        title="Favorita Accuracy Versus Planning Volatility",
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
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.scatter(data[x_column], data[y_column], color="#2f6f8f", s=45)
    for row in data.itertuples(index=False):
        ax.annotate(
            _short_strategy_label(getattr(row, "strategy")),
            (getattr(row, x_column), getattr(row, y_column)),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)


def _save_example_planning_signal(decisions: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Save an example actual-demand and planning-signal plot."""
    strategy_subset = ["global_best_model", "family_best_model", "stability_aware_selector"]
    candidate = decisions[decisions["strategy"].isin(strategy_subset)].copy()
    totals = candidate.groupby("series_id")["actual"].sum().sort_values(ascending=False)
    example_series = totals.index[0]
    plot_data = candidate[candidate["series_id"] == example_series].copy()

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    actual = plot_data[plot_data["strategy"] == strategy_subset[0]].sort_values("date")
    ax.plot(actual["date"], actual["actual"], color="#333333", linewidth=2.0, label="Actual Demand")
    colors = {
        "global_best_model": "#2f6f8f",
        "family_best_model": "#8f5f2f",
        "stability_aware_selector": "#6f8f2f",
    }
    for strategy in strategy_subset:
        strategy_data = plot_data[plot_data["strategy"] == strategy].sort_values("date")
        if strategy_data.empty:
            continue
        ax.plot(
            strategy_data["date"],
            strategy_data["planning_signal"],
            linewidth=1.8,
            label=_short_strategy_label(strategy),
            color=colors[strategy],
        )
    ax.set_title("Favorita Example Planning Signal")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)


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
        "stability_aware_selector": "Stability-Aware",
    }
    return replacements.get(strategy, strategy.replace("_", " ").title())


if __name__ == "__main__":
    main()
