"""Run M5 robustness checks for the operational planning feasibility thesis."""

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

from data_loaders.m5_loader import load_m5_modeling_table, validate_m5_files
from decision_layer.feasibility_dp_selector import (
    BudgetedDPFeasibilitySelector,
    DPFeasibilitySelector,
    GreedyFeasibilitySelector,
    OracleDPFeasibilitySelector,
)
from decision_layer.no_leakage import attach_actuals_for_evaluation, drop_future_outcomes, require_no_future_outcomes
from decision_layer.strategy_metadata import ensure_strategy_metadata, summarize_strategy_metadata
from evaluation.forecast_metrics import mean_absolute_error, weighted_absolute_percentage_error
from evaluation.inventory_metrics import compute_holding_cost, compute_service_level, compute_shortage_cost
from evaluation.planning_utility import add_normalized_planning_loss, compute_total_planning_loss
from evaluation.stability_metrics import compute_absolute_plan_change, compute_percentage_plan_change
from planning_environment.execution_capacity import compute_execution_capacity, compute_execution_violation
from planning_environment.planning_actions import forecast_to_inventory_target
from reporting.latex_export import export_summary_table
from utils.config import load_config
from utils.logging_utils import setup_logger
from visualization.plots import apply_paper_style, format_axis, place_legend, save_paper_figure, strategy_color, strategy_linestyle, strategy_marker


MODEL_FEATURE_MAP = {
    "naive_last_value": "demand_lag_1",
    "seasonal_naive": "demand_lag_7",
    "moving_average": "demand_rolling_mean_28",
    "exponential_smoothing": "demand_ewm_alpha_0_3",
}

M5_STRATEGY_ORDER = [
    "global_best_model",
    "simple_ensemble",
    "operational_loss_ensemble",
    "feasibility_aware_smoothed_alpha_0_25",
    "feasibility_aware_selector",
    "greedy_feasibility_selector",
    "dp_feasibility_selector",
    "budgeted_dp_feasibility_selector",
    "best_stability_model",
    "oracle_dp_feasibility_selector",
    "oracle_realized_demand",
]

REQUIRED_SUMMARY_COLUMNS = [
    "dataset_name",
    "run_mode",
    "grain_level",
    "intermittency_bucket",
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
    "normalized_inventory_component",
    "normalized_volatility_component",
    "normalized_execution_component",
    "normalized_switch_component",
    "normalized_total_loss",
    "gap_to_oracle",
    "rank_by_WAPE",
    "rank_by_execution_penalty",
    "rank_by_normalized_total_loss",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the M5 robustness pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--run-mode", choices=["quick", "medium", "full", "quick_mode", "medium_mode", "full_mode"], default="quick")
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--paper-table-dir", default="paper/tables")
    parser.add_argument("--paper-figure-dir", default="paper/figures")
    return parser.parse_args()


def main() -> None:
    """Run M5 robustness checks and export paper-ready assets."""
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

    logger = setup_logger("m5_robustness_pipeline", log_file=log_dir / "m5_robustness_pipeline.log")
    raw_data_dir = Path(args.raw_data_dir or config.get("m5_pipeline", {}).get("raw_data_dir", "data/raw/m5"))

    if run_mode == "full" and not bool(config.get("m5_pipeline", {}).get("enable_full_mode", False)):
        logger.warning("M5 full_mode is disabled by default. Set m5_pipeline.enable_full_mode=true to run it.")
        return

    try:
        validate_m5_files(raw_data_dir)
    except (FileNotFoundError, ValueError) as error:
        logger.warning("%s", error)
        write_missing_data_instructions(table_dir, raw_data_dir, str(error))
        return

    max_series = args.max_series if args.max_series is not None else resolve_m5_max_series(run_mode, config)
    logger.info("Running M5 robustness pipeline in %s mode with max_series=%s.", run_mode, max_series)
    modeling_table, quality_report = load_m5_modeling_table(
        raw_data_dir=raw_data_dir,
        run_mode=run_mode,
        max_series=max_series,
        min_history_length=int(config.get("m5_pipeline", {}).get("min_history_length", 365)),
        min_nonzero_observations=int(config.get("m5_pipeline", {}).get("min_nonzero_observations", 10)),
        output_table_dir=table_dir,
    )
    quality_report.to_csv(table_dir / "m5_data_quality_report.csv", index=False)
    modeling_table = assign_splits(modeling_table, config)

    item_results, item_decisions = run_grain_experiment(
        modeling_table=modeling_table,
        config=config,
        run_mode=run_mode,
        grain_level="item_store",
        scenario_name="baseline",
        lambda_execution=None,
    )
    large_scale = item_results

    hierarchy = run_hierarchy_sensitivity(modeling_table, config, run_mode)
    intermittent = run_intermittent_demand_stress(item_decisions, config, run_mode)
    scenarios = load_dataco_execution_scenarios(config, table_dir, logger)
    scenario_results = []
    for _, scenario in scenarios.iterrows():
        scenario_summary = summarize_decisions_for_scenario(
            item_decisions,
            config=config,
            run_mode=run_mode,
            grain_level="item_store",
            intermittency_bucket="all",
            scenario_name=str(scenario["scenario_name"]),
            lambda_execution=float(scenario["lambda_execution"]),
        )
        scenario_results.append(scenario_summary)
    dataco_scenario = pd.concat(scenario_results, ignore_index=True)

    robustness_summary = pd.concat(
        [
            large_scale.assign(module_name="large_scale_replication"),
            hierarchy.assign(module_name="hierarchy_sensitivity"),
            intermittent.assign(module_name="intermittent_demand_stress"),
            dataco_scenario.assign(module_name="dataco_scenario_robustness"),
        ],
        ignore_index=True,
        sort=False,
    )

    write_m5_outputs(
        large_scale=large_scale,
        hierarchy=hierarchy,
        intermittent=intermittent,
        dataco_scenario=dataco_scenario,
        robustness_summary=robustness_summary,
        table_dir=table_dir,
        paper_table_dir=paper_table_dir,
    )
    write_m5_figures(
        large_scale=large_scale,
        hierarchy=hierarchy,
        intermittent=intermittent,
        dataco_scenario=dataco_scenario,
        figure_dir=figure_dir,
        paper_figure_dir=paper_figure_dir,
    )
    logger.info("M5 robustness pipeline completed successfully.")


def normalize_run_mode(run_mode: str) -> str:
    """Normalize run-mode aliases."""
    value = str(run_mode).strip().lower()
    aliases = {"quick_mode": "quick", "medium_mode": "medium", "full_mode": "full"}
    return aliases.get(value, value)


def resolve_m5_max_series(run_mode: str, config: Mapping[str, object]) -> Optional[int]:
    """Return the configured M5 sample size."""
    m5_config = config.get("m5_pipeline", {})
    if run_mode == "quick":
        return int(m5_config.get("quick_mode_max_series", 200))
    if run_mode == "medium":
        return int(m5_config.get("medium_mode_max_series", 1500))
    return m5_config.get("full_mode_max_series")


def write_missing_data_instructions(table_dir: Path, raw_data_dir: Path, reason: str) -> None:
    """Write an explicit missing-data audit table without failing other workflows."""
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "dataset_name": "m5",
                "status": "missing_required_data",
                "raw_data_dir": str(raw_data_dir),
                "reason": reason,
                "required_files": "calendar.csv, sell_prices.csv, sales_train_validation.csv",
            }
        ]
    ).to_csv(table_dir / "m5_missing_data_instructions.csv", index=False)


def assign_splits(modeling_table: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    """Assign chronological train, validation, and test splits."""
    frame = modeling_table.copy()
    validation_horizon = int(config.get("m5_pipeline", {}).get("validation_horizon", 28))
    test_horizon = int(config.get("m5_pipeline", {}).get("test_horizon", 28))
    dates = sorted(pd.to_datetime(frame["date"]).unique())
    if len(dates) <= validation_horizon + test_horizon:
        raise ValueError("M5 modeling table does not contain enough dates for validation and test splits.")
    validation_dates = set(dates[-(validation_horizon + test_horizon) : -test_horizon])
    test_dates = set(dates[-test_horizon:])
    frame["split"] = "train"
    frame.loc[frame["date"].isin(validation_dates), "split"] = "validation"
    frame.loc[frame["date"].isin(test_dates), "split"] = "test"
    frame["horizon"] = frame.groupby(["series_id", "split"]).cumcount() + 1
    return frame


def run_grain_experiment(
    modeling_table: pd.DataFrame,
    config: Mapping[str, object],
    run_mode: str,
    grain_level: str,
    scenario_name: str,
    lambda_execution: Optional[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build forecasts, decisions, and summaries for one planning grain."""
    scenario_config = copy.deepcopy(dict(config))
    if lambda_execution is not None:
        scenario_config.setdefault("planning_loss_weights", {})["lambda_execution"] = float(lambda_execution)
    forecasts = build_candidate_forecasts(modeling_table)
    forecast_metrics = summarize_forecast_metrics(forecasts)
    decisions = build_decision_outputs(forecasts, modeling_table, forecast_metrics, scenario_config)
    summary = summarize_decisions_for_scenario(
        decisions,
        config=scenario_config,
        run_mode=run_mode,
        grain_level=grain_level,
        intermittency_bucket="all",
        scenario_name=scenario_name,
        lambda_execution=float(scenario_config.get("planning_loss_weights", {}).get("lambda_execution", 1.0)),
    )
    return summary, decisions


def build_candidate_forecasts(modeling_table: pd.DataFrame) -> pd.DataFrame:
    """Build transparent M5 forecast candidates for validation and test rows."""
    rows = []
    prediction_frame = modeling_table[modeling_table["split"].isin(["validation", "test"])].copy()
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
    for model_name, feature_column in MODEL_FEATURE_MAP.items():
        if feature_column not in prediction_frame.columns:
            continue
        frame = prediction_frame[base_columns].copy()
        frame["model_name"] = model_name
        frame["forecast"] = pd.to_numeric(prediction_frame[feature_column], errors="coerce").fillna(0.0).clip(lower=0.0)
        frame = frame.rename(columns={"demand": "actual"})
        rows.append(frame)
    if not rows:
        raise ValueError("No M5 forecast candidates could be built from modeling features.")
    return pd.concat(rows, ignore_index=True)


def summarize_forecast_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return validation and test WAPE by model."""
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
    return pd.DataFrame(records).sort_values(["split", "WAPE", "model_name"])


def build_decision_outputs(
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Build M5 strategy decisions and evaluate planning outcomes."""
    test_forecasts = forecasts[forecasts["split"] == "test"].copy()
    safety_multiplier = float(config.get("m5_pipeline", {}).get("safety_stock_multiplier", 0.5))
    test_forecasts["safety_stock"] = (
        pd.to_numeric(test_forecasts["demand_rolling_std_28"], errors="coerce").fillna(0.0) * safety_multiplier
    ).clip(lower=0.0)
    deployable_test_forecasts = drop_future_outcomes(test_forecasts)
    validation_metrics = forecast_metrics[forecast_metrics["split"] == "validation"].copy()
    global_best_model = str(validation_metrics.sort_values("WAPE").iloc[0]["model_name"])
    best_stability_model = best_stability_candidate(forecasts, modeling_table, config, fallback_model=global_best_model)
    operational_weights = operational_loss_model_weights(forecasts, modeling_table, forecast_metrics, config)
    expected_losses_raw = validation_expected_losses(forecasts, modeling_table, config)
    series_expected_losses, global_expected_losses = split_expected_losses(expected_losses_raw)
    realized_inventory_costs = realized_inventory_costs_by_series_model(test_forecasts, config)

    selected_frames = []
    selected_frames.append(fixed_model_selection(deployable_test_forecasts, global_best_model, "global_best_model"))
    selected_frames.append(simple_ensemble_selection(deployable_test_forecasts))
    selected_frames.append(weighted_ensemble_selection(deployable_test_forecasts, operational_weights, "operational_loss_ensemble"))
    selected_frames.append(smoothed_global_best_selection(deployable_test_forecasts, global_best_model, config, alpha=0.25))
    selected_frames.append(feasibility_aware_selection(deployable_test_forecasts, expected_losses_raw, config))
    selected_frames.append(greedy_feasibility_selection(deployable_test_forecasts, series_expected_losses, global_expected_losses, config))
    selected_frames.append(dp_feasibility_selection(deployable_test_forecasts, series_expected_losses, global_expected_losses, config))
    selected_frames.append(budgeted_dp_feasibility_selection(deployable_test_forecasts, series_expected_losses, global_expected_losses, config))
    selected_frames.append(fixed_model_selection(deployable_test_forecasts, best_stability_model, "best_stability_model"))
    deployable_decisions = attach_actuals_for_evaluation(
        pd.concat(selected_frames, ignore_index=True),
        test_forecasts,
        key_columns=("date", "series_id"),
    )
    oracle_dp_decisions = attach_actuals_for_evaluation(
        oracle_dp_feasibility_selection(
            deployable_test_forecasts,
            series_expected_losses,
            global_expected_losses,
            realized_inventory_costs,
            config,
        ),
        test_forecasts,
        key_columns=("date", "series_id"),
    )
    oracle_decisions = oracle_selection(test_forecasts)
    return evaluate_selected_decisions(pd.concat([deployable_decisions, oracle_dp_decisions, oracle_decisions], ignore_index=True), config)


def fixed_model_selection(test_forecasts: pd.DataFrame, model_name: str, strategy: str) -> pd.DataFrame:
    """Return rows for a fixed model strategy."""
    selected = test_forecasts[test_forecasts["model_name"] == model_name].copy()
    if selected.empty:
        selected = test_forecasts[test_forecasts["model_name"] == test_forecasts["model_name"].iloc[0]].copy()
    selected["selected_model"] = selected["model_name"]
    selected["strategy"] = strategy
    return selected


def simple_ensemble_selection(test_forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return equal-weight ensemble decisions."""
    require_no_future_outcomes(test_forecasts, "simple_ensemble_selection")
    base_columns = decision_base_columns(include_actual=False)
    ensemble = test_forecasts.groupby(base_columns, dropna=False).agg(forecast=("forecast", "mean")).reset_index()
    ensemble["model_name"] = "simple_ensemble"
    ensemble["selected_model"] = "simple_ensemble"
    ensemble["strategy"] = "simple_ensemble"
    return ensemble


def weighted_ensemble_selection(test_forecasts: pd.DataFrame, weights: Mapping[str, float], strategy: str) -> pd.DataFrame:
    """Return weighted ensemble decisions."""
    require_no_future_outcomes(test_forecasts, "weighted_ensemble_selection")
    frame = test_forecasts.copy()
    if not weights:
        models = sorted(frame["model_name"].unique())
        weights = {model: 1.0 / float(len(models)) for model in models}
    frame["model_weight"] = frame["model_name"].map(weights).fillna(0.0)
    frame["weighted_forecast"] = frame["forecast"] * frame["model_weight"]
    base_columns = decision_base_columns(include_actual=False)
    ensemble = (
        frame.groupby(base_columns, dropna=False)
        .agg(forecast=("weighted_forecast", "sum"), weight_sum=("model_weight", "sum"))
        .reset_index()
    )
    ensemble["forecast"] = ensemble["forecast"] / ensemble["weight_sum"].replace(0.0, np.nan)
    ensemble["forecast"] = ensemble["forecast"].fillna(0.0)
    ensemble = ensemble.drop(columns=["weight_sum"])
    ensemble["model_name"] = strategy
    ensemble["selected_model"] = strategy
    ensemble["strategy"] = strategy
    return ensemble


def smoothed_global_best_selection(test_forecasts: pd.DataFrame, model_name: str, config: Mapping[str, object], alpha: float) -> pd.DataFrame:
    """Return global-best forecasts with gradually adapted executable plans."""
    require_no_future_outcomes(test_forecasts, "smoothed_global_best_selection")
    selected = fixed_model_selection(test_forecasts, model_name, "feasibility_aware_smoothed_alpha_0_25")
    records = []
    for _, group in selected.sort_values(["series_id", "date"]).groupby("series_id"):
        previous_plan = None
        for row in group.itertuples(index=False):
            candidate_plan = float(forecast_to_inventory_target([row.forecast], [row.safety_stock])[0])
            final_plan = candidate_plan if previous_plan is None else float(alpha) * candidate_plan + (1.0 - float(alpha)) * previous_plan
            previous_plan = final_plan
            record = row._asdict()
            record["planning_signal_override"] = final_plan
            records.append(record)
    return pd.DataFrame(records)


def feasibility_aware_selection(
    test_forecasts: pd.DataFrame,
    expected_losses: Mapping[Tuple[str, str], Mapping[str, float]],
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Return a simple interpretable feasibility-aware selector."""
    require_no_future_outcomes(test_forecasts, "feasibility_aware_selection")
    weights = config.get("planning_loss_weights", {})
    max_plan_change_rate = float(config.get("stability", {}).get("max_plan_change_rate", 0.20))
    switch_penalty = float(config.get("feasibility_analysis", {}).get("feasibility_selector", {}).get("switch_penalty", 0.02))
    candidate_groups = {key: group.copy() for key, group in test_forecasts.groupby(["series_id", "date"], sort=False)}
    base_rows = test_forecasts[decision_base_columns(include_actual=False)].drop_duplicates(["series_id", "date"]).sort_values(["series_id", "date"])
    states: Dict[str, Dict[str, object]] = {}
    records = []
    for row in base_rows.itertuples(index=False):
        candidates = candidate_groups[(row.series_id, row.date)]
        state = states.get(row.series_id, {})
        previous_model = state.get("selected_model")
        previous_plan = state.get("planning_signal")
        scored = []
        for candidate in candidates.itertuples(index=False):
            planning_signal = float(forecast_to_inventory_target([candidate.forecast], [row.safety_stock])[0])
            if previous_plan is None:
                plan_change_pct = 0.0
                execution_violation = 0.0
            else:
                plan_change_abs = abs(planning_signal - float(previous_plan))
                plan_change_pct = plan_change_abs / max(abs(float(previous_plan)), 1e-8)
                execution_violation = max(plan_change_abs - abs(float(previous_plan)) * max_plan_change_rate, 0.0)
            losses = expected_losses.get((row.series_id, candidate.model_name), expected_losses.get(("global", candidate.model_name), {}))
            switch_cost = 0.0 if previous_model is None or previous_model == candidate.model_name else switch_penalty
            score = (
                float(weights.get("alpha_forecast", 1.0)) * float(losses.get("wape", 1.0))
                + float(weights.get("beta_inventory", 1.0)) * float(losses.get("inventory_cost_per_demand_unit", 0.0))
                + float(weights.get("lambda_volatility", 0.5)) * plan_change_pct
                + float(weights.get("lambda_switch", 0.5)) * switch_cost
                + float(weights.get("lambda_execution", 1.0)) * execution_violation / max(abs(float(previous_plan or planning_signal)), 1e-8)
            )
            scored.append((score, candidate, planning_signal))
        _, best_candidate, best_plan = min(scored, key=lambda value: (value[0], value[1].model_name))
        states[row.series_id] = {"selected_model": best_candidate.model_name, "planning_signal": best_plan}
        record = best_candidate._asdict()
        record["selected_model"] = best_candidate.model_name
        record["strategy"] = "feasibility_aware_selector"
        records.append(record)
    return pd.DataFrame(records)


def greedy_feasibility_selection(
    test_forecasts: pd.DataFrame,
    expected_losses: Mapping[Tuple[str, str], Mapping[str, float]],
    global_expected_losses: Mapping[str, Mapping[str, float]],
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Return one-step minimum expected operational-cost selections."""
    require_no_future_outcomes(test_forecasts, "greedy_feasibility_selection")
    selector = GreedyFeasibilitySelector(
        expected_losses=expected_losses,
        global_expected_losses=global_expected_losses,
        weights=config.get("planning_loss_weights", {}),
        switch_penalty=selector_switch_penalty(config),
        max_plan_change_rate=selector_max_plan_change_rate(config),
        calibration_group_column="series_id",
        strategy_name="greedy_feasibility_selector",
    )
    return selector.select(test_forecasts)


def dp_feasibility_selection(
    test_forecasts: pd.DataFrame,
    expected_losses: Mapping[Tuple[str, str], Mapping[str, float]],
    global_expected_losses: Mapping[str, Mapping[str, float]],
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Return finite-horizon DP selections using validation-derived costs."""
    require_no_future_outcomes(test_forecasts, "dp_feasibility_selection")
    selector = DPFeasibilitySelector(
        expected_losses=expected_losses,
        global_expected_losses=global_expected_losses,
        weights=config.get("planning_loss_weights", {}),
        switch_penalty=selector_switch_penalty(config),
        max_plan_change_rate=selector_max_plan_change_rate(config),
        calibration_group_column="series_id",
        strategy_name="dp_feasibility_selector",
    )
    return selector.select(test_forecasts)


def budgeted_dp_feasibility_selection(
    test_forecasts: pd.DataFrame,
    expected_losses: Mapping[Tuple[str, str], Mapping[str, float]],
    global_expected_losses: Mapping[str, Mapping[str, float]],
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Return finite-horizon DP selections under a hard switch budget."""
    require_no_future_outcomes(test_forecasts, "budgeted_dp_feasibility_selection")
    selector = BudgetedDPFeasibilitySelector(
        expected_losses=expected_losses,
        global_expected_losses=global_expected_losses,
        weights=config.get("planning_loss_weights", {}),
        switch_penalty=selector_switch_penalty(config),
        max_plan_change_rate=selector_max_plan_change_rate(config),
        max_switches=selector_switch_budget(config),
        calibration_group_column="series_id",
        strategy_name="budgeted_dp_feasibility_selector",
    )
    return selector.select(test_forecasts)


def oracle_dp_feasibility_selection(
    test_forecasts: pd.DataFrame,
    expected_losses: Mapping[Tuple[str, str], Mapping[str, float]],
    global_expected_losses: Mapping[str, Mapping[str, float]],
    realized_inventory_costs: Mapping[Tuple[object, ...], float],
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Return non-deployable Realized-Inventory Oracle DP selections."""
    require_no_future_outcomes(test_forecasts, "oracle_dp_feasibility_selection")
    selector = OracleDPFeasibilitySelector(
        expected_losses=expected_losses,
        realized_inventory_costs=realized_inventory_costs,
        global_expected_losses=global_expected_losses,
        weights=config.get("planning_loss_weights", {}),
        switch_penalty=selector_switch_penalty(config),
        max_plan_change_rate=selector_max_plan_change_rate(config),
        calibration_group_column="series_id",
        strategy_name="oracle_dp_feasibility_selector",
    )
    return selector.select(test_forecasts)


def selector_switch_penalty(config: Mapping[str, object]) -> float:
    """Return the configured soft switch penalty for selector scoring."""
    return float(config.get("feasibility_analysis", {}).get("feasibility_selector", {}).get("switch_penalty", 0.02))


def selector_max_plan_change_rate(config: Mapping[str, object]) -> float:
    """Return the execution-capacity plan-change rate used by selectors."""
    return float(config.get("stability", {}).get("max_plan_change_rate", 0.20))


def selector_switch_budget(config: Mapping[str, object]) -> int:
    """Return the hard switch budget for budgeted DP selectors."""
    dp_config = config.get("feasibility_analysis", {}).get("dp_selector", {})
    return int(dp_config.get("max_switches", config.get("stability", {}).get("max_model_switches_per_window", 2)))


def oracle_selection(test_forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return non-deployable realized-demand upper bound."""
    oracle = test_forecasts[decision_base_columns()].drop_duplicates(["series_id", "date"]).copy()
    oracle["forecast"] = oracle["actual"]
    oracle["model_name"] = "oracle_realized_demand"
    oracle["selected_model"] = "oracle_realized_demand"
    oracle["strategy"] = "oracle_realized_demand"
    return oracle


def decision_base_columns(include_actual: bool = True) -> List[str]:
    """Return columns that identify a M5 planning decision row."""
    columns = [
        "date",
        "series_id",
        "split",
        "horizon",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "demand_rolling_std_28",
        "zero_demand_rate_28",
        "safety_stock",
    ]
    if include_actual:
        columns.insert(4, "actual")
    return columns


def validation_expected_losses(
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    config: Mapping[str, object],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Return validation forecast and inventory losses by series-model."""
    validation = forecasts[forecasts["split"] == "validation"].copy()
    safety_multiplier = float(config.get("m5_pipeline", {}).get("safety_stock_multiplier", 0.5))
    validation["safety_stock"] = pd.to_numeric(validation["demand_rolling_std_28"], errors="coerce").fillna(0.0) * safety_multiplier
    planning_config = config.get("planning", {})
    losses: Dict[Tuple[str, str], Dict[str, float]] = {}
    global_records = []
    for (series_id, model_name), group in validation.groupby(["series_id", "model_name"]):
        actual = group["actual"].to_numpy(dtype=float)
        forecast = group["forecast"].to_numpy(dtype=float)
        signal = forecast_to_inventory_target(forecast, group["safety_stock"].to_numpy(dtype=float))
        inventory = compute_holding_cost(signal, actual, float(planning_config.get("holding_cost_rate", 1.0))) + compute_shortage_cost(
            signal,
            actual,
            float(planning_config.get("shortage_cost_rate", 5.0)),
        )
        demand_total = max(float(np.sum(np.abs(actual))), 1e-8)
        record = {
            "wape": weighted_absolute_percentage_error(actual, forecast),
            "inventory_cost_per_demand_unit": float(np.sum(inventory) / demand_total),
        }
        losses[(series_id, model_name)] = record
        global_records.append({"model_name": model_name, **record})
    global_frame = pd.DataFrame(global_records)
    for model_name, group in global_frame.groupby("model_name"):
        losses[("global", model_name)] = {
            "wape": float(group["wape"].mean()),
            "inventory_cost_per_demand_unit": float(group["inventory_cost_per_demand_unit"].mean()),
        }
    return losses


def split_expected_losses(
    expected_losses: Dict[Tuple[str, str], Dict[str, float]],
) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Split expected losses into series-level and global-level dictionaries."""
    series_losses = {key: value for key, value in expected_losses.items() if key[0] != "global"}
    global_losses = {key[1]: value for key, value in expected_losses.items() if key[0] == "global"}
    return series_losses, global_losses


def realized_inventory_costs_by_series_model(
    test_forecasts: pd.DataFrame,
    config: Mapping[str, object],
) -> Dict[Tuple[object, ...], float]:
    """Return period-specific test-realized inventory costs for Oracle DP only.

    The primary key is ``(series_id, model_name, date)``. Global period-level
    fallbacks are provided only for missing candidate rows.
    """
    planning_config = config.get("planning", {})
    frame = test_forecasts.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    actual = frame["actual"].to_numpy(dtype=float)
    signal = forecast_to_inventory_target(
        frame["forecast"].to_numpy(dtype=float),
        frame["safety_stock"].to_numpy(dtype=float),
    )
    inventory = compute_holding_cost(
        signal,
        actual,
        float(planning_config.get("holding_cost_rate", 1.0)),
    ) + compute_shortage_cost(
        signal,
        actual,
        float(planning_config.get("shortage_cost_rate", 5.0)),
    )
    frame["realized_inventory_cost"] = inventory / np.maximum(np.abs(actual), 1.0)

    realized_costs: Dict[Tuple[object, ...], float] = {}
    for row in frame.itertuples(index=False):
        realized_costs[(str(row.series_id), str(row.model_name), pd.Timestamp(row.date))] = float(row.realized_inventory_cost)

    for (model_name, date), group in frame.groupby(["model_name", "date"]):
        realized_costs[("global", str(model_name), pd.Timestamp(date))] = float(group["realized_inventory_cost"].mean())

    for (series_id, model_name), group in frame.groupby(["series_id", "model_name"]):
        realized_costs[(str(series_id), str(model_name))] = float(group["realized_inventory_cost"].mean())

    for model_name, group in frame.groupby("model_name"):
        realized_costs[("global", str(model_name))] = float(group["realized_inventory_cost"].mean())
    return realized_costs


def operational_loss_model_weights(
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    config: Mapping[str, object],
) -> Dict[str, float]:
    """Return inverse validation planning-loss weights by model."""
    validation = forecasts[forecasts["split"] == "validation"].copy()
    safety_multiplier = float(config.get("m5_pipeline", {}).get("safety_stock_multiplier", 0.5))
    validation["safety_stock"] = pd.to_numeric(validation["demand_rolling_std_28"], errors="coerce").fillna(0.0) * safety_multiplier
    frames = []
    for model_name in sorted(validation["model_name"].unique()):
        selected = fixed_model_selection(validation, model_name, "global_best_model" if model_name == validation["model_name"].iloc[0] else "individual_{}".format(model_name))
        selected["strategy"] = "individual_{}".format(model_name)
        frames.append(selected)
    decisions = evaluate_selected_decisions(pd.concat(frames, ignore_index=True), config)
    summary = summarize_planning_utility(decisions, config.get("planning_loss_weights", {}))
    reference_strategy = summary.sort_values("weighted_absolute_percentage_error").iloc[0]["strategy"]
    summary, _, _ = add_normalized_planning_loss(
        summary,
        config.get("planning_loss_weights", {}),
        reference_strategy=reference_strategy,
        dataset_name="m5",
        run_mode="validation",
        split_name="validation",
    )
    losses = {
        str(row.strategy).replace("individual_", ""): max(float(row.normalized_total_loss), 1e-6)
        for row in summary.itertuples(index=False)
    }
    raw = {model: 1.0 / loss for model, loss in losses.items()}
    total = sum(raw.values())
    return {model: value / total for model, value in raw.items()} if total > 0 else {}


def best_stability_candidate(
    forecasts: pd.DataFrame,
    modeling_table: pd.DataFrame,
    config: Mapping[str, object],
    fallback_model: str,
) -> str:
    """Return validation model with lowest planning-signal volatility."""
    validation = forecasts[forecasts["split"] == "validation"].copy()
    safety_multiplier = float(config.get("m5_pipeline", {}).get("safety_stock_multiplier", 0.5))
    validation["safety_stock"] = pd.to_numeric(validation["demand_rolling_std_28"], errors="coerce").fillna(0.0) * safety_multiplier
    records = []
    for model_name, group in validation.groupby("model_name"):
        volatility = 0.0
        for _, series_group in group.sort_values(["series_id", "date"]).groupby("series_id"):
            signal = forecast_to_inventory_target(series_group["forecast"], series_group["safety_stock"])
            volatility += float(np.sum(compute_percentage_plan_change(signal)))
        records.append({"model_name": model_name, "planning_volatility": volatility})
    if not records:
        return fallback_model
    return str(pd.DataFrame(records).sort_values(["planning_volatility", "model_name"]).iloc[0]["model_name"])


def evaluate_selected_decisions(selected_decisions: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    """Evaluate inventory, stability, switching, and execution metrics."""
    frame = ensure_strategy_metadata(selected_decisions).sort_values(["strategy", "series_id", "date"]).reset_index(drop=True)
    planning_config = config.get("planning", {})
    stability_config = config.get("stability", {})
    weights = config.get("planning_loss_weights", {})
    default_signal = forecast_to_inventory_target(frame["forecast"], frame["safety_stock"])
    if "planning_signal_override" in frame.columns:
        override = pd.to_numeric(frame["planning_signal_override"], errors="coerce")
        frame["planning_signal"] = override.where(override.notna(), default_signal)
    else:
        frame["planning_signal"] = default_signal
    frame["inventory_target"] = frame["planning_signal"]
    actual = frame["actual"].to_numpy(dtype=float)
    frame["holding_cost"] = compute_holding_cost(frame["inventory_target"], actual, float(planning_config.get("holding_cost_rate", 1.0)))
    frame["shortage_cost"] = compute_shortage_cost(frame["inventory_target"], actual, float(planning_config.get("shortage_cost_rate", 5.0)))
    frame["total_inventory_cost"] = frame["holding_cost"] + frame["shortage_cost"]
    frame["service_level_hit"] = (frame["inventory_target"] >= frame["actual"]).astype(int)
    frame["absolute_plan_change"] = 0.0
    frame["plan_change_pct"] = 0.0
    frame["model_switch_flag"] = 0
    frame["execution_adaptation_penalty"] = 0.0
    frame["execution_violation"] = 0
    max_plan_change_rate = float(stability_config.get("max_plan_change_rate", 0.20))
    for (_, _), index in frame.groupby(["strategy", "series_id"]).groups.items():
        signal = frame.loc[index, "planning_signal"].to_numpy(dtype=float)
        abs_change = compute_absolute_plan_change(signal)
        pct_change = compute_percentage_plan_change(signal)
        capacity = compute_execution_capacity(signal, max_plan_change_rate=max_plan_change_rate)
        execution = compute_execution_violation(signal, capacity)
        models = frame.loc[index, "selected_model"].astype(str).tolist()
        switches = np.array([0] + [int(left != right) for left, right in zip(models[:-1], models[1:])])
        frame.loc[index, "absolute_plan_change"] = abs_change
        frame.loc[index, "plan_change_pct"] = pct_change
        frame.loc[index, "execution_adaptation_penalty"] = execution
        frame.loc[index, "execution_violation"] = (execution > 0.0).astype(int)
        frame.loc[index, "model_switch_flag"] = switches
    forecast_error = (frame["actual"] - frame["forecast"]).abs()
    frame["total_planning_loss"] = (
        float(weights.get("alpha_forecast", 1.0)) * forecast_error
        + float(weights.get("beta_inventory", 1.0)) * frame["total_inventory_cost"]
        + float(weights.get("lambda_volatility", 0.5)) * frame["plan_change_pct"]
        + float(weights.get("lambda_switch", 0.5)) * frame["model_switch_flag"]
        + float(weights.get("lambda_execution", 1.0)) * frame["execution_adaptation_penalty"]
    )
    return frame


def summarize_planning_utility(decisions: pd.DataFrame, loss_weights: Mapping[str, float]) -> pd.DataFrame:
    """Return strategy-level M5 planning utility metrics."""
    records = []
    for strategy, group in decisions.groupby("strategy"):
        actual = group["actual"].to_numpy(dtype=float)
        forecast = group["forecast"].to_numpy(dtype=float)
        record = {
            "strategy": strategy,
            "selected_model_count": group["selected_model"].nunique(),
            "mean_absolute_error": mean_absolute_error(actual, forecast),
            "weighted_absolute_percentage_error": weighted_absolute_percentage_error(actual, forecast),
            "total_inventory_cost": float(group["total_inventory_cost"].sum()),
            "planning_signal_volatility_total": float(group["plan_change_pct"].sum()),
            "model_switching_cost_total": float(group["model_switch_flag"].sum()),
            "model_switch_count": int(group["model_switch_flag"].sum()),
            "execution_adaptation_penalty_total": float(group["execution_adaptation_penalty"].sum()),
            "total_planning_loss": compute_total_planning_loss(
                forecast_error=np.abs(actual - forecast),
                inventory_cost=group["total_inventory_cost"].to_numpy(dtype=float),
                planning_signal_volatility=group["plan_change_pct"].to_numpy(dtype=float),
                model_switching_cost=group["model_switch_flag"].to_numpy(dtype=float),
                execution_adaptation_penalty=group["execution_adaptation_penalty"].to_numpy(dtype=float),
                weights=loss_weights,
            ),
            "service_level": compute_service_level(group["inventory_target"], actual),
            "execution_violation_rate": float(group["execution_violation"].mean()),
            "max_period_plan_change_pct": float(group["plan_change_pct"].max()),
        }
        record.update(summarize_strategy_metadata(group))
        records.append(
            record
        )
    return pd.DataFrame(records)


def summarize_decisions_for_scenario(
    decisions: pd.DataFrame,
    config: Mapping[str, object],
    run_mode: str,
    grain_level: str,
    intermittency_bucket: str,
    scenario_name: str,
    lambda_execution: float,
) -> pd.DataFrame:
    """Return normalized strategy summary for one M5 scenario."""
    weights = dict(config.get("planning_loss_weights", {}))
    weights["lambda_execution"] = float(lambda_execution)
    summary = summarize_planning_utility(decisions, weights)
    summary, _, _ = add_normalized_planning_loss(
        summary,
        weights,
        reference_strategy="global_best_model",
        dataset_name="m5",
        run_mode=run_mode,
        split_name="test",
    )
    return finalize_summary(summary, dataset_name="m5", run_mode=run_mode, grain_level=grain_level, intermittency_bucket=intermittency_bucket, scenario_name=scenario_name)


def finalize_summary(
    summary: pd.DataFrame,
    dataset_name: str,
    run_mode: str,
    grain_level: str,
    intermittency_bucket: str,
    scenario_name: str,
) -> pd.DataFrame:
    """Return output columns required by the M5 robustness analysis."""
    table = ensure_strategy_metadata(summary)
    table["dataset_name"] = dataset_name
    table["run_mode"] = run_mode
    table["grain_level"] = grain_level
    table["intermittency_bucket"] = intermittency_bucket
    table["scenario_name"] = scenario_name
    table["method_name"] = table["strategy"].map(short_strategy_label)
    table["WAPE"] = table["weighted_absolute_percentage_error"]
    table["inventory_cost"] = table["total_inventory_cost"]
    table["planning_volatility"] = table["planning_signal_volatility_total"]
    table["execution_penalty"] = table["execution_adaptation_penalty_total"]
    oracle_loss = table.loc[table["strategy"] == "oracle_realized_demand", "normalized_total_loss"]
    oracle_value = float(oracle_loss.iloc[0]) if not oracle_loss.empty else np.nan
    table["gap_to_oracle"] = table["normalized_total_loss"] - oracle_value
    table["rank_by_WAPE"] = table["WAPE"].rank(method="min").astype(int)
    table["rank_by_execution_penalty"] = table["execution_penalty"].rank(method="min").astype(int)
    table["rank_by_normalized_total_loss"] = table["normalized_total_loss"].rank(method="min").astype(int)
    return table[REQUIRED_SUMMARY_COLUMNS + ["strategy"]].sort_values(["scenario_name", "rank_by_normalized_total_loss", "method_name"]).reset_index(drop=True)


def run_hierarchy_sensitivity(modeling_table: pd.DataFrame, config: Mapping[str, object], run_mode: str) -> pd.DataFrame:
    """Run hierarchy sensitivity across item-store, department-store, and category-store grains."""
    rows = []
    for grain_level in ["item_store", "department_store", "category_store"]:
        grain_table = modeling_table if grain_level == "item_store" else aggregate_to_grain(modeling_table, grain_level)
        summary, _ = run_grain_experiment(grain_table, config, run_mode, grain_level, "baseline", None)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def aggregate_to_grain(modeling_table: pd.DataFrame, grain_level: str) -> pd.DataFrame:
    """Aggregate selected M5 data to a higher planning grain and rebuild features."""
    if grain_level == "department_store":
        group_columns = ["date", "dept_id", "cat_id", "store_id", "state_id", "split"]
        id_columns = ["dept_id", "store_id"]
    elif grain_level == "category_store":
        group_columns = ["date", "cat_id", "store_id", "state_id", "split"]
        id_columns = ["cat_id", "store_id"]
    else:
        raise ValueError("Unsupported M5 hierarchy grain: {}".format(grain_level))
    aggregate = (
        modeling_table.groupby(group_columns, dropna=False)
        .agg(
            demand=("demand", "sum"),
            sell_price=("sell_price", "mean"),
            has_event=("has_event", "max"),
            event_count=("event_count", "max"),
            snap_active=("snap_active", "max"),
            price_available=("price_available", "max"),
        )
        .reset_index()
    )
    if grain_level == "department_store":
        aggregate["item_id"] = aggregate["dept_id"]
        aggregate["series_id"] = aggregate["dept_id"].astype(str) + "__" + aggregate["store_id"].astype(str)
    else:
        aggregate["dept_id"] = aggregate["cat_id"]
        aggregate["item_id"] = aggregate["cat_id"]
        aggregate["series_id"] = aggregate["cat_id"].astype(str) + "__" + aggregate["store_id"].astype(str)
    aggregate["horizon"] = aggregate.groupby(["series_id", "split"]).cumcount() + 1
    from data_loaders.m5_loader import add_m5_features

    return add_m5_features(aggregate)


def run_intermittent_demand_stress(decisions: pd.DataFrame, config: Mapping[str, object], run_mode: str) -> pd.DataFrame:
    """Summarize item-store results by intermittency bucket."""
    profile = decisions.groupby("series_id")["actual"].agg(zero_rate=lambda values: float((values == 0).mean()), mean_demand="mean", std_demand="std").reset_index()
    profile["volatility_ratio"] = profile["std_demand"].fillna(0.0) / profile["mean_demand"].replace(0.0, np.nan).fillna(1.0)
    profile["intermittency_score"] = profile["zero_rate"] + profile["volatility_ratio"].clip(upper=5.0) / 5.0
    try:
        profile["intermittency_bucket"] = pd.qcut(
            profile["intermittency_score"],
            q=3,
            labels=["low_intermittency", "medium_intermittency", "high_intermittency"],
            duplicates="drop",
        )
    except ValueError:
        profile["intermittency_bucket"] = "medium_intermittency"
    profile["intermittency_bucket"] = profile["intermittency_bucket"].astype(str)
    frame = decisions.merge(profile[["series_id", "intermittency_bucket"]], on="series_id", how="left")
    rows = []
    lambda_execution = float(config.get("planning_loss_weights", {}).get("lambda_execution", 1.0))
    for bucket, group in frame.groupby("intermittency_bucket"):
        rows.append(
            summarize_decisions_for_scenario(
                group,
                config=config,
                run_mode=run_mode,
                grain_level="item_store",
                intermittency_bucket=str(bucket),
                scenario_name="baseline",
                lambda_execution=lambda_execution,
            )
        )
    return pd.concat(rows, ignore_index=True)


def load_dataco_execution_scenarios(config: Mapping[str, object], table_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load generated DataCo execution scenarios or configured fallbacks."""
    candidate_paths = [
        table_dir / "generated_execution_risk_scenarios.csv",
        Path("outputs/tables/generated_execution_risk_scenarios.csv"),
    ]
    for path in candidate_paths:
        if path.exists():
            scenarios = pd.read_csv(path)
            if {"scenario_name", "lambda_execution"}.issubset(scenarios.columns):
                logger.info("Loaded DataCo-informed execution scenarios from %s.", path)
                return scenarios
    logger.warning("Generated DataCo scenario table was unavailable. Using configured M5 execution-risk fallbacks.")
    fallback = config.get("execution_risk_scenarios_fallback", {})
    records = []
    for scenario_name in ["baseline", "dataco_low", "dataco_median", "dataco_high", "dataco_severe"]:
        records.append(
            {
                "scenario_name": scenario_name,
                "lambda_execution": float(fallback.get(scenario_name, {}).get("lambda_execution", 0.10)),
                "source": "config_fallback",
                "fallback_used": True,
            }
        )
    return pd.DataFrame(records)


def write_m5_outputs(
    large_scale: pd.DataFrame,
    hierarchy: pd.DataFrame,
    intermittent: pd.DataFrame,
    dataco_scenario: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    table_dir: Path,
    paper_table_dir: Path,
) -> None:
    """Write M5 CSV and LaTeX table outputs."""
    outputs = {
        "m5_robustness_summary": robustness_summary,
        "m5_large_scale_replication": large_scale,
        "m5_hierarchy_sensitivity": hierarchy,
        "m5_intermittent_demand_stress": intermittent,
        "m5_dataco_scenario_robustness": dataco_scenario,
    }
    for stem, data in outputs.items():
        data.to_csv(table_dir / "{}.csv".format(stem), index=False)
        export_summary_table(
            data=data,
            table_name="{}_table".format(stem),
            output_dir=paper_table_dir,
            caption=m5_table_caption(stem),
            label="tab:{}".format(stem.replace("_", "-")),
            numeric_precision=3,
            resize_to_textwidth=True,
        )


def m5_table_caption(stem: str) -> str:
    """Return an English caption for a M5 table."""
    captions = {
        "m5_robustness_summary": "M5 robustness summary across scale, hierarchy, intermittency, and execution scenarios.",
        "m5_large_scale_replication": "M5 large-scale replication of the accuracy-feasibility planning comparison.",
        "m5_hierarchy_sensitivity": "M5 hierarchy sensitivity by planning grain.",
        "m5_intermittent_demand_stress": "M5 intermittent-demand stress-test summary.",
        "m5_dataco_scenario_robustness": "M5 DataCo-informed execution scenario robustness summary.",
    }
    return captions.get(stem, stem.replace("_", " ").title())


def write_m5_figures(
    large_scale: pd.DataFrame,
    hierarchy: pd.DataFrame,
    intermittent: pd.DataFrame,
    dataco_scenario: pd.DataFrame,
    figure_dir: Path,
    paper_figure_dir: Path,
) -> None:
    """Write M5 robustness figures as PNG and PDF."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    paper_figure_dir.mkdir(parents=True, exist_ok=True)
    save_m5_scatter(
        large_scale,
        x_column="WAPE",
        y_column="execution_penalty",
        x_label="Weighted Absolute Percentage Error",
        y_label="Execution Penalty",
        png_path=figure_dir / "m5_accuracy_vs_execution_penalty.png",
        pdf_path=paper_figure_dir / "m5_accuracy_vs_execution_penalty.pdf",
    )
    save_m5_line(
        hierarchy,
        x_column="grain_level",
        y_column="normalized_total_loss",
        x_label="Planning Grain",
        y_label="Normalized Total Loss",
        png_path=figure_dir / "m5_hierarchy_sensitivity.png",
        pdf_path=paper_figure_dir / "m5_hierarchy_sensitivity.pdf",
    )
    save_m5_line(
        intermittent,
        x_column="intermittency_bucket",
        y_column="normalized_total_loss",
        x_label="Intermittency Bucket",
        y_label="Normalized Total Loss",
        png_path=figure_dir / "m5_intermittent_demand_stress.png",
        pdf_path=paper_figure_dir / "m5_intermittent_demand_stress.pdf",
    )
    save_m5_line(
        dataco_scenario,
        x_column="scenario_name",
        y_column="normalized_total_loss",
        x_label="DataCo-Informed Execution Scenario",
        y_label="Normalized Total Loss",
        png_path=figure_dir / "m5_dataco_scenario_robustness.png",
        pdf_path=paper_figure_dir / "m5_dataco_scenario_robustness.pdf",
    )


def save_m5_scatter(data: pd.DataFrame, x_column: str, y_column: str, x_label: str, y_label: str, png_path: Path, pdf_path: Path) -> None:
    """Save a labeled M5 scatter figure."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.35, 4.45))
    plot_data = data.dropna(subset=[x_column, y_column]).copy()
    for index, row in enumerate(plot_data.itertuples(index=False)):
        ax.scatter(
            getattr(row, x_column),
            getattr(row, y_column),
            marker=strategy_marker(getattr(row, "strategy")),
            color=strategy_color(getattr(row, "strategy"), index),
            edgecolor="white",
            linewidth=0.6,
            s=72,
            zorder=3,
        )
        ax.annotate(
            getattr(row, "method_name"),
            (getattr(row, x_column), getattr(row, y_column)),
            xytext=(5, 5 if index % 2 == 0 else -8),
            textcoords="offset points",
            fontsize=7.1,
        )
    format_axis(ax, x_label=x_label, y_label=y_label, grid_axis="both")
    save_paper_figure(fig, png_path, pdf_path)


def save_m5_line(data: pd.DataFrame, x_column: str, y_column: str, x_label: str, y_label: str, png_path: Path, pdf_path: Path) -> None:
    """Save a M5 line figure by method."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.35, 4.45))
    plot_data = data.copy()
    x_order = list(dict.fromkeys(plot_data[x_column].astype(str).tolist()))
    positions = {value: index for index, value in enumerate(x_order)}
    for index, strategy in enumerate(M5_STRATEGY_ORDER):
        strategy_data = plot_data[plot_data["strategy"] == strategy].copy()
        if strategy_data.empty:
            continue
        strategy_data["x_position"] = strategy_data[x_column].astype(str).map(positions)
        strategy_data = strategy_data.sort_values("x_position")
        ax.plot(
            strategy_data["x_position"],
            strategy_data[y_column],
            marker=strategy_marker(strategy),
            color=strategy_color(strategy, index),
            linestyle=strategy_linestyle(strategy),
            linewidth=1.7,
            label=short_strategy_label(strategy),
        )
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([label.replace("_", "\n").title() for label in x_order])
    format_axis(ax, x_label=x_label, y_label=y_label)
    place_legend(ax, columns=3)
    save_paper_figure(fig, png_path, pdf_path)


def short_strategy_label(strategy: str) -> str:
    """Return compact English strategy labels."""
    labels = {
        "global_best_model": "Global Best",
        "simple_ensemble": "Simple Ensemble",
        "operational_loss_ensemble": "Operational Ensemble",
        "feasibility_aware_smoothed_alpha_0_25": "Smoothed Alpha 0.25",
        "feasibility_aware_selector": "Feasibility-Aware",
        "greedy_feasibility_selector": "Greedy Feasibility",
        "dp_feasibility_selector": "DP Feasibility",
        "budgeted_dp_feasibility_selector": "Budgeted DP",
        "best_stability_model": "Best Stability",
        "oracle_dp_feasibility_selector": "Realized-Inventory Oracle DP",
        "oracle_realized_demand": "Realized Demand Oracle",
    }
    return labels.get(strategy, strategy.replace("_", " ").title())


if __name__ == "__main__":
    main()
