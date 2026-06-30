"""Deterministic feasibility-aware selectors for finite-horizon planning."""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from decision_layer.no_leakage import require_no_future_outcomes
from planning_environment.planning_actions import forecast_to_inventory_target


LossKey = Tuple[str, str]


@dataclass(frozen=True)
class SelectorState:
    """State retained by dynamic selectors."""

    cost: float
    path: Tuple[str, ...]
    rows: Tuple[int, ...]
    switches: int
    previous_plan: float


class GreedyFeasibilitySelector:
    """Select the lowest one-step expected operational-cost model per period.

    The selector is deployable: candidate scoring uses validation-derived
    forecast and inventory-cost estimates plus current forecast-implied plan
    movement. It does not use realized test-period demand.
    """

    def __init__(
        self,
        expected_losses: Mapping[LossKey, Mapping[str, float]],
        global_expected_losses: Optional[Mapping[str, Mapping[str, float]]] = None,
        weights: Optional[Mapping[str, float]] = None,
        switch_penalty: float = 0.02,
        max_plan_change_rate: float = 0.20,
        calibration_group_column: str = "series_id",
        strategy_name: str = "greedy_feasibility_selector",
    ) -> None:
        self.expected_losses = expected_losses
        self.global_expected_losses = global_expected_losses or {}
        self.weights = weights or {}
        self.switch_penalty = float(switch_penalty)
        self.max_plan_change_rate = float(max_plan_change_rate)
        self.calibration_group_column = calibration_group_column
        self.strategy_name = strategy_name

    def select(self, candidate_forecasts: pd.DataFrame) -> pd.DataFrame:
        """Return selected forecast rows for all series and periods."""
        require_no_future_outcomes(candidate_forecasts, "{}.select".format(self.__class__.__name__))
        selected_rows: List[pd.Series] = []
        for _, series_frame in candidate_forecasts.sort_values(["series_id", "date", "model_name"]).groupby("series_id", sort=False):
            previous_model = None
            previous_plan = None
            for _, period_frame in series_frame.groupby("date", sort=False):
                scored_candidates = []
                for row in _period_candidates(period_frame):
                    score, plan = self._stage_cost(row, previous_model, previous_plan)
                    scored_candidates.append((score, str(row["model_name"]), plan, row))
                _, _, selected_plan, selected = min(scored_candidates, key=lambda value: (value[0], value[1]))
                selected_rows.append(selected)
                previous_model = str(selected["model_name"])
                previous_plan = selected_plan
        return _selected_frame(selected_rows, self.strategy_name)

    def _stage_cost(
        self,
        row: pd.Series,
        previous_model: Optional[str],
        previous_plan: Optional[float],
    ) -> Tuple[float, float]:
        plan = _planning_signal(row)
        base_cost = self._base_expected_cost(row)
        plan_change_pct, execution_violation = _transition_burden(
            current_plan=plan,
            previous_plan=previous_plan,
            max_plan_change_rate=self.max_plan_change_rate,
        )
        switch_cost = 0.0 if previous_model is None or previous_model == str(row["model_name"]) else self.switch_penalty
        score = (
            base_cost
            + float(self.weights.get("lambda_volatility", 0.5)) * plan_change_pct
            + float(self.weights.get("lambda_switch", 0.5)) * switch_cost
            + float(self.weights.get("lambda_execution", 1.0)) * execution_violation
        )
        return float(score), plan

    def _base_expected_cost(self, row: pd.Series) -> float:
        model_name = str(row["model_name"])
        calibration_group = str(row.get(self.calibration_group_column, row.get("series_id", "global")))
        losses = self.expected_losses.get(
            (calibration_group, model_name),
            self.global_expected_losses.get(model_name, self.expected_losses.get(("global", model_name), {})),
        )
        return (
            float(self.weights.get("alpha_forecast", 1.0)) * float(losses.get("wape", 1.0))
            + float(self.weights.get("beta_inventory", 1.0)) * float(losses.get("inventory_cost_per_demand_unit", 0.0))
        )


class DPFeasibilitySelector(GreedyFeasibilitySelector):
    """Finite-horizon dynamic-programming selector for cumulative loss."""

    def __init__(
        self,
        expected_losses: Mapping[LossKey, Mapping[str, float]],
        global_expected_losses: Optional[Mapping[str, Mapping[str, float]]] = None,
        weights: Optional[Mapping[str, float]] = None,
        switch_penalty: float = 0.02,
        max_plan_change_rate: float = 0.20,
        calibration_group_column: str = "series_id",
        strategy_name: str = "dp_feasibility_selector",
    ) -> None:
        super().__init__(
            expected_losses=expected_losses,
            global_expected_losses=global_expected_losses,
            weights=weights,
            switch_penalty=switch_penalty,
            max_plan_change_rate=max_plan_change_rate,
            calibration_group_column=calibration_group_column,
            strategy_name=strategy_name,
        )

    def select(self, candidate_forecasts: pd.DataFrame) -> pd.DataFrame:
        """Return selected forecast rows from finite-horizon DP."""
        require_no_future_outcomes(candidate_forecasts, "{}.select".format(self.__class__.__name__))
        selected_rows: List[pd.Series] = []
        for _, series_frame in candidate_forecasts.sort_values(["series_id", "date", "model_name"]).groupby("series_id", sort=False):
            selected_rows.extend(self._select_series(series_frame))
        return _selected_frame(selected_rows, self.strategy_name)

    def _select_series(self, series_frame: pd.DataFrame) -> List[pd.Series]:
        dp: Dict[str, SelectorState] = {}
        for _, period_frame in series_frame.groupby("date", sort=False):
            candidates = _period_candidates(period_frame)
            next_dp: Dict[str, SelectorState] = {}
            if not dp:
                for row in candidates:
                    model_name = str(row["model_name"])
                    stage_cost, plan = self._stage_cost(row, previous_model=None, previous_plan=None)
                    next_dp[model_name] = SelectorState(
                        cost=stage_cost,
                        path=(model_name,),
                        rows=(int(row.name),),
                        switches=0,
                        previous_plan=plan,
                    )
            else:
                for row in candidates:
                    model_name = str(row["model_name"])
                    best_state = None
                    for previous_model, state in dp.items():
                        stage_cost, plan = self._stage_cost(row, previous_model=previous_model, previous_plan=state.previous_plan)
                        switches = state.switches + int(previous_model != model_name)
                        candidate_state = SelectorState(
                            cost=state.cost + stage_cost,
                            path=state.path + (model_name,),
                            rows=state.rows + (int(row.name),),
                            switches=switches,
                            previous_plan=plan,
                        )
                        best_state = _min_state(best_state, candidate_state)
                    if best_state is not None:
                        next_dp[model_name] = best_state
            dp = next_dp
        if not dp:
            return []
        final_state = min(dp.values(), key=_state_sort_key)
        return [series_frame.loc[index] for index in final_state.rows]


class BudgetedDPFeasibilitySelector(DPFeasibilitySelector):
    """Finite-horizon DP selector with a hard maximum switch budget."""

    def __init__(
        self,
        expected_losses: Mapping[LossKey, Mapping[str, float]],
        global_expected_losses: Optional[Mapping[str, Mapping[str, float]]] = None,
        weights: Optional[Mapping[str, float]] = None,
        switch_penalty: float = 0.02,
        max_plan_change_rate: float = 0.20,
        max_switches: int = 2,
        calibration_group_column: str = "series_id",
        strategy_name: str = "budgeted_dp_feasibility_selector",
    ) -> None:
        super().__init__(
            expected_losses=expected_losses,
            global_expected_losses=global_expected_losses,
            weights=weights,
            switch_penalty=switch_penalty,
            max_plan_change_rate=max_plan_change_rate,
            calibration_group_column=calibration_group_column,
            strategy_name=strategy_name,
        )
        self.max_switches = int(max_switches)

    def _select_series(self, series_frame: pd.DataFrame) -> List[pd.Series]:
        dp: Dict[Tuple[str, int], SelectorState] = {}
        for _, period_frame in series_frame.groupby("date", sort=False):
            candidates = _period_candidates(period_frame)
            next_dp: Dict[Tuple[str, int], SelectorState] = {}
            if not dp:
                for row in candidates:
                    model_name = str(row["model_name"])
                    stage_cost, plan = self._stage_cost(row, previous_model=None, previous_plan=None)
                    state = SelectorState(
                        cost=stage_cost,
                        path=(model_name,),
                        rows=(int(row.name),),
                        switches=0,
                        previous_plan=plan,
                    )
                    next_dp[(model_name, 0)] = _min_state(next_dp.get((model_name, 0)), state)
            else:
                for row in candidates:
                    model_name = str(row["model_name"])
                    for (previous_model, used_switches), state in dp.items():
                        new_switches = used_switches + int(previous_model != model_name)
                        if new_switches > self.max_switches:
                            continue
                        stage_cost, plan = self._stage_cost(row, previous_model=previous_model, previous_plan=state.previous_plan)
                        candidate_state = SelectorState(
                            cost=state.cost + stage_cost,
                            path=state.path + (model_name,),
                            rows=state.rows + (int(row.name),),
                            switches=new_switches,
                            previous_plan=plan,
                        )
                        key = (model_name, new_switches)
                        next_dp[key] = _min_state(next_dp.get(key), candidate_state)
            dp = next_dp
            if not dp:
                raise ValueError("No feasible DP path remains under the configured switch budget.")
        if not dp:
            return []
        final_state = min(dp.values(), key=_state_sort_key)
        return [series_frame.loc[index] for index in final_state.rows]


def _period_candidates(period_frame: pd.DataFrame) -> List[pd.Series]:
    """Return deterministic candidate rows for a single period."""
    return [
        row
        for _, row in period_frame.sort_values("model_name").drop_duplicates(["model_name"], keep="first").iterrows()
    ]


def _selected_frame(rows: Sequence[pd.Series], strategy_name: str) -> pd.DataFrame:
    """Return selected rows as a strategy decision frame."""
    frame = pd.DataFrame([row.to_dict() for row in rows])
    if frame.empty:
        return frame
    frame["selected_model"] = frame["model_name"].astype(str)
    frame["strategy"] = strategy_name
    return frame


def _planning_signal(row: pd.Series) -> float:
    """Return forecast-implied inventory target for a candidate row."""
    return float(forecast_to_inventory_target([float(row["forecast"])], [float(row.get("safety_stock", 0.0))])[0])


def _transition_burden(
    current_plan: float,
    previous_plan: Optional[float],
    max_plan_change_rate: float,
) -> Tuple[float, float]:
    """Return normalized plan volatility and execution violation burden."""
    if previous_plan is None:
        return 0.0, 0.0
    previous_value = float(previous_plan)
    plan_change_abs = abs(float(current_plan) - previous_value)
    scale = max(abs(previous_value), 1e-8)
    plan_change_pct = plan_change_abs / scale
    execution_capacity = abs(previous_value) * float(max_plan_change_rate)
    execution_violation = max(plan_change_abs - execution_capacity, 0.0) / scale
    return float(plan_change_pct), float(execution_violation)


def _state_sort_key(state: SelectorState) -> Tuple[float, int, Tuple[str, ...]]:
    """Return deterministic state sort key."""
    rounded_cost = float(np.round(state.cost, 12))
    return rounded_cost, int(state.switches), state.path


def _min_state(left: Optional[SelectorState], right: SelectorState) -> SelectorState:
    """Return the lower-cost deterministic state."""
    if left is None:
        return right
    return min([left, right], key=_state_sort_key)
