"""Features that describe planning signal stability and policy context."""

from typing import Iterable

import pandas as pd


def add_planning_signal_change_features(
    data: pd.DataFrame,
    series_id_columns: Iterable[str],
    date_column: str,
    planning_signal_column: str,
) -> pd.DataFrame:
    """Return a copy with absolute and percentage planning signal changes.

    These features describe the executable plan rather than the raw forecast.
    They matter because the paper studies whether forecast-driven plans change
    faster than execution infrastructure can absorb.

    The first row for each series receives zero change because no prior plan is
    available in the observed window.
    """
    id_columns = list(series_id_columns)
    frame = data.copy().sort_values(id_columns + [date_column])
    previous = frame.groupby(id_columns)[planning_signal_column].shift(1)
    absolute_change = (frame[planning_signal_column] - previous).abs().fillna(0.0)
    denominator = previous.abs().clip(lower=1e-8)
    frame["absolute_planning_signal_change"] = absolute_change
    frame["percentage_planning_signal_change"] = (absolute_change / denominator).fillna(0.0)
    return frame


def add_policy_context_features(data: pd.DataFrame, service_level_target: float, execution_capacity: float) -> pd.DataFrame:
    """Return a copy with policy context columns used by planning simulations.

    Policy context features make planning assumptions visible in experiment
    tables. This is useful for research clarity because a forecast can have
    different operational implications under different service or execution
    constraints.
    """
    frame = data.copy()
    frame["service_level_target"] = float(service_level_target)
    frame["execution_capacity"] = float(execution_capacity)
    return frame
