"""Basic feature engineering utilities for chronological demand tables."""

from typing import Iterable

import pandas as pd


def add_calendar_features(data: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Return a copy of the data with transparent calendar features.

    Calendar features are simple and interpretable, which is important for a
    research repository focused on planning logic rather than leaderboard model
    complexity.

    The function avoids modifying input data in place so experiments remain
    reproducible and easy to audit.
    """
    frame = data.copy()
    dates = pd.to_datetime(frame[date_column])
    frame["day_of_week"] = dates.dt.dayofweek
    frame["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    frame["month"] = dates.dt.month
    frame["year"] = dates.dt.year
    return frame


def add_lag_features(
    data: pd.DataFrame,
    series_id_columns: Iterable[str],
    date_column: str,
    target_column: str,
    lags: Iterable[int],
) -> pd.DataFrame:
    """Return a copy of the data with demand lag features.

    Lag features are common transparent baselines for demand forecasting. They
    are included because they help build understandable candidate forecasts
    before adding more complex models.

    The chronological sort protects against lookahead leakage.
    """
    id_columns = list(series_id_columns)
    frame = data.copy().sort_values(id_columns + [date_column])
    for lag in lags:
        frame["{}_lag_{}".format(target_column, int(lag))] = frame.groupby(id_columns)[target_column].shift(int(lag))
    return frame
