"""Chronological splitting utilities for demand planning experiments."""

from typing import Dict, List, Optional

import pandas as pd


def chronological_split(
    data: pd.DataFrame,
    date_column: str,
    train_start_date: Optional[str] = None,
    train_end_date: Optional[str] = None,
    validation_start_date: Optional[str] = None,
    validation_end_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
    test_end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Return train, validation, and test frames using date boundaries.

    Chronological splits prevent lookahead leakage. This is essential for demand
    planning research because plans must be evaluated as if future demand were
    unknown at the time the plan was made.

    Boundaries may be omitted, in which case the corresponding side of the date
    range is left open.
    """
    frame = data.copy()
    frame[date_column] = pd.to_datetime(frame[date_column])

    def _between(start_date: Optional[str], end_date: Optional[str]) -> pd.Series:
        mask = pd.Series(True, index=frame.index)
        if start_date is not None:
            mask = mask & (frame[date_column] >= pd.to_datetime(start_date))
        if end_date is not None:
            mask = mask & (frame[date_column] <= pd.to_datetime(end_date))
        return mask

    return {
        "train": frame.loc[_between(train_start_date, train_end_date)].copy(),
        "validation": frame.loc[_between(validation_start_date, validation_end_date)].copy(),
        "test": frame.loc[_between(test_start_date, test_end_date)].copy(),
    }


def rolling_time_windows(
    data: pd.DataFrame,
    date_column: str,
    train_window_size: int,
    validation_window_size: int,
    test_window_size: int,
    step_size: int,
) -> List[Dict[str, pd.Timestamp]]:
    """Return rolling chronological window boundaries.

    Rolling windows support repeated operational backtests. They matter because
    planning systems update over time, and the stability problem can look
    different across different historical contexts.

    The function returns boundaries only; later scripts can use those boundaries
    to create actual train, validation, and test frames.
    """
    dates = pd.Series(pd.to_datetime(data[date_column]).dropna().unique()).sort_values().reset_index(drop=True)
    total_window = int(train_window_size) + int(validation_window_size) + int(test_window_size)
    windows = []
    start = 0
    while start + total_window <= len(dates):
        train_start = dates.iloc[start]
        train_end = dates.iloc[start + int(train_window_size) - 1]
        validation_start = dates.iloc[start + int(train_window_size)]
        validation_end = dates.iloc[start + int(train_window_size) + int(validation_window_size) - 1]
        test_start = dates.iloc[start + int(train_window_size) + int(validation_window_size)]
        test_end = dates.iloc[start + total_window - 1]
        windows.append(
            {
                "train_start_date": train_start,
                "train_end_date": train_end,
                "validation_start_date": validation_start,
                "validation_end_date": validation_end,
                "test_start_date": test_start,
                "test_end_date": test_end,
            }
        )
        start += int(step_size)
    return windows
