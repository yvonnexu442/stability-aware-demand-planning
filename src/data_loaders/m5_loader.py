"""Placeholder loader for the M5 retail demand dataset."""

from pathlib import Path
from typing import Union

import pandas as pd


def load_m5_dataset(raw_data_dir: Union[str, Path]) -> pd.DataFrame:
    """Load the M5 dataset after a dataset-specific implementation is added.

    M5-style data is useful for retail demand planning experiments, but this
    first step keeps the repository focused on research logic and core metrics.

    The loader contract will later return a normalized demand table with date,
    planning unit identifiers, and demand columns.
    """
    raise NotImplementedError("M5 loading will be implemented in a later dataset integration step.")
