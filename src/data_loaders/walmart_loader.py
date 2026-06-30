"""Placeholder loader for Walmart-style retail demand data."""

from pathlib import Path
from typing import Union

import pandas as pd


def load_walmart_dataset(raw_data_dir: Union[str, Path]) -> pd.DataFrame:
    """Load Walmart-style demand data after a dataset-specific implementation is added.

    Walmart-style data can support item-store operational planning experiments.
    It is intentionally not implemented in this first skeleton.

    The placeholder preserves the future module boundary without adding dataset
    assumptions before the research framework is stable.
    """
    raise NotImplementedError("Walmart loading will be implemented in a later dataset integration step.")
