"""Placeholder loader for the Rossmann sales dataset."""

from pathlib import Path
from typing import Union

import pandas as pd


def load_rossmann_dataset(raw_data_dir: Union[str, Path]) -> pd.DataFrame:
    """Load the Rossmann sales dataset after a dataset-specific implementation is added.

    Rossmann is a possible future service-retail planning dataset. This skeleton
    does not implement its ingestion yet because the first research milestone is
    the stability-aware planning framework.

    The loader will later return a normalized demand table for downstream
    feature engineering and simulation.
    """
    raise NotImplementedError("Rossmann loading will be implemented in a later dataset integration step.")
