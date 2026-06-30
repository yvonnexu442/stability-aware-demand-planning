"""Placeholder loader for the Corporacion Favorita demand dataset."""

from pathlib import Path
from typing import Union

import pandas as pd


def load_favorita_dataset(raw_data_dir: Union[str, Path]) -> pd.DataFrame:
    """Load the Favorita dataset after a dataset-specific implementation is added.

    Favorita is a useful future benchmark for item-store-family demand planning,
    but this first repository step does not implement full dataset ingestion.

    Keeping this placeholder makes the intended dataset boundary explicit while
    preventing the project from turning into a data-loading exercise too early.
    """
    raise NotImplementedError("Favorita loading will be implemented in a later dataset integration step.")
