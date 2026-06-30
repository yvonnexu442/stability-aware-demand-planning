"""Plotting helpers for future tables and figures."""

from pathlib import Path
from typing import Union

import pandas as pd


def save_placeholder_figure_note(output_path: Union[str, Path], message: str) -> Path:
    """Save a text note where a future figure will be generated.

    This placeholder keeps the visualization module boundary explicit without
    introducing figure logic before experiments are defined.

    The function is intentionally simple and should be replaced by actual
    plotting helpers when the paper's figures become clear.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(message), encoding="utf-8")
    return path


def validate_plot_table(data: pd.DataFrame, required_columns: list) -> None:
    """Validate that a table contains the columns needed for a future plot.

    Plot validation protects paper reproducibility. It prevents silent figure
    generation from incomplete or inconsistent experiment outputs.
    """
    missing = set(required_columns).difference(set(data.columns))
    if missing:
        raise ValueError("Plot table is missing required columns: {}".format(sorted(missing)))
