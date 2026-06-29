"""Data loading and normalization utilities."""

from __future__ import print_function

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


STANDARD_COLUMNS = ["date", "item_id", "demand"]


def read_registry(registry_path):
    with open(registry_path, "r") as handle:
        return json.load(handle)


def make_synthetic_panel(n_items=8, n_periods=180, seed=7):
    """Generate a small daily panel with trend, weekly seasonality, and shocks."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2021-01-01", periods=n_periods, freq="D")
    rows = []

    for item_idx in range(n_items):
        base = rng.uniform(35.0, 120.0)
        trend = rng.uniform(-0.04, 0.08)
        weekly_phase = rng.uniform(0.0, 2.0 * np.pi)
        weekly_amp = rng.uniform(0.08, 0.28)
        noise_scale = rng.uniform(3.0, 12.0)
        shock_days = set(rng.choice(np.arange(n_periods), size=max(2, n_periods // 30), replace=False))

        for t, date in enumerate(dates):
            weekly = 1.0 + weekly_amp * np.sin((2.0 * np.pi * t / 7.0) + weekly_phase)
            level = base * (1.0 + trend * t / max(n_periods, 1))
            shock = rng.uniform(0.25, 0.55) * base if t in shock_days else 0.0
            observed = max(0.0, level * weekly + shock + rng.normal(0.0, noise_scale))
            rows.append(
                {
                    "date": date,
                    "item_id": "item_{:03d}".format(item_idx + 1),
                    "demand": float(round(observed, 2)),
                }
            )

    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def load_dataset(
    dataset_key,
    registry_path="configs/datasets.json",
    raw_dir="data/raw",
    processed_dir="data/processed",
    force_download=False,
):
    """Load a dataset and return a standardized daily panel."""
    registry = read_registry(registry_path)
    if dataset_key not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError("Unknown dataset '{}'. Available datasets: {}".format(dataset_key, available))

    entry = registry[dataset_key]
    dataset_type = entry.get("type")
    if dataset_type == "synthetic":
        return make_synthetic_panel(
            n_items=int(entry.get("n_items", 8)),
            n_periods=int(entry.get("n_periods", 180)),
            seed=int(entry.get("seed", 7)),
        )

    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    processed_file = processed_path / "{}.csv".format(dataset_key)
    if processed_file.exists() and not force_download:
        return _read_standard_csv(processed_file)

    archive_path = raw_path / entry["filename"]
    if force_download or not archive_path.exists():
        download_url(entry["url"], archive_path)

    extracted_path = extract_zip_member(archive_path, entry["member"], raw_path / dataset_key)
    panel = standardize_dataset(extracted_path, entry)
    panel.to_csv(processed_file, index=False)
    return panel


def download_url(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        with open(destination, "wb") as output:
            shutil.copyfileobj(response, output)


def extract_zip_member(archive_path, member, output_dir):
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / Path(member).name

    with zipfile.ZipFile(str(archive_path), "r") as archive:
        with archive.open(member) as source:
            with open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)

    return target_path


def standardize_dataset(path, entry):
    recipe = entry.get("recipe")
    if recipe == "uci_bike_day":
        return standardize_uci_bike_day(path)
    if recipe == "uci_online_retail":
        return standardize_uci_online_retail(path, top_items=int(entry.get("top_items", 50)))
    raise ValueError("No standardization recipe configured for '{}'".format(recipe))


def standardize_uci_bike_day(path):
    raw = pd.read_csv(path)
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["dteday"]),
            "item_id": "bike_total",
            "demand": raw["cnt"].astype(float),
        }
    )
    return panel[STANDARD_COLUMNS].sort_values(["item_id", "date"]).reset_index(drop=True)


def standardize_uci_online_retail(path, top_items=50):
    try:
        raw = pd.read_excel(path)
    except ImportError as exc:
        raise RuntimeError(
            "Reading Online Retail requires an Excel reader. Install openpyxl or xlrd."
        ) from exc

    required = {"StockCode", "Quantity", "InvoiceDate", "UnitPrice"}
    missing = required.difference(set(raw.columns))
    if missing:
        raise ValueError("Online Retail file is missing expected columns: {}".format(sorted(missing)))

    filtered = raw.copy()
    filtered = filtered[filtered["Quantity"].notnull()]
    filtered = filtered[filtered["InvoiceDate"].notnull()]
    filtered = filtered[filtered["Quantity"] > 0]
    filtered = filtered[filtered["UnitPrice"] >= 0]
    filtered["date"] = pd.to_datetime(filtered["InvoiceDate"]).dt.floor("D")
    filtered["item_id"] = filtered["StockCode"].astype(str)

    panel = (
        filtered.groupby(["date", "item_id"], as_index=False)["Quantity"]
        .sum()
        .rename(columns={"Quantity": "demand"})
    )

    top_item_ids = (
        panel.groupby("item_id")["demand"]
        .sum()
        .sort_values(ascending=False)
        .head(top_items)
        .index
    )
    panel = panel[panel["item_id"].isin(top_item_ids)]
    panel["demand"] = panel["demand"].astype(float)
    return complete_daily_panel(panel[STANDARD_COLUMNS])


def complete_daily_panel(panel):
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    frames = []
    full_dates = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")

    for item_id, item_frame in panel.groupby("item_id"):
        base = pd.DataFrame({"date": full_dates})
        base["item_id"] = str(item_id)
        merged = base.merge(item_frame, on=["date", "item_id"], how="left")
        merged["demand"] = merged["demand"].fillna(0.0)
        frames.append(merged)

    completed = pd.concat(frames, ignore_index=True)
    return completed[STANDARD_COLUMNS].sort_values(["item_id", "date"]).reset_index(drop=True)


def _read_standard_csv(path):
    panel = pd.read_csv(path)
    missing = set(STANDARD_COLUMNS).difference(set(panel.columns))
    if missing:
        raise ValueError("Standard data file is missing columns: {}".format(sorted(missing)))
    panel = panel[STANDARD_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["item_id"] = panel["item_id"].astype(str)
    panel["demand"] = panel["demand"].astype(float)
    return panel.sort_values(["item_id", "date"]).reset_index(drop=True)
