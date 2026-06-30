"""Download raw public demand datasets into the expected raw data layout."""

import argparse
import logging
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Mapping

from utils.logging_utils import setup_logger


DATA_SOURCES: Dict[str, Dict[str, str]] = {
    "favorita": {
        "source_type": "kaggle_competition",
        "slug": "favorita-grocery-sales-forecasting",
        "description": "Corporacion Favorita grocery sales forecasting data.",
    },
    "m5": {
        "source_type": "kaggle_competition",
        "slug": "m5-forecasting-accuracy",
        "description": "M5 Forecasting Accuracy retail demand data.",
    },
    "walmart": {
        "source_type": "kaggle_competition",
        "slug": "walmart-recruiting-store-sales-forecasting",
        "description": "Walmart store sales forecasting data.",
    },
    "rossmann": {
        "source_type": "kaggle_competition",
        "slug": "rossmann-store-sales",
        "description": "Rossmann Store Sales data.",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for raw data download."""
    parser = argparse.ArgumentParser(description="Download raw benchmark demand datasets.")
    parser.add_argument("--raw-data-dir", default="data/raw")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(DATA_SOURCES.keys()),
        choices=sorted(DATA_SOURCES.keys()),
        help="Dataset keys to download.",
    )
    parser.add_argument(
        "--create-directories-only",
        action="store_true",
        help="Create expected raw directories without downloading data.",
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Unzip downloaded archives inside each dataset directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Create raw data directories and optionally download configured datasets."""
    args = parse_args()
    logger = setup_logger("download_raw_data")
    raw_data_dir = Path(args.raw_data_dir)
    selected_sources = {name: DATA_SOURCES[name] for name in args.datasets}

    create_raw_data_directories(raw_data_dir, selected_sources.keys(), logger)
    if args.create_directories_only:
        logger.info("Created raw data directories without downloading data.")
        return

    ensure_kaggle_cli_available()
    for dataset_name, source in selected_sources.items():
        download_kaggle_competition(
            dataset_name=dataset_name,
            source=source,
            raw_data_dir=raw_data_dir,
            unzip=args.unzip,
            logger=logger,
        )


def create_raw_data_directories(raw_data_dir: Path, dataset_names: Iterable[str], logger: logging.Logger) -> None:
    """Create one raw data directory for each configured dataset."""
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name in dataset_names:
        dataset_dir = raw_data_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured raw data directory exists: %s", dataset_dir)


def ensure_kaggle_cli_available() -> None:
    """Raise a clear error if the Kaggle CLI is not available."""
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI is not available. Install it and configure ~/.kaggle/kaggle.json "
            "before downloading Kaggle-hosted datasets."
        )


def download_kaggle_competition(
    dataset_name: str,
    source: Mapping[str, str],
    raw_data_dir: Path,
    unzip: bool,
    logger: logging.Logger,
) -> None:
    """Download one Kaggle competition archive into its raw data directory."""
    if source.get("source_type") != "kaggle_competition":
        raise ValueError("Unsupported source type for {}: {}".format(dataset_name, source.get("source_type")))

    dataset_dir = raw_data_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    slug = source["slug"]
    command = ["kaggle", "competitions", "download", "-c", slug, "-p", str(dataset_dir)]
    logger.info("Downloading %s from Kaggle competition slug: %s", dataset_name, slug)
    subprocess.run(command, check=True)

    if unzip:
        unzip_archives(dataset_dir, logger)


def unzip_archives(dataset_dir: Path, logger: logging.Logger) -> None:
    """Unzip all zip archives in a dataset raw directory."""
    for archive_path in sorted(dataset_dir.glob("*.zip")):
        logger.info("Unzipping archive: %s", archive_path)
        with zipfile.ZipFile(str(archive_path), "r") as archive:
            archive.extractall(str(dataset_dir))


if __name__ == "__main__":
    main()
