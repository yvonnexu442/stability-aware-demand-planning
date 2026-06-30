"""Profile the DataCo supply chain dataset and save research suitability tables."""

import argparse

from data_loaders.dataco_loader import profile_dataco_dataset
from utils.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DataCo profiling."""
    parser = argparse.ArgumentParser(description="Profile the DataCo supply chain dataset.")
    parser.add_argument("--raw-data-dir", default="data/raw/dataco")
    parser.add_argument("--output-dir", default="outputs/tables")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--skip-access-logs", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run DataCo profiling and save CSV tables for research review."""
    args = parse_args()
    logger = setup_logger("profile_dataco")
    profile_tables = profile_dataco_dataset(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        nrows=args.nrows,
        include_access_logs=not args.skip_access_logs,
    )
    for table_name, frame in profile_tables.items():
        logger.info("Generated %s with %s rows.", table_name, len(frame))
    logger.info("DataCo profiling outputs were written to %s.", args.output_dir)


if __name__ == "__main__":
    main()
