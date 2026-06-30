"""Entry point placeholder for a future single experiment run."""

from utils.config import load_config
from utils.logging_utils import setup_logger


def main() -> None:
    """Load configuration and report that single experiments are not implemented yet."""
    logger = setup_logger("run_single_experiment")
    config = load_config("configs/default.yaml")
    logger.info("Loaded dataset name: %s", config["data"]["dataset_name"])
    logger.info("Single experiment execution will be implemented in a later step.")


if __name__ == "__main__":
    main()
