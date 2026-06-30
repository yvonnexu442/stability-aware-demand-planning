"""Entry point placeholder for future full experiment runs."""

from utils.config import load_config
from utils.logging_utils import setup_logger


def main() -> None:
    """Load configuration and report that full experiments are not implemented yet."""
    logger = setup_logger("run_all_experiments")
    config = load_config("configs/default.yaml")
    logger.info("Loaded project configuration for run mode: %s", config["project"]["run_mode"])
    logger.info("Full dataset experiments will be implemented in a later step.")


if __name__ == "__main__":
    main()
