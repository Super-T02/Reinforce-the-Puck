import logging
import logging.config
import os

import yaml


def init_logger(path: str | None = None):
    """Initialize the logger.

    Args:
        path (str | None): Path to save the log file.
    """
    if path is not None:
        if not path.endswith(".yaml"):
            raise ValueError("Invalid log configuration file format. Must be a YAML.")

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        # Ensure the log directory exists
        log_dir = os.path.dirname(config["handlers"]["file"]["filename"])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    logging.info("Logging initialized")
