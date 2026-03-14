"""Logging configuration for the gandalf package."""

import logging
import sys


def configure_logging(level=logging.INFO):
    """Configure logging for the gandalf package.

    Sets up a StreamHandler on stderr with a standard format.
    All gandalf.* loggers inherit this configuration.

    Args:
        level: Logging level (default: logging.INFO).
    """
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger("gandalf")
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
