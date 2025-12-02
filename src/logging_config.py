"""Logging configuration for the Better Elo system."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> None:
    """Set up logging configuration for the application.

    Args:
        level: Logging level (default: INFO).
        format_string: Custom format string for log messages.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set specific logger levels to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: Name of the logger (usually __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


# Initialize logging when module is imported
setup_logging()
