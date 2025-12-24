"""Logging utilities."""

import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger
    """
    return logging.getLogger(name)


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    format_string: str | None = None,
) -> None:
    """Set up logging configuration.

    Args:
        level: Logging level
        log_file: Optional file to log to
        format_string: Custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
    )


class ProgressLogger:
    """Simple progress logger for long-running operations."""

    def __init__(self, name: str, total: int, log_interval: int = 10):
        """Initialize progress logger.

        Args:
            name: Operation name
            total: Total number of items
            log_interval: Log every N percent
        """
        self.name = name
        self.total = total
        self.log_interval = log_interval
        self.current = 0
        self.last_logged = -1
        self.logger = get_logger(__name__)

    def update(self, n: int = 1) -> None:
        """Update progress.

        Args:
            n: Number of items completed
        """
        self.current += n
        pct = int(100 * self.current / self.total) if self.total > 0 else 100
        if pct >= self.last_logged + self.log_interval:
            self.logger.info(f"{self.name}: {pct}% ({self.current}/{self.total})")
            self.last_logged = pct

    def finish(self) -> None:
        """Log completion."""
        self.logger.info(f"{self.name}: Complete ({self.total}/{self.total})")
