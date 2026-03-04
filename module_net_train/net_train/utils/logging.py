from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


def setup_logger(name: str, level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("%(message)s")

    rich_handler = RichHandler(rich_tracebacks=True)
    rich_handler.setFormatter(fmt)
    logger.addHandler(rich_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
