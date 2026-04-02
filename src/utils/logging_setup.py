"""
Logging configuration for the Wildlife Detector pipeline.

Call ``setup_logging(cfg)`` once at application startup.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config import Config


def setup_logging(cfg: Config) -> None:
    """Configure the root logger based on the pipeline config.

    Args:
        cfg: Pipeline configuration object.
    """
    log_cfg = cfg.logging
    level = getattr(logging, str(log_cfg.level).upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if bool(log_cfg.log_to_file):
        log_dir = Path(str(log_cfg.log_dir))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "wildlife_detector.log")
        handlers.append(file_handler)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
