"""Tests for the logging setup module."""

from __future__ import annotations

import logging

from src.config import load_config
from src.utils.logging_setup import setup_logging


def test_setup_logging_configures_root_logger() -> None:
    cfg = load_config()
    setup_logging(cfg)
    root = logging.getLogger()
    assert root.level == logging.INFO


def test_setup_logging_adds_handlers() -> None:
    cfg = load_config()
    setup_logging(cfg)
    root = logging.getLogger()
    assert len(root.handlers) >= 1
