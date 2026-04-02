"""
Configuration loader for the Wildlife Detector pipeline.

Reads config/default.yaml, optionally merges config/local.yaml overrides,
and exposes a validated, frozen configuration object.

Usage:
    from src.config import load_config
    cfg = load_config()                     # default.yaml
    cfg = load_config("config/custom.yaml") # explicit path
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Project root is two levels up from this file (src/config.py → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
LOCAL_OVERRIDE = PROJECT_ROOT / "config" / "local.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive to *base*).

    Args:
        base: Base dictionary (values used when key absent in override).
        override: Override dictionary (takes precedence).

    Returns:
        New dictionary with merged values.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class Config:
    """Lightweight, dot-accessible wrapper around a configuration dict.

    Supports nested attribute access::

        cfg = Config({"detection": {"confidence_threshold": 0.25}})
        cfg.detection.confidence_threshold  # 0.25

    The underlying dict is available via ``cfg.raw``.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    # -- dict-style access ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        """Return value for *key*, or *default* if missing."""
        return self._data.get(key, default)

    # -- dot-style access ----------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(
                f"Config has no attribute '{name}'. "
                f"Available keys: {list(self._data.keys())}"
            ) from None
        if isinstance(value, dict):
            return Config(value)
        return value

    # -- representation ------------------------------------------------------
    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    @property
    def raw(self) -> dict[str, Any]:
        """Return the underlying plain dict."""
        return self._data


def _resolve_paths(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert relative path strings under the ``paths`` section to absolute
    ``Path`` objects anchored at ``PROJECT_ROOT``."""
    paths = cfg_dict.get("paths", {})
    for key, value in paths.items():
        if isinstance(value, str):
            p = Path(value)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            paths[key] = str(p.resolve())
    cfg_dict["paths"] = paths
    return cfg_dict


def load_config(path: str | Path | None = None) -> Config:
    """Load and return the pipeline configuration.

    Resolution order:
      1. ``config/default.yaml``    (always loaded)
      2. ``config/local.yaml``      (merged on top if it exists)
      3. *path* argument            (merged on top if provided)

    Args:
        path: Optional path to an additional YAML override file.

    Returns:
        A :class:`Config` instance with merged settings.

    Raises:
        FileNotFoundError: If *path* is given but does not exist.
    """
    # 1. Load defaults
    if not DEFAULT_CONFIG.exists():
        raise FileNotFoundError(
            f"Default config not found: {DEFAULT_CONFIG}\n"
            "Make sure you're running from the project root."
        )
    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    logger.debug("Loaded default config from %s", DEFAULT_CONFIG)

    # 2. Merge local overrides (optional, git-ignored)
    if LOCAL_OVERRIDE.exists():
        with open(LOCAL_OVERRIDE, "r", encoding="utf-8") as fh:
            local = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, local)
        logger.debug("Merged local overrides from %s", LOCAL_OVERRIDE)

    # 3. Merge explicit override file
    if path is not None:
        override_path = Path(path)
        if not override_path.exists():
            raise FileNotFoundError(f"Config override file not found: {override_path}")
        with open(override_path, "r", encoding="utf-8") as fh:
            extra = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, extra)
        logger.debug("Merged explicit overrides from %s", override_path)

    # Resolve relative paths
    cfg = _resolve_paths(cfg)

    return Config(cfg)
