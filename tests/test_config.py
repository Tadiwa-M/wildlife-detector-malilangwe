"""Tests for the configuration system."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import Config, _deep_merge, load_config


class TestDeepMerge:
    """Unit tests for the _deep_merge helper."""

    def test_flat_override(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        assert _deep_merge(base, override) == {"a": 1, "b": 99}

    def test_nested_override(self) -> None:
        base = {"x": {"y": 1, "z": 2}}
        override = {"x": {"z": 42}}
        result = _deep_merge(base, override)
        assert result == {"x": {"y": 1, "z": 42}}

    def test_add_new_key(self) -> None:
        base = {"a": 1}
        override = {"b": 2}
        assert _deep_merge(base, override) == {"a": 1, "b": 2}

    def test_original_unchanged(self) -> None:
        base = {"a": {"b": 1}}
        override = {"a": {"b": 99}}
        _deep_merge(base, override)
        assert base["a"]["b"] == 1  # base not mutated


class TestConfig:
    """Tests for the Config wrapper class."""

    def test_dot_access(self) -> None:
        cfg = Config({"detection": {"confidence_threshold": 0.25}})
        assert cfg.detection.confidence_threshold == 0.25

    def test_dict_access(self) -> None:
        cfg = Config({"a": 1})
        assert cfg["a"] == 1

    def test_get_with_default(self) -> None:
        cfg = Config({"a": 1})
        assert cfg.get("missing", 42) == 42

    def test_contains(self) -> None:
        cfg = Config({"a": 1})
        assert "a" in cfg
        assert "z" not in cfg

    def test_missing_key_raises(self) -> None:
        cfg = Config({"a": 1})
        with pytest.raises(AttributeError, match="no attribute 'missing'"):
            _ = cfg.missing


class TestLoadConfig:
    """Integration tests for load_config (requires default.yaml on disk)."""

    def test_loads_default(self) -> None:
        cfg = load_config()
        assert cfg.project.name == "Malilangwe Wildlife Detector"
        assert cfg.detection.confidence_threshold == 0.25

    def test_override_merges(self) -> None:
        override = {"detection": {"confidence_threshold": 0.9}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            yaml.dump(override, tmp)
            tmp_path = tmp.name

        cfg = load_config(tmp_path)
        assert cfg.detection.confidence_threshold == 0.9
        # Other defaults still intact
        assert cfg.detection.iou_threshold == 0.45

    def test_missing_override_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/override.yaml")
