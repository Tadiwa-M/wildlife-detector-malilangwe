"""Tests for the dataset validation and YAML generation module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import load_config
from src.data.dataset import generate_dataset_yaml, get_class_distribution, validate_dataset


class TestValidateDataset:
    """Tests for validate_dataset."""

    def test_returns_stats_dict(self) -> None:
        cfg = load_config()
        stats = validate_dataset(cfg)
        assert "splits" in stats
        assert "total_labels" in stats

    def test_label_counts_nonzero(self) -> None:
        cfg = load_config()
        stats = validate_dataset(cfg)
        # Labels are included in the repo even if images aren't
        assert stats["total_labels"] > 0

    def test_all_splits_present(self) -> None:
        cfg = load_config()
        stats = validate_dataset(cfg)
        for split in ("train", "val", "test"):
            assert split in stats["splits"]


class TestGenerateDatasetYaml:
    """Tests for generate_dataset_yaml."""

    def test_generates_valid_yaml(self) -> None:
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_waid.yaml"
            result = generate_dataset_yaml(cfg, output_path=out_path)
            assert result.exists()

            with open(result) as f:
                data = yaml.safe_load(f)

            assert data["nc"] == 6
            assert len(data["names"]) == 6
            assert "train" in data
            assert "val" in data
            assert "test" in data


class TestClassDistribution:
    """Tests for get_class_distribution."""

    def test_returns_all_classes(self) -> None:
        cfg = load_config()
        dist = get_class_distribution(cfg, split="train")
        class_names = list(cfg.dataset.class_names)
        for name in class_names:
            assert name in dist

    def test_counts_are_nonnegative(self) -> None:
        cfg = load_config()
        dist = get_class_distribution(cfg, split="train")
        for count in dist.values():
            assert count >= 0
