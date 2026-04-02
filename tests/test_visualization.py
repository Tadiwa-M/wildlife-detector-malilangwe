"""Tests for visualization utilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config, load_config

try:
    from src.detection.detector import Detection
    from src.utils.visualization import draw_detections, draw_summary

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@pytest.mark.skipif(not HAS_DEPS, reason="visualization deps not installed")
class TestDrawDetections:
    """Tests for draw_detections."""

    def _make_frame(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def _make_detection(self) -> Detection:
        return Detection(
            bbox=(100.0, 100.0, 200.0, 200.0),
            confidence=0.85,
            class_id=0,
            class_name="sheep",
        )

    def test_returns_copy(self) -> None:
        cfg = load_config()
        frame = self._make_frame()
        det = self._make_detection()
        result = draw_detections(frame, [det], cfg)
        # Should be a different array (copy)
        assert result is not frame

    def test_output_shape_matches_input(self) -> None:
        cfg = load_config()
        frame = self._make_frame()
        det = self._make_detection()
        result = draw_detections(frame, [det], cfg)
        assert result.shape == frame.shape

    def test_empty_detections(self) -> None:
        cfg = load_config()
        frame = self._make_frame()
        result = draw_detections(frame, [], cfg)
        # With no detections, output should be identical to input
        assert np.array_equal(result, frame)

    def test_with_track_ids(self) -> None:
        cfg = load_config()
        frame = self._make_frame()
        det = self._make_detection()
        result = draw_detections(frame, [det], cfg, track_ids=[42])
        assert result.shape == frame.shape


@pytest.mark.skipif(not HAS_DEPS, reason="visualization deps not installed")
class TestDrawSummary:
    """Tests for draw_summary."""

    def test_returns_copy(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = Detection(
            bbox=(10.0, 10.0, 50.0, 50.0),
            confidence=0.9,
            class_id=1,
            class_name="cattle",
        )
        result = draw_summary(frame, [det])
        assert result is not frame

    def test_empty_detections(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_summary(frame, [])
        assert np.array_equal(result, frame)
