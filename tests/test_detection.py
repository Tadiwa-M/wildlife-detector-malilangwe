"""Tests for the Detection data class.

Gracefully skips if ultralytics is not installed (CI-light / container).
"""

import pytest

try:
    from src.detection.detector import Detection
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


@pytest.mark.skipif(not HAS_ULTRALYTICS, reason="ultralytics not installed")
def test_detection_repr() -> None:
    det = Detection(
        bbox=(10.0, 20.0, 100.0, 200.0),
        confidence=0.87,
        class_id=1,
        class_name="elephant",
    )
    r = repr(det)
    assert "elephant" in r
    assert "0.87" in r


@pytest.mark.skipif(not HAS_ULTRALYTICS, reason="ultralytics not installed")
def test_detection_to_dict() -> None:
    det = Detection(
        bbox=(1.0, 2.0, 3.0, 4.0),
        confidence=0.5,
        class_id=0,
        class_name="buffalo",
    )
    d = det.to_dict()
    assert d["class_name"] == "buffalo"
    assert d["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert d["confidence"] == 0.5
