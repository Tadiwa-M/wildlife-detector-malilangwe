"""
Visualization utilities for the Wildlife Detector pipeline.

Draws bounding boxes, class labels, confidence scores, and track IDs
onto images or video frames using config-driven styling.

Usage:
    from src.utils.visualization import draw_detections
    annotated = draw_detections(frame, detections, cfg)
"""

from __future__ import annotations

import cv2
import numpy as np

from src.config import Config
from src.detection.detector import Detection

# Fallback colour if a class is not in the palette
_DEFAULT_COLOR = (0, 255, 0)  # green (BGR)


def _get_class_color(class_name: str, cfg: Config) -> tuple[int, int, int]:
    """Look up the BGR colour for a class from the config palette."""
    palette: dict = cfg.visualization.raw.get("class_colors", {})
    color = palette.get(class_name)
    if color is None:
        return _DEFAULT_COLOR
    return tuple(color)  # type: ignore[return-value]


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    cfg: Config,
    *,
    track_ids: list[int | None] | None = None,
) -> np.ndarray:
    """Draw bounding boxes and labels onto an image.

    Args:
        image:      BGR numpy array (will be copied, original unchanged).
        detections: List of ``Detection`` objects for this frame.
        cfg:        Pipeline config (controls line thickness, font, colours).
        track_ids:  Optional per-detection track IDs (from BoT-SORT).

    Returns:
        Annotated copy of the image.
    """
    vis_cfg = cfg.visualization
    thickness = int(vis_cfg.line_thickness)
    font_scale = float(vis_cfg.font_scale)
    show_conf = bool(vis_cfg.show_confidence)
    show_tid = bool(vis_cfg.show_track_id)

    annotated = image.copy()

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = (int(c) for c in det.bbox)
        color = _get_class_color(det.class_name, cfg)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Build label string
        parts: list[str] = [det.class_name]
        if show_conf:
            parts.append(f"{det.confidence:.0%}")
        if show_tid and track_ids is not None and idx < len(track_ids):
            tid = track_ids[idx]
            if tid is not None:
                parts.append(f"ID:{tid}")
        label = " ".join(parts)

        # Label background
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(
            annotated,
            (x1, y1 - th - baseline - 4),
            (x1 + tw, y1),
            color,
            cv2.FILLED,
        )
        # Label text (black on colour background for readability)
        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_summary(
    image: np.ndarray,
    detections: list[Detection],
    *,
    position: tuple[int, int] = (10, 30),
) -> np.ndarray:
    """Overlay a species-count summary in the top-left corner.

    Args:
        image:      BGR numpy array (will be copied).
        detections: Detections for the current frame.
        position:   (x, y) for the first line of text.

    Returns:
        Annotated copy of the image.
    """
    annotated = image.copy()
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1

    x, y = position
    for species, count in sorted(counts.items()):
        text = f"{species}: {count}"
        cv2.putText(
            annotated, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        y += 25

    return annotated
