"""
Multi-object tracking module using BoT-SORT via Ultralytics.

Provides a ``Tracker`` class that runs YOLOv11 detection + BoT-SORT tracking
on video files or frame sequences.

Usage:
    from src.tracking.tracker import Tracker
    tracker = Tracker(cfg)
    for frame, detections, track_ids in tracker.track_video("input.mp4"):
        ...  # visualize or log

Note:
    This module will be fully implemented in Roadmap Step 5.
    The current version provides the interface and basic plumbing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

from src.config import Config
from src.detection.detector import Detection, Detector

logger = logging.getLogger(__name__)


class Tracker:
    """Config-driven BoT-SORT wildlife tracker.

    Wraps Ultralytics' built-in tracking mode, which couples
    YOLOv11 detection with BoT-SORT association frame-by-frame.

    Args:
        cfg: Pipeline configuration object.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._detector = Detector(cfg)

        trk_cfg = cfg.tracking
        self._tracker_type = str(trk_cfg.tracker)
        self._tracker_config = str(trk_cfg.tracker_config)
        self._track_buffer = int(trk_cfg.track_buffer)
        self._min_track_length = int(trk_cfg.min_track_length)

        logger.info(
            "Tracker initialised: %s (buffer=%d, min_length=%d)",
            self._tracker_type,
            self._track_buffer,
            self._min_track_length,
        )

    def track_video(
        self,
        video_path: str | Path,
        *,
        save: bool = False,
    ) -> Generator[tuple[np.ndarray, list[Detection], list[int]], None, None]:
        """Run detection + tracking on a video file, yielding per-frame results.

        Args:
            video_path: Path to the input video.
            save:       Whether to save the annotated video.

        Yields:
            Tuple of (frame, detections, track_ids) for each frame.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        det_cfg = self._cfg.detection

        results_gen = self._detector.model.track(
            source=str(video_path),
            conf=float(det_cfg.confidence_threshold),
            iou=float(det_cfg.iou_threshold),
            imgsz=int(det_cfg.image_size),
            device=str(det_cfg.device),
            tracker=self._tracker_config,
            stream=True,
            verbose=False,
        )

        for result in results_gen:
            frame = result.orig_img
            detections: list[Detection] = []
            track_ids: list[int] = []

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = result.names.get(cls_id, f"class_{cls_id}")
                    tid = int(box.id[0].cpu().numpy()) if box.id is not None else -1

                    detections.append(
                        Detection(
                            bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name,
                        )
                    )
                    track_ids.append(tid)

            yield frame, detections, track_ids

        logger.info("Tracking complete for %s", video_path.name)
