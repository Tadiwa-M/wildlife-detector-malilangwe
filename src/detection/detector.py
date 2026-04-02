"""
Detection module for the Wildlife Detector pipeline.

Provides a ``Detector`` class that wraps an Ultralytics YOLOv11 model
with config-driven defaults for confidence, IoU, device, and image size.

Usage:
    from src.config import load_config
    from src.detection.detector import Detector

    cfg = load_config()
    det = Detector(cfg)
    results = det.predict("path/to/image.jpg")
    detections = det.parse_results(results)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from src.config import Config

logger = logging.getLogger(__name__)


class Detection:
    """A single bounding-box detection.

    Attributes:
        bbox:       (x1, y1, x2, y2) pixel coordinates.
        confidence: Detection confidence ∈ [0, 1].
        class_id:   Integer class index.
        class_name: Human-readable class label.
    """

    __slots__ = ("bbox", "confidence", "class_id", "class_name")

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        confidence: float,
        class_id: int,
        class_name: str,
    ) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    def __repr__(self) -> str:
        x1, y1, x2, y2 = (round(c, 1) for c in self.bbox)
        return (
            f"Detection({self.class_name}, conf={self.confidence:.2f}, "
            f"bbox=[{x1}, {y1}, {x2}, {y2}])"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary (JSON-friendly)."""
        return {
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


class Detector:
    """Config-driven YOLOv11 wildlife detector.

    Args:
        cfg: Pipeline configuration object (from ``load_config``).
        model_path: Override the model weight path from config.
    """

    def __init__(self, cfg: Config, model_path: str | Path | None = None) -> None:
        self._cfg = cfg
        det_cfg = cfg.detection

        # Resolve model path
        self._model_path = Path(
            model_path or cfg.paths.default_model
        )
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self._model_path}\n"
                "Download a pretrained checkpoint or update paths.default_model in config."
            )

        # Store inference parameters from config
        self._conf = float(det_cfg.confidence_threshold)
        self._iou = float(det_cfg.iou_threshold)
        self._imgsz = int(det_cfg.image_size)
        self._device = str(det_cfg.device)
        self._max_det = int(det_cfg.max_detections)
        self._half = bool(det_cfg.half_precision)
        self._augment = bool(det_cfg.augment)

        # Class-name mapping (WAID classes for fine-tuned model, COCO for pretrained)
        self._class_names: dict[int, str] = {}

        # Load model
        logger.info("Loading model from %s on device=%s", self._model_path, self._device)
        self._model = YOLO(str(self._model_path))

        # Populate class names from the loaded model
        if hasattr(self._model, "names"):
            self._class_names = dict(self._model.names)
            logger.info("Model classes (%d): %s", len(self._class_names), self._class_names)

    # -- public properties ---------------------------------------------------

    @property
    def class_names(self) -> dict[int, str]:
        """Mapping of class index → class name from the loaded model."""
        return self._class_names

    @property
    def model(self) -> YOLO:
        """Underlying Ultralytics YOLO model (advanced use)."""
        return self._model

    # -- inference -----------------------------------------------------------

    def predict(
        self,
        source: str | Path | np.ndarray,
        *,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        save: bool = False,
        save_dir: str | Path | None = None,
    ) -> list:
        """Run detection on an image, directory, or video frame.

        Args:
            source:   Path to image/video/directory, or a numpy BGR array.
            conf:     Override confidence threshold.
            iou:      Override IoU threshold.
            imgsz:    Override input image size.
            save:     Whether to save annotated results to disk.
            save_dir: Directory for saved results (defaults to config output_dir).

        Returns:
            List of ``ultralytics.engine.results.Results`` objects.
        """
        kwargs: dict[str, Any] = {
            "source": source if not isinstance(source, Path) else str(source),
            "conf": conf if conf is not None else self._conf,
            "iou": iou if iou is not None else self._iou,
            "imgsz": imgsz if imgsz is not None else self._imgsz,
            "device": self._device,
            "max_det": self._max_det,
            "half": self._half,
            "augment": self._augment,
            "verbose": False,
        }

        if save:
            kwargs["save"] = True
            kwargs["project"] = str(save_dir or self._cfg.paths.output_dir)

        results = self._model.predict(**kwargs)
        logger.debug("Inference complete — %d result frame(s)", len(results))
        return results

    # -- result parsing ------------------------------------------------------

    @staticmethod
    def parse_results(results: list) -> list[list[Detection]]:
        """Convert raw Ultralytics results into structured ``Detection`` lists.

        Args:
            results: Output from :meth:`predict`.

        Returns:
            A list (one per input frame) of ``Detection`` lists.
        """
        all_detections: list[list[Detection]] = []

        for result in results:
            frame_dets: list[Detection] = []
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                all_detections.append(frame_dets)
                continue

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names.get(cls_id, f"class_{cls_id}")

                frame_dets.append(
                    Detection(
                        bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )

            all_detections.append(frame_dets)

        return all_detections

    # -- convenience ---------------------------------------------------------

    def detect(
        self, source: str | Path | np.ndarray, **kwargs: Any
    ) -> list[list[Detection]]:
        """Shorthand: predict + parse in one call.

        Returns:
            Parsed detection lists (one list per input frame).
        """
        results = self.predict(source, **kwargs)
        return self.parse_results(results)
