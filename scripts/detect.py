"""
Run config-driven wildlife detection on images or directories.

Usage:
    python scripts/detect.py --source path/to/image.jpg
    python scripts/detect.py --source path/to/folder/ --save
    python scripts/detect.py --source img.jpg --conf 0.4 --config config/custom.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.detection.detector import Detector
from src.utils.logging_setup import setup_logging
from src.utils.visualization import draw_detections, draw_summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Malilangwe Wildlife Detector — config-driven inference"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to image, directory, or video file.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config override file.",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Override confidence threshold.",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated results to the output directory.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display results in a window (requires GUI).",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for the detection script."""
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)

    detector = Detector(cfg)
    results = detector.predict(args.source, conf=args.conf, save=args.save)
    parsed = detector.parse_results(results)

    for i, (result, detections) in enumerate(zip(results, parsed)):
        print(f"\n--- Frame {i} ---")
        print(f"  Detections: {len(detections)}")
        for det in detections:
            print(f"    {det}")

        if args.show:
            frame = result.orig_img
            annotated = draw_detections(frame, detections, cfg)
            annotated = draw_summary(annotated, detections)
            cv2.imshow("Wildlife Detector", annotated)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()

    total = sum(len(d) for d in parsed)
    print(f"\nTotal detections across {len(parsed)} frame(s): {total}")


if __name__ == "__main__":
    main()
