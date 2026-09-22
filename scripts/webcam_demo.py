#!/usr/bin/env python3
"""Run pretrained YOLO11n detection on the default macOS webcam."""

from __future__ import annotations

import sys
import time
from argparse import ArgumentParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n.pt"
WINDOW_NAME = "Smart Glasses YOLO11n Webcam Demo"


def open_camera(cv2, camera_index: int):
    """Open one explicitly selected camera and validate its first frame."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Unable to open camera {camera_index}. Check macOS camera permissions, "
            "ensure no other app is using it, and confirm a camera is connected."
        )

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError(
            f"Unable to read the first frame from camera {camera_index} "
            f"(ret={ret}, frame_is_none={frame is None})."
        )
    return cap, frame


def detect_and_annotate(model, frame):
    """Run one YOLO inference and return Ultralytics' annotated frame."""
    result = model(frame, verbose=False)[0]
    return result.plot()


def main() -> int:
    """Manage camera lifecycle and run the live inference loop until Q is pressed."""
    parser = ArgumentParser(description="Run YOLO11n detection on one OpenCV camera index.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index to use (default: 0)")
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        print(
            f"ERROR: YOLO11n weights were not found at {MODEL_PATH}. "
            "Run 'python scripts/yolo_smoke_test.py' first.",
            file=sys.stderr,
        )
        return 1

    try:
        import cv2
    except Exception as error:
        print(f"ERROR: Unable to import OpenCV: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    cap = None
    try:
        # Open and validate only the explicitly requested camera before inference.
        cap, frame = open_camera(cv2, args.camera)

        try:
            from ultralytics import YOLO

            model = YOLO(str(MODEL_PATH))
        except Exception as error:
            print(f"ERROR: Unable to load YOLO11n weights: {type(error).__name__}: {error}", file=sys.stderr)
            return 1

        print("Webcam demo started. Press Q in the video window to quit.")

        previous_time = time.perf_counter()
        while True:
            # Run inference and draw YOLO's boxes, labels, and scores.
            annotated_frame = detect_and_annotate(model, frame)

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            previous_time = current_time
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, annotated_frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError(
                    f"Unable to read a frame from camera {args.camera} "
                    f"(ret={ret}, frame_is_none={frame is None})."
                )
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"ERROR: Webcam inference failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        # Always return the camera to macOS and close the OpenCV display window.
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    print("Webcam demo stopped cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
