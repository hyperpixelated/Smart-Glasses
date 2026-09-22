"""Run one pretrained YOLO11n inference to verify the ML runtime."""

from __future__ import annotations

import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "smoke_test"
OUTPUT_IMAGE = OUTPUT_DIR / "annotated_bus.jpg"


def main() -> int:
    """Load YOLO11n, infer on a packaged sample image, and save its annotation."""
    try:
        # Keep setup notices from third-party imports out of the smoke-test report.
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            import ultralytics
            from ultralytics import YOLO
    except Exception as error:
        print(f"ERROR: Unable to import Ultralytics: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    image_path = Path(ultralytics.__file__).resolve().parent / "assets" / "bus.jpg"
    if not image_path.is_file():
        print(f"ERROR: Bundled Ultralytics test image was not found: {image_path}", file=sys.stderr)
        return 1

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Ultralytics downloads these official pretrained weights only when absent.
        model = YOLO(str(MODEL_PATH))
    except Exception as error:
        print(f"ERROR: Unable to load pretrained YOLO11n weights: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    try:
        result = model(str(image_path), verbose=False)[0]
    except Exception as error:
        print(f"ERROR: YOLO11n inference failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    try:
        result.save(filename=str(OUTPUT_IMAGE))
    except Exception as error:
        print(f"ERROR: Unable to save annotated image: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    if not OUTPUT_IMAGE.is_file():
        print(f"ERROR: Annotated image was not created: {OUTPUT_IMAGE}", file=sys.stderr)
        return 1

    print("YOLO11n smoke test succeeded")
    print(f"Test image: {image_path}")
    print(f"Annotated image: {OUTPUT_IMAGE}")
    print("Detections:")
    for index, box in enumerate(result.boxes, start=1):
        class_id = int(box.cls[0].item())
        class_name = result.names[class_id]
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        print(
            f"  {index}. class_id={class_id}, class_name={class_name}, "
            f"confidence={confidence:.4f}, bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
