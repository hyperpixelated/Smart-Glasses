from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent

DETECTOR_PATH = (
    BASE_DIR
    / "runs/detect/outputs/training/datacluster_yolo11n_v1/weights/best.pt"
)

CLASSIFIER_PATH = (
    BASE_DIR
    / "runs/classify/runs/classify/sign_classifier_v1/weights/best.pt"
)

detector = YOLO(str(DETECTOR_PATH))
classifier = YOLO(str(CLASSIFIER_PATH))

app = FastAPI(title="Smart Glasses")

app.add_middleware(CORSMiddleware, allow_origins=["https://smart-glasses-six.vercel.app"], allow_credentials=False, allow_methods=["POST"], allow_headers=["*"])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        return {"detected": False, "error": "Invalid image"}

    detections = detector.predict(
        source=frame,
        conf=0.35,
        verbose=False,
    )

    best_detection = None

    for box in detections[0].boxes:
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist(),
        )

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        classification = classifier.predict(
            source=crop,
            verbose=False,
        )[0]

        class_id = classification.probs.top1
        class_confidence = float(
            classification.probs.top1conf
        )

        class_name = classifier.names[class_id]

        candidate = {
            "detector_confidence": confidence,
            "classification_confidence": class_confidence,
            "class_name": class_name,
            "box": [x1, y1, x2, y2],
        }

        if (
            best_detection is None
            or class_confidence
            > best_detection["classification_confidence"]
        ):
            best_detection = candidate

    if best_detection is None:
        return {
            "detected": False,
            "message": "No traffic sign detected",
        }

    return {
        "detected": True,
        **best_detection,
    }
