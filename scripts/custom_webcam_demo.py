import cv2
from ultralytics import YOLO

MODEL_PATH = "runs/detect/outputs/training/datacluster_yolo11n_v1/weights/best.pt"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

print("Custom traffic-sign detector started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame.")
        break

    results = model.predict(
        source=frame,
        conf=0.35,
        verbose=False,
    )

    annotated = results[0].plot()

    cv2.imshow("Smart Glasses - Traffic Sign Detector", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
