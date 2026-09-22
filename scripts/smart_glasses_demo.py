import cv2
from ultralytics import YOLO

DETECTOR_PATH = "runs/detect/outputs/training/datacluster_yolo11n_v1/weights/best.pt"
CLASSIFIER_PATH = "runs/classify/runs/classify/sign_classifier_v1/weights/best.pt"

detector = YOLO(DETECTOR_PATH)
classifier = YOLO(CLASSIFIER_PATH)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

print("Smart Glasses detection + classification started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame.")
        break

    results = detector.predict(
        source=frame,
        conf=0.35,
        verbose=False
    )

    annotated = frame.copy()

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        classification = classifier.predict(
            source=crop,
            verbose=False
        )

        result = classification[0]

        class_id = result.probs.top1
        confidence = float(result.probs.top1conf)
        class_name = classifier.names[class_id]

        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Smart Glasses", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
