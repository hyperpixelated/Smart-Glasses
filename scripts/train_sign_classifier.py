from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")

model.train(
    data="data/processed/sign_classifier",
    epochs=30,
    imgsz=224,
    batch=32,
    device="mps",
    workers=2,
    project="runs/classify",
    name="sign_classifier_v1",
    patience=8,
    seed=42,
    plots=True,
)

print("\nTraining complete.")
print("Best model:")
print("runs/classify/sign_classifier_v1/weights/best.pt")
