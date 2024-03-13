from ultralytics import YOLO

model = YOLO("yolov8x-oiv7.pt")

results = model.predict(source="0", show=True, conf=0.4)

print(results)