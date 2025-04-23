from ultralytics import YOLO

# Load a YOLOv8 model (Nano version)
model = YOLO("yolov8n.pt")  # or yolov8s.pt, yolov8m.pt

# Train the model
model.train(data="data.yaml", epochs=10, imgsz=640)
