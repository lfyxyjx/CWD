from ultralytics import YOLO
model = YOLO("ultralytics/models/v8/yolov8s-wcd.yaml")
results = model.train(data='ultralytics/models/v8/my.yaml',epochs=300, device='0', batch=8, imgsz=640)