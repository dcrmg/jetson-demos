from ultralytics import YOLO

# Load a YOLOv10n PyTorch model
print('Load the yolo model')
model = YOLO("yolov10n.pt")

print('exporting onnx')
model.export(format='onnx')
print('done')