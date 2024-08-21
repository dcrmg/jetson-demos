from ultralytics import YOLO
import cv2
import matplotlib.pyplot as ptl

# Load a YOLOv10n PyTorch model
model = YOLO("yolov10n.pt")

print('Run inference')
results = model('image.jpg')
annotated_img = results[0].plot()

cv2.imwrite('result_ultralytics.jpg', annotated_img)
print('done')
