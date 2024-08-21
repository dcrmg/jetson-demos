import cv2
import numpy as np
import random

yolo_inference_h, yolo_inference_w = 640, 640

def preprocess(img):
    img = cv2.resize(img, (yolo_inference_w, yolo_inference_h), cv2.INTER_LINEAR)
    img = np.transpose(img, (2, 0, 1))
    img = img/255
    return np.expand_dims(img, axis=0).astype(np.float32)

def postprocess(img ,output):
    h, w, _ = img.shape
    for i in range(output.shape[0]):
        confidence = output[i][4]
        if confidence > 0.3:
            label = int(output[i][5])
            xmin = int(output[i][0] * w / yolo_inference_w)
            ymin = int(output[i][1] * h / yolo_inference_h)
            xmax = int(output[i][2] * w / yolo_inference_w)
            ymax = int(output[i][3] * h / yolo_inference_h)
            score = '%.2f'%output[i][4]
            
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), 255, 4)
            img = cv2.putText(img, f'{COCO_Classes[label]} {score}', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0, 0, 255), thickness=3)
    return img

COCO_Classes = {0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}