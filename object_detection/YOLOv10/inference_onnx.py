import cv2
import onnxruntime
from utils import preprocess, postprocess

session = onnxruntime.InferenceSession('yolov10n.onnx', None)

model_info_input = session.get_inputs()
for item in model_info_input:
    print(f"name:{item.name}, shape:{item.shape}, type:{item.type}")
model_info_output = session.get_outputs()
for item in model_info_output:
    print(f"name:{item.name}, shape:{item.shape}, type:{item.type}")

input_name = model_info_input[0].name
output_name = model_info_output[0].name

img_source = cv2.imread('image.jpg')
h, w, _ = img_source.shape
print(f'h:{h}, w:{w}')

img = preprocess(img_source)
result = session.run([output_name], {input_name: img})[0][0]
img = postprocess(img_source, result)

cv2.imwrite('result_onnx.jpg', img)
print('done')

