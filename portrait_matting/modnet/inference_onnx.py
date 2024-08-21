"""run_onnx.py

A simple script for verifying the modnet.onnx model.
"""

import numpy as np
import cv2
import onnxruntime

session = onnxruntime.InferenceSession('modnet_448_256.onnx', None)
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
img = cv2.resize(img_source, (448, 256), cv2.INTER_LINEAR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose((2, 0, 1)).astype(np.float32)
img = (img - 127.5) / 127.5
img = np.expand_dims(img, axis=0)

result = session.run([output_name], {input_name: img})
matte = (np.squeeze(result[0]) * 255).astype(np.uint8)
matte = cv2.resize(matte, dsize=(w, h), interpolation = cv2.INTER_LINEAR)
img_result = img_source * (matte[..., np.newaxis] / 255.)
cv2.imwrite('result_onnx.jpg', img_result.astype(np.uint8))
print('done')
