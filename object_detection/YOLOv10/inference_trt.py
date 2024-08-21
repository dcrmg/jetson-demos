# coding=gbk
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoprimaryctx
import numpy as np
import cv2
from utils import preprocess, postprocess

trt.init_libnvinfer_plugins(None, "")
 
def get_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
 
TRT_LOGGER = trt.Logger()
engine = get_engine("yolov10n.engine")

#创建上下文
context = engine.create_execution_context()
 
def run_trt(img_in):
    # 查询输入输出name和shape
    for i in range(engine.num_bindings):
        name = engine.get_tensor_name(i)
        size = context.get_tensor_shape(name)
        print(f'name:{name} shape:{size}')
    # 在Host内存中根据大小分配空间
    host_input = cuda.pagelocked_empty(trt.volume(context.get_tensor_shape(engine.get_tensor_name(0))), dtype=np.float32)
    host_output = cuda.pagelocked_empty(trt.volume(context.get_tensor_shape(engine.get_tensor_name(1))), dtype=np.float32)
 
    # 在Device显存中根据Host内存大小分配空间
    device_input = cuda.mem_alloc(host_input.nbytes)
    device_output = cuda.mem_alloc(host_output.nbytes)
    
    # 创建Stream流操作队列
    stream = cuda.Stream()
    # 设置host input 数据
    np.copyto(host_input, img_in.ravel())
    # 将host中数据传输到device中的操作加入stream
    cuda.memcpy_htod_async(device_input, host_input, stream)
    # stream流的异步推理
    context.execute_async_v2(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    # 将device中数据传输到host
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    # Synchronize the stream
    stream.synchronize()
    return host_output 
 
img_source = cv2.imread('image.jpg')
h, w, _ = img_source.shape
print(f'h:{h}, w:{w}')

img = preprocess(img_source)
output = run_trt(img)
output = output.reshape(300, 6)
img = postprocess(img_source, output)

cv2.imwrite('result_trt.jpg', img)
print('done')


