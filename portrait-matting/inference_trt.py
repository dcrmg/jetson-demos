# coding=gbk
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoprimaryctx
import numpy as np
import cv2

trt.init_libnvinfer_plugins(None, "")
 
def get_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
 
TRT_LOGGER = trt.Logger()

engine = get_engine("modnet_448_256.engine")

#����������
context = engine.create_execution_context()
 
def run_trt(img_in):
    # ��ѯ�������name��shape
    for i in range(engine.num_bindings):
        name = engine.get_tensor_name(i)
        size = context.get_tensor_shape(name)
        print(f'name:{name} shape:{size}')
    # ��Host�ڴ��и��ݴ�С����ռ�
    host_input = cuda.pagelocked_empty(trt.volume(context.get_tensor_shape('input')), dtype=np.float32)
    host_output = cuda.pagelocked_empty(trt.volume(context.get_tensor_shape('output')), dtype=np.float32)
 
    # ��Device�Դ��и���Host�ڴ��С����ռ�
    device_input = cuda.mem_alloc(host_input.nbytes)
    device_output = cuda.mem_alloc(host_output.nbytes)
    
    # ����Stream����������
    stream = cuda.Stream()
    # ����host input ����
    np.copyto(host_input, img_in.ravel())
    # ��host�����ݴ��䵽device�еĲ�������stream
    cuda.memcpy_htod_async(device_input, host_input, stream)
    # stream�����첽����
    context.execute_async_v2(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    # ��device�����ݴ��䵽host
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    # Synchronize the stream
    stream.synchronize()
    return host_output 
 
img_source = cv2.imread('image.jpg')
h, w, _ = img_source.shape
img = cv2.resize(img_source, (448, 256), cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose((2, 0, 1)).astype(np.float32)
img = (img - 127.5) / 127.5
img = np.expand_dims(img, axis=0)
output = run_trt(img)

output_uint = (output.reshape(256, 448, 1) * 255).astype(np.uint8)
matte = cv2.resize(output_uint, (w, h), interpolation=cv2.INTER_AREA)
img_result = img_source * (matte[..., np.newaxis] / 255.)
cv2.imwrite('result_trt.jpg', img_result.astype(np.uint8))
print('done')
