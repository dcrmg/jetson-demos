# coding=gbk
import tensorrt as trt
import pycuda.driver as cuda
#import pycuda.autoinit
import numpy as np
import cv2

import pycuda.autoprimaryctx

def get_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def process_post(img, source_h=360, source_w=640):
    output_uint = (img.reshape(256, 448, 1) * 255).astype(np.uint8)
    matte = cv2.resize(output_uint, (source_w, source_h), interpolation=cv2.INTER_AREA)
    return matte

# ����gstreamer�ܵ�����
def gstreamer_pipeline(
    capture_width=1280//2, #����ͷԤ�����ͼ����
    capture_height=720//2, #����ͷԤ�����ͼ��߶�
    display_width=1280//2, #������ʾ��ͼ����
    display_height=720//2, #������ʾ��ͼ��߶�
    framerate=60//2,       #����֡��
    flip_method=0,      #�Ƿ���תͼ��
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def run_trt():
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
            
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER) 
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # ��֡��ʾ
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img11 = cap.read()
            img1 = process_pre(img11)
                      
            # ����host input ����
            np.copyto(host_input, img1.ravel())
            # ��host�����ݴ��䵽device
            cuda.memcpy_htod_async(device_input, host_input, stream)
            # stream�����첽����
            context.execute_async_v2(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
            # ��device�����ݴ��䵽host
            cuda.memcpy_dtoh_async(host_output, device_output, stream)
            # Synchronize the stream
            #stream.synchronize()
            
            output = host_output
            matte = process_post(output)
            img_result = (img11 * (matte[..., np.newaxis] / 255.)).astype(np.uint8)
            cv2.imshow("CSI Camera", img_result)
 
            keyCode = cv2.waitKey(30) & 0xFF
            if keyCode == 27:# ESC���˳�
                break
 
        cap.release()
        cv2.destroyAllWindows()

def process_pre(img_source, inference_h=256, inference_w=448):
    img = cv2.resize(img_source, (inference_w, inference_h), cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img = (img - 127.5) / 127.5
    img = np.expand_dims(img, axis=0)
    return img




if __name__ == "__main__":
    #sudo service nvargus-daemon restart

    trt.init_libnvinfer_plugins(None, "")
    TRT_LOGGER = trt.Logger()

    engine = get_engine("modnet_448_256.engine")

    #����������
    context = engine.create_execution_context()
    run_trt()
