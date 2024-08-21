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

# 设置gstreamer管道参数
def gstreamer_pipeline(
    capture_width=1280//2, #摄像头预捕获的图像宽度
    capture_height=720//2, #摄像头预捕获的图像高度
    display_width=1280//2, #窗口显示的图像宽度
    display_height=720//2, #窗口显示的图像高度
    framerate=60//2,       #捕获帧率
    flip_method=0,      #是否旋转图像
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
    # 查询输入输出name和shape
    for i in range(engine.num_bindings):
        name = engine.get_tensor_name(i)
        size = context.get_tensor_shape(name)
        print(f'name:{name} shape:{size}')
    
    # 在Host内存中根据大小分配空间
    host_input = cuda.pagelocked_empty(trt.volume(context.get_tensor_shape('input')), dtype=np.float32)
    host_output = cuda.pagelocked_empty(trt.volume(context.get_tensor_shape('output')), dtype=np.float32)
 
    # 在Device显存中根据Host内存大小分配空间
    device_input = cuda.mem_alloc(host_input.nbytes)
    device_output = cuda.mem_alloc(host_output.nbytes)
    
    # 创建Stream流操作队列
    stream = cuda.Stream()
            
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER) 
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # 逐帧显示
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img11 = cap.read()
            img1 = process_pre(img11)
                      
            # 设置host input 数据
            np.copyto(host_input, img1.ravel())
            # 将host中数据传输到device
            cuda.memcpy_htod_async(device_input, host_input, stream)
            # stream流的异步推理
            context.execute_async_v2(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
            # 将device中数据传输到host
            cuda.memcpy_dtoh_async(host_output, device_output, stream)
            # Synchronize the stream
            #stream.synchronize()
            
            output = host_output
            matte = process_post(output)
            img_result = (img11 * (matte[..., np.newaxis] / 255.)).astype(np.uint8)
            cv2.imshow("CSI Camera", img_result)
 
            keyCode = cv2.waitKey(30) & 0xFF
            if keyCode == 27:# ESC键退出
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

    #创建上下文
    context = engine.create_execution_context()
    run_trt()
