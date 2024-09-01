import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoprimaryctx
import numpy as np
import cv2
from utils import preprocess, postprocess

def get_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def gstreamer_pipeline(
    capture_width=640,
    capture_height=640,
    display_width=640,
    display_height=640,
    framerate=60//2,
    flip_method=0,
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

def run_trt(engine):
    context = engine.create_execution_context()
    for i in range(engine.num_bindings):
        name = engine.get_tensor_name(i)
        size = context.get_tensor_shape(name)
        print(f'name:{name} shape:{size}')
    host_input = cuda.pagelocked_empty(trt.volume(context.get_tensor_shape(engine.get_tensor_name(0))), dtype=np.float32)
    host_output = cuda.pagelocked_empty(trt.volume(context.get_tensor_shape(engine.get_tensor_name(1))), dtype=np.float32)
 
    device_input = cuda.mem_alloc(host_input.nbytes)
    device_output = cuda.mem_alloc(host_output.nbytes)
    
    stream = cuda.Stream()
            
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER) 
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img11 = cap.read()
            img = preprocess(img11)
            np.copyto(host_input, img.ravel())
            cuda.memcpy_htod_async(device_input, host_input, stream)
            context.execute_async_v2(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(host_output, device_output, stream)
            # Synchronize the stream
            stream.synchronize()
            output = host_output.reshape(300, 6)
            img = postprocess(img11, output)
            cv2.imshow("CSI Camera", img)
            print(img.shape)
 
            keyCode = cv2.waitKey(30) & 0xFF
            if keyCode == 27:
                break
 
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    #sudo service nvargus-daemon restart
    trt.init_libnvinfer_plugins(None, "")
    TRT_LOGGER = trt.Logger()

    engine = get_engine("yolov10n.engine")

    run_trt(engine)
