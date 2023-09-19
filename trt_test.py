
import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from utils.tool import *

CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4


class FastestDet_TRT(object):
    """
    description: A FastestDet class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path, batch_size):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        self.batch_size = batch_size

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))

            # Multiply by since we use dynamic batch size in our trt plan file
            size = trt.volume(engine.get_binding_shape(binding)) * self.batch_size * -1

            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def infer(self, img_path):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess

        img, ori_img = self.preprocess_image(input_img_path=img_path)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], img.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)

        # set again since use dynamic batch size
        context.set_binding_shape(0, (self.batch_size, 3, self.input_h, self.input_w))

        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)

        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]

        output = self.post_process(output)

        H, W, _ = ori_img.shape
        scale_h, scale_w = H / 352, W / 352

        for box in output[0]:
            box = box.tolist()
            obj_score = box[4]
            x1, y1 = int(box[0] * W), int(box[1] * H)
            x2, y2 = int(box[2] * W), int(box[3] * H)
            cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)

        return ori_img, end - start

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        
    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)
        
    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, input_img_path):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            img:  the processed image
            ori_img: the original image
        """
        ori_img = cv2.imread(input_img_path)
        res_img = cv2.resize(ori_img, (352, 352), interpolation = cv2.INTER_LINEAR) 
        img = res_img.reshape(1, 352, 352, 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.float() / 255.0

        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(img)
        return image, ori_img

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output):
        pred = np.reshape(output[0:], (1, 85, 22, 22))
        device = torch.device("cpu")
        output = handle_preds_trt(pred, device, conf_thresh=0.25, nms_thresh=0.45)

        return output


class inferThread(threading.Thread):
    def __init__(self, trt_wrapper, image_path, output_path):
        threading.Thread.__init__(self)
        self.trt_wrapper = trt_wrapper
        self.image_path = image_path
        self.output_path = output_path

    def run(self):
        image_raw, use_time = self.trt_wrapper.infer(self.image_path)
        cv2.imwrite(self.output_path, image_raw)
        print("result save to {}, time usage: {}".format(self.output_path, use_time))



if __name__ == "__main__":

    engine_file_path = "FastestDet.engine"
    input_image_path = "data/3.jpg"
    output_image_path = "data/result.jpg"

    trt_wrapper = FastestDet_TRT(engine_file_path, batch_size=1)
    try:
        print('batch size is', trt_wrapper.batch_size)

        thread1 = inferThread(trt_wrapper, input_image_path, output_image_path)
        thread1.start()
        thread1.join()
    finally:
        # destroy the instance
        trt_wrapper.destroy()
