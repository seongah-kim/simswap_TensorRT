import os
from sys import breakpointhook

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import logging
from torch.cuda import nvtx
import contextlib

@contextlib.contextmanager
def nvtx_range(msg):
	depth = torch.cuda.nvtx.range_push(msg)
	try:
		yield depth
	finally:
		torch.cuda.nvtx.range_pop()

TRT_LOGGER = trt.Logger()
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    
    def __repr__(self):
        return self.__str__()

def get_engine(trt_file_path):
    if os.path.exists(trt_file_path):
        print(f"Read engine from {trt_file_path}")
        with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as trt_runtime:
            return trt_runtime.deserialize_cuda_engine(f.read())
    else:
        print(f"There is no TensorRT engine {trt_file_path}")
        exit(1)

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    stream = cuda.Stream()
    return inputs, outputs, bindings, stream


def inference_engine(trt_engine, b_align_crop_tenor, latend_id):
    trt_context = trt_engine.create_execution_context()
    trt_context.set_binding_shape(0, (16, 3, 244, 244))
    trt_context.set_binding_shape(1, (16, 512))
    inputs, outputs, bindings, stream = allocate_buffers(trt_engine)

    # trt_context.set_optimization_profile_async(0, stream.handle)
    # trt_context.set_optimization_profile_async(1, stream.handle)

    inputs[0].host = np.ascontiguousarray(b_align_crop_tenor, dtype=np.float16)
    inputs[1].host = np.ascontiguousarray(latend_id, dtype=np.float16)
    [cuda.memcpy_htod_async(input.device, input.host, stream) for input in inputs]
    # print("b_align_crop_tenor: ", inputs[0].host-b_align_crop_tenor.ravel())
    # print("latend_id: ", inputs[1].host - latend_id.ravel())
    trt_context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(output.host, output.device, stream) for output in outputs]
    stream.synchronize()
    return [output.host for output in outputs]

if __name__ == "__main__":
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    trt_file_path = 'dynamic_batch.plan'
    trt_engine = get_engine(trt_file_path)

    b_align_crop_tenor = torch.Tensor(np.load("b_align_crop_tensor_input0.npy")).contiguous()
    latend_id = torch.Tensor(np.load("latend_input.npy")).contiguous()
    # print("b_align_crop_tenor: ", b_align_crop_tenor)
    # print("latend_id: ", latend_id)
    batch_b = np.repeat(b_align_crop_tenor, 16, axis=0).contiguous() # shape : (16, 3, 244, 244)
    batch_l = np.repeat(latend_id, 16, axis=0).contiguous() # shape : (16, 512)
    # print(batch_b.shape)
    with nvtx_range("Start model"):
        result = np.array(inference_engine(trt_engine, batch_b, batch_l))

    output = np.load("swap result0.npy")
    print(output.shape)
    print(result.shape)
    result_np = result.reshape((16, 3, 224, 224))
    # np.save("trt_result.npy", result_np)

    # diff = output - result_np
    # if diff.sum() == 0:
    #     print("TensorRT is success")
    # else:
    #     print(diff.sum())
    diff = result_np[0]-output
    if diff.sum() == 0:
        print("TensorRT is success")
    else:
        print(diff.sum())