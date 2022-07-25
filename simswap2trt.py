import sys
from os import path
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import networks
from models import create_model
from base_model import BaseModel
from norm import SpecificNorm
from argparse import ArgumentParser
import json
import time
import onnx


# 이미지로 추출하고 싶은 경우 test_wholeimage_swapsingle.py에 npy 파일 넣어서 볼 수 있음
if __name__ == '__main__':
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()
    # pic_a_path = "/workspace/simswap2trt/data/anne.jpeg"
    # device = torch.device("cuda:0")

    # output_x = np.load("latend_input.npy")
    # output_tensor = torch.from_numpy(output_x)
    # latend_id = output_tensor.to(device)

    # output_x = np.load("b_align_crop_tensor_input0.npy")
    # output_tensor = torch.from_numpy(output_x)
    # b_align_crop_tenor = output_tensor.to(device)

    # x1 = torch.ones(1, 3, 224, 224).cuda()
    # x2 = torch.ones(1, 512).cuda()
    # # scripted_module = torch.jit.trace(simswap_model, (x1, x2))
    # # scripted_module.save("check_layer.pt")

    # scripted_module =torch.jit.load("Generator_Adain_Upsample_torchscript.pt")

    # output = scripted_module(b_align_crop_tenor, latend_id)
    # dynamic_axes = {'input_0' : {0 : 'batch_size'},
    #                 'output_0' : {0 : 'batch_size'}}

    # torch.onnx.export(
    #     scripted_module,
    #     (x1, x2),
    #     'dynamic_batch_Generator_Adain_Upsample_torchscript.onnx',
    #     input_names=['x1','x2'],
    #     output_names=['outputs'],
    #     export_params=True,
    #     #example_outputs=scripted_module(x1, x2),
    #     opset_version=11,
    #     dynamic_axes = dynamic_axes)

    # # ##### 4. Check ONNX model #####
    # onnx_model = onnx.load('dynamic_batch_Generator_Adain_Upsample_torchscript.onnx')
    # print(onnx.checker.check_model(onnx_model))


    # import onnxruntime as ort

    # ort_sess = ort.InferenceSession('dynamic_batch_folded.onnx')
    # #ort_outs = ort_sess.run(None, {'b_align_crop_tenor': b_align_crop_tenor, 'latend_id':latend_id})

    # latend_id = np.load("latend_input.npy")

    # b_align_crop_tenor = np.load("b_align_crop_tensor_input0.npy")
 
    # ort_inputs = {ort_sess.get_inputs()[0].name: b_align_crop_tenor, ort_sess.get_inputs()[1].name: latend_id}
    # ort_outs = ort_sess.run(None, ort_inputs)

    # swap_result = ort_outs
    # np.save("folded_onnx_result.npy", swap_result)

    # output = np.load("swap result0.npy")

    # diff = swap_result - output
    # if diff.sum() == 0:
    #     print("good~")
    # else:
    #     print(diff.sum())
    

    # end.record()
    # torch.cuda.synchronize()
    # print("time :", start.elapsed_time(end)/1000)

## 에러 :
## AttributeError: 'NoneType' object has no attribute 'serialize' -> folded하니까 해결!
    import tensorrt as trt

    onnx_file_name = 'dynamic_batch_folded.onnx'
    trt_file_name = 'dynamic_batch.plan'
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # builder = trt.Builder(TRT_LOGGER)
    # network = builder.create_network(explicit_batch_flag)
    # print("network.num_layers", network.num_layers) #0,,,,,,...............

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
        
        builder_config.max_workspace_size = (1 << 30)
        # Set FP16 modie
        builder_config.set_flag(trt.BuilderFlag.FP16)
        builder.max_batch_size = 16

        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_file_name, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(f"Parser Error: {parser.get_error(error)}")
        # network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
        #engine = builder.build_engine(network, builder_config)
        engine = builder.build_serialized_network(network, builder_config)
        #buf = engine.serialize()
        print("network.num_layers", network.num_layers)
        with open(trt_file_name, 'wb') as f:
            f.write(engine)