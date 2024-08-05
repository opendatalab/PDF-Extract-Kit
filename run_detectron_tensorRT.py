import torch
from rough_layout import *
with open('configs/model_configs.yaml') as f:
    model_configs = yaml.load(f, Loader=yaml.FullLoader)


img_size  = model_configs['model_args']['img_size']
conf_thres= model_configs['model_args']['conf_thres']
iou_thres = model_configs['model_args']['iou_thres']
device    = model_configs['model_args']['device']
dpi       = model_configs['model_args']['pdf_dpi']
layout_model = get_layout_model(model_configs)

batched_inputs = torch.load("convension/detectron2/batched_inputs.pt")

import torch_tensorrt

model = layout_model.predictor.model.eval()
x = batched_inputs[0]['image'][None]

inputs = [x]
trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)
import os
os.makedirs("models/layout/",exist_ok=True)
torch_tensorrt.save(trt_gm, "models/layout//trt.ep", inputs=inputs) # PyTorch only supports Python runtime for an ExportedProgram. For C++ deployment, use a TorchScript file
torch_tensorrt.save(trt_gm, "models/layout//trt.ts", output_format="torchscript", inputs=inputs)
