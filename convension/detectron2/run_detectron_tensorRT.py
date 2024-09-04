# import torch
# from rough_layout import *
# with open('configs/model_configs.yaml') as f:
#     model_configs = yaml.load(f, Loader=yaml.FullLoader)


# img_size  = model_configs['model_args']['img_size']
# conf_thres= model_configs['model_args']['conf_thres']
# iou_thres = model_configs['model_args']['iou_thres']
# device    = model_configs['model_args']['device']
# dpi       = model_configs['model_args']['pdf_dpi']
# layout_model = get_layout_model(model_configs)

# batched_inputs = torch.load("convension/detectron2/batched_inputs.pt")

# import torch_tensorrt

# model = layout_model.predictor.model.eval()
# x = batched_inputs[0]['image'][None]

# inputs = [x]
# trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)
# import os
# os.makedirs("models/layout/",exist_ok=True)
# torch_tensorrt.save(trt_gm, "models/layout//trt.ep", inputs=inputs) # PyTorch only supports Python runtime for an ExportedProgram. For C++ deployment, use a TorchScript file
# torch_tensorrt.save(trt_gm, "models/layout//trt.ts", output_format="torchscript", inputs=inputs)
import os 
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
from rough_layout import *
with open('configs/model_configs.yaml') as f:
    model_configs = yaml.load(f, Loader=yaml.FullLoader)

img_size  = model_configs['model_args']['img_size']
conf_thres= model_configs['model_args']['conf_thres']
iou_thres = model_configs['model_args']['iou_thres']
device    = model_configs['model_args']['device']
dpi       = model_configs['model_args']['pdf_dpi']
layout_model = get_layout_model(model_configs)
mfd_model    = get_batch_YOLO_model(model_configs) 

self = layout_model.predictor.model

batched_inputs = torch.load("convension/detectron2/batched_inputs.pt")
with torch.no_grad():
    images   = self.preprocess_image(batched_inputs)
    input    = self.get_batch(batched_inputs, images)
    features = self.backbone(input)
    proposals, _ = self.proposal_generator(images, features, None)
    results, _ = self.roi_heads(images, features, proposals, None)

model = layout_model.predictor.model.backbone
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, images):
        with torch.inference_mode():
            outputs = self.model({'images':images})
        return (
            outputs['p2'],
            outputs['p3'],
            outputs['p4'],
            outputs['p5'],
            outputs['p6'],
        )

# Wrap the model
wrapped_model = ModelWrapper(model)

import torch
import torch.onnx

# Assuming `model` is your pre-trained model
model.eval()  # Set the model to evaluation mode
# Create a sample input tensor with the shape (B, 3, 1052, 800)
batch_size = 1  # Adjust as needed
sample_input = torch.randn(batch_size, 3, 1056, 800).cuda()

# Define the path where the ONNX model will be saved
onnx_model_path = "model.onnx"

# Export the model
torch.onnx.export(
    wrapped_model,                  # The wrapped model to be exported
    (sample_input,),                # The input example (tuple)
    onnx_model_path,                # The path where the model will be saved
    input_names=["images"],         # The names of the input tensors
    output_names=['p2','p3','p4','p5','p6'],  # The names of the output tensors
    # dynamic_axes={
    #     'images': {0: 'batch_size'},          # Variable length axes for input
    #     'p2': {0: 'batch_size'},          # Variable length axes for output
    #     'p3': {0: 'batch_size'},
    #     'p4': {0: 'batch_size'},
    #     'p5': {0: 'batch_size'},
    #     'p6': {0: 'batch_size'}
    # },
    opset_version=17 # ONNX opset version (can be adjusted)
)
