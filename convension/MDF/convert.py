import sys
import os
__dir__ = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__dir__)))
from get_batch_yolo import mfd_process, get_batch_YOLO_model
import yaml
with open('configs/model_configs.yaml') as f:
    model_configs = yaml.load(f, Loader=yaml.FullLoader)

img_size  = model_configs['model_args']['img_size']
conf_thres= model_configs['model_args']['conf_thres']
iou_thres = model_configs['model_args']['iou_thres']
device    = model_configs['model_args']['device']
dpi       = model_configs['model_args']['pdf_dpi']

inner_batch_size = 32
mfd_model    = get_batch_YOLO_model(model_configs,inner_batch_size) 
mfd_model.export(format="engine",half=True,imgsz=(1888,1472), batch=inner_batch_size, simplify=True)  # creates 'yolov8n.engine'
oldname = model_configs['model_args']['mfd_weight']
os.rename(
    oldname[:-3]+f'.engine',
    oldname[:-3]+f'.b{inner_batch_size}.engine'

)