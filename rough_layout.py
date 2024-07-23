


from get_data_utils import *

from typing import List
from PIL import Image
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from transformers import NougatImageProcessor
UNIFIED_WIDTH  = 1650
UNIFIED_HEIGHT = 2150
def process_pdf_page_to_image(page, dpi):
    

    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    
    # if pix.width > UNIFIED_WIDTH or pix.height > UNIFIED_HEIGHT:
    #     pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    #     image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)


    # Resize the image while maintaining aspect ratio
    img_aspect = image.width / image.height
    target_aspect = UNIFIED_WIDTH / UNIFIED_HEIGHT

    if img_aspect > target_aspect:
        # Image is wider than target aspect ratio
        new_width = UNIFIED_WIDTH
        new_height = int(UNIFIED_WIDTH / img_aspect)
    else:
        # Image is taller than target aspect ratio
        new_width = int(UNIFIED_HEIGHT * img_aspect)
        new_height = UNIFIED_HEIGHT

    # Resize the image
    image = image.resize((new_width, new_height), Image.ANTIALIAS)

    pad_width = (UNIFIED_WIDTH - image.width) // 2
    pad_height = (UNIFIED_HEIGHT - image.height) // 2
    
    # Create a new image with the unified size and paste the resized image onto it with padding
    new_image = Image.new('RGB', (UNIFIED_WIDTH, UNIFIED_HEIGHT), (255, 255, 255))
    new_image.paste(image, (pad_width, pad_height))
   
    return new_image


class PDFImageDataset(IterableDataset):
    client = None
    def __init__(self, metadata_filepath):
        super().__init__()
        self.metadata= self.smart_read_json(metadata_filepath)
        self.pathlist= [t['path'] for t in self.metadata]
        self.last_read_pdf_buffer = {}
        self.dpi = 200

    def smart_read_json(self, json_path):
        if "s3" in json_path and self.client is None: self.client = build_client()
        if json_path.startswith("s3"): json_path = "opendata:"+ json_path
        return read_json_from_path(json_path, self.client)
    
    def smart_write_json(self, data, targetpath):
        if "s3" in targetpath and self.client is None: self.client = build_client()
        if json_path.startswith("s3"): json_path = "opendata:"+ json_path
        write_json_to_path(data, targetpath, self.client)
    
    def smart_load_pdf(self, pdf_path):
        if "s3" in pdf_path and self.client is None: self.client = build_client()
        if pdf_path.startswith("s3"): pdf_path = "opendata:"+ pdf_path
        return read_pdf_from_path(pdf_path, self.client)
    
    def clean_pdf_buffer(self):
        keys = list(self.last_read_pdf_buffer.keys())
        for key in keys:
            self.last_read_pdf_buffer[key].close()
            del self.last_read_pdf_buffer[key]

    def get_pdf_buffer(self,path):
        if "s3" in path and self.client is None: self.client = build_client()
        if path.startswith("s3"): path = "opendata:"+ path
        if path not in self.last_read_pdf_buffer:
            self.clean_pdf_buffer()
            self.last_read_pdf_buffer[path] = self.smart_load_pdf(path)
        pdf_buffer = self.last_read_pdf_buffer[path]
        return pdf_buffer
    
    def get_pdf_by_index(self,index):
        pdf_path  = self.pathlist[index]
        return self.get_pdf_buffer(pdf_path)

    
    
        
    def __len__(self):
        return len(self.pathlist)
    
    def __iter__(self):
        self.current_pdf_index = 0
        self.current_page_index = 0
        self.last_read_pdf_buffer = {}
        return self
    
    @property
    def current_doc(self):
        if len(self.last_read_pdf_buffer)==0:return None
        return list(self.last_read_pdf_buffer.values())[0]

    def __next__(self):
        if self.current_pdf_index >= len(self.pathlist):
            raise StopIteration
        
        if self.current_page_index >= self.get_pdf_by_index(self.current_pdf_index).page_count:
            self.current_pdf_index += 1
            self.current_page_index = 0

            if self.current_pdf_index >= len(self.pathlist):
                raise StopIteration

        page  = self.get_pdf_by_index(self.current_pdf_index).load_page(self.current_page_index)
        image = process_pdf_page_to_image(page, self.dpi)
 
        image = np.array(image)[:,:,::-1].copy()
        
        return image
    
dataset = PDFImageDataset("part-66210c190659-000035.jsonl")
dataloader = DataLoader(dataset, batch_size=4)        

import yaml
from modules.layoutlmv3.model_init import Layoutlmv3_Predictor
def rough_layout(layout_model, image):
    layout_res = layout_model(image, ignore_catids=[])
    return layout_res
def layout_model_init(weight):
    model = Layoutlmv3_Predictor(weight)
    return model

with open('configs/model_configs.yaml') as f:
    model_configs = yaml.load(f, Loader=yaml.FullLoader)
layout_model = layout_model_init(model_configs['model_args']['layout_weight'])
img_size  = model_configs['model_args']['img_size']
conf_thres= model_configs['model_args']['conf_thres']
iou_thres = model_configs['model_args']['iou_thres']
device    = model_configs['model_args']['device']
dpi       = model_configs['model_args']['pdf_dpi']

for i, images in enumerate(dataloader):
    print(f"Batch {i + 1}")
    print(images.shape)  # Should print torch.Size([4, 3, 224, 224])
    layout_res = layout_model(images, ignore_catids=[])
    print(layout_res.shape)  # Should print torch.Size([4, 3, 224, 224])
