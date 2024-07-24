


from get_batch_yolo import mfd_process, get_batch_YOLO_model
from get_batch_layout_model import get_layout_model
from get_data_utils import *
from typing import List
from PIL import Image
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import torch
import numpy as np
from tqdm.auto import tqdm
import yaml
from dataaccelerate import DataPrefetcher 
from torch.utils.data import IterableDataset, get_worker_info

UNIFIED_WIDTH  = 1650
UNIFIED_HEIGHT = 2150
def pad_image_to_ratio(image, target_ratio):
    """
    Pads the given PIL.Image object to fit the specified width-height ratio
    by adding padding only to the bottom and right sides.

    :param image: PIL.Image object
    :param target_ratio: Desired width/height ratio (e.g., 16/9)
    :return: New PIL.Image object with the padding applied
    """
    # Original dimensions
    orig_width, orig_height = image.size
    orig_ratio = orig_width / orig_height

    # Calculate target dimensions
    if orig_ratio < target_ratio:
        # Taller than target ratio, pad width
        target_height = orig_height
        target_width = int(target_height * target_ratio)
    else:
        # Wider than target ratio, pad height
        target_width = orig_width
        target_height = int(target_width / target_ratio)

    # Calculate padding needed
    pad_right = target_width - orig_width
    pad_bottom = target_height - orig_height

    # Create new image with target dimensions and a white background
    new_image = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    new_image.paste(image, (0, 0))

    return new_image

def process_pdf_page_to_image(page, dpi):
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    if pix.width > 3000 or pix.height > 3000:
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    else:
        image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)

    image = pad_image_to_ratio(image, UNIFIED_WIDTH / UNIFIED_HEIGHT)
    
    image = np.array(image)[:,:,::-1]
    return image.copy()

class PDFImageDataset(IterableDataset):
    client = None
    #client = build_client()
    def __init__(self, metadata_filepath, aug, input_format, mfd_pre_transform):
        super().__init__()
        self.metadata= self.smart_read_json(metadata_filepath)
        #self.pathlist= [t['path'] for t in self.metadata]
        self.last_read_pdf_buffer = {}
        self.dpi = 200
        self.aug = aug
        self.input_format = input_format
        self.mfd_pre_transform = mfd_pre_transform
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
        pdf_path  = self.metadata[index]['path']
        return self.get_pdf_buffer(pdf_path)

    
    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            self.current_pdf_index = 0
            self.current_page_index = 0
        else:  # in a worker process
            # split workload
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            self.current_pdf_index = worker_id
            self.current_page_index = 0

        self.pdf = self.get_pdf_by_index(self.current_pdf_index)
        return self
    
    @property
    def current_doc(self):
        if len(self.last_read_pdf_buffer)==0:return None
        return list(self.last_read_pdf_buffer.values())[0]

    def __next__(self):
        if self.current_pdf_index >= len(self.metadata):
            raise StopIteration
        
        if self.current_page_index >= self.get_pdf_by_index(self.current_pdf_index).page_count:
            worker_info = get_worker_info()
            step_for_pdf= 1 if worker_info is None else worker_info.num_workers
            self.current_pdf_index += step_for_pdf
            self.current_page_index = 0

            if self.current_pdf_index >= len(self.metadata):
                raise StopIteration

        page  = self.get_pdf_by_index(self.current_pdf_index).load_page(self.current_page_index)
        current_page_index = self.current_page_index
        current_pdf_index  = self.current_pdf_index
        self.current_page_index += 1
        oimage = process_pdf_page_to_image(page, self.dpi)
        original_image = oimage[:, :, ::-1] if self.input_format == "RGB" else oimage
        height, width = original_image.shape[:2]
        image = self.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[:,:1042,:800]
        mfd_image=self.prepare_for_mfd_model(oimage)
        #print(self.current_pdf_index, self.current_page_index)
        return current_pdf_index, current_page_index, mfd_image, image, height, width
    
    def prepare_for_mfd_model(self, im:np.ndarray):
        if self.mfd_pre_transform is None :return im
        assert im.ndim==3
        im = [im]
        im = np.stack(self.mfd_pre_transform(im))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = im.astype('float')/255
        im = torch.from_numpy(im)
        return im[0]

def custom_collate_fn(batch):
    oimages = []
    images = []
    heights = []
    widths = []

    for oimage, image, height, width in batch:
        oimages.append(oimage)
        images.append(torch.tensor(image))
        heights.append(torch.tensor(height))
        widths.append(torch.tensor(width))

    images = torch.stack(images)
    heights = torch.stack(heights)
    widths = torch.stack(widths)

    return oimages, images, heights, widths

def clean_layout_dets(layout_dets):
    rows = []
    for t in layout_dets:
        rows.append({
        "category_id":int(t['category_id']),
        "poly":[int(t) for t in t['poly']],
        "score":float(t['score'])
        })
        
    return rows

def deal_with_one_dataset(pdf_path, result_path, layout_model, mfd_model, inner_batch_size=4, batch_size=32,num_workers=8):
    dataset    = PDFImageDataset(pdf_path,layout_model.predictor.aug,layout_model.predictor.input_format,
                                mfd_pre_transform=mfd_process(mfd_model.predictor.args.imgsz,mfd_model.predictor.model.stride,mfd_model.predictor.model.pt))

    dataloader = DataLoader(dataset, batch_size=batch_size,collate_fn=None, num_workers=num_workers)        
    featcher   = DataPrefetcher(dataloader,device='cuda')
    data_to_save = []
    inner_batch_size = inner_batch_size
    pbar  = tqdm(total=len(dataset.metadata),position=1,leave=True,desc="PDF Pages")
    pdf_passed = set()
    batch = featcher.next()
    while batch is not None:
        pdf_index_batch,page_ids_batch, oimages_batch,images_batch,heights_batch, widths_batch = batch
    #for pdf_index,oimages_batch,images_batch,heights_batch, widths_batch in dataloader:
        pdf_index = set([t.item() for t in pdf_index_batch])
        new_pdf_processed = pdf_index - pdf_passed
        pdf_passed        = pdf_passed|pdf_index
        
        for j in tqdm(range(0, len(oimages_batch), inner_batch_size),position=2,leave=False,desc="mini-Batch"):
            pdf_index = pdf_index_batch[j:j+inner_batch_size]
            page_ids  = page_ids_batch[j:j+inner_batch_size]
            oimages = oimages_batch[j:j+inner_batch_size]
            images  = images_batch[j:j+inner_batch_size]
            heights = heights_batch[j:j+inner_batch_size]
            widths  = widths_batch[j:j+inner_batch_size]

            layout_res = layout_model((images,heights, widths), ignore_catids=[])
            mfd_res    = mfd_model.predict(oimages, imgsz=img_size, conf=0.3, iou=0.5, verbose=False)
            
            for pdf_id, page_id, layout_det, mfd_det, layout_height, layout_width in zip(pdf_index, page_ids, layout_res, mfd_res, heights, widths):
                mfd_height,mfd_width = mfd_det.orig_shape
                pdf_id = int(pdf_id)
                page_id= int(page_id)
                pdf_path = dataset.metadata[pdf_id]['path']
                this_row = {
                    "pdf_source":pdf_path,
                    "page_id": page_id,
                    "layout": {'height':int(layout_height), 'width':int(layout_width), 'layout_dets':clean_layout_dets(layout_det['layout_dets'])},
                    "mfd":{'height':int(mfd_height), 'width':int(mfd_width), 
                        "bbox_cls" :mfd_det.boxes.cls.cpu().numpy().astype('uint8').tolist(), 
                        "bbox_conf":mfd_det.boxes.conf.cpu().numpy().astype('float16').tolist(), 
                        "bbox_xyxy":mfd_det.boxes.xyxy.cpu().numpy().astype('uint8').tolist(), 
                        }
                }
                data_to_save.append(this_row)
        pbar.update(len(new_pdf_processed))
        batch = featcher.next()

    write_json_to_path
    with open(result_path,'w') as f:
        for row in data_to_save:
            f.write(json.dumps(row)+'\n')

if __name__ == "__main__":

    with open('configs/model_configs.yaml') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)

    img_size  = model_configs['model_args']['img_size']
    conf_thres= model_configs['model_args']['conf_thres']
    iou_thres = model_configs['model_args']['iou_thres']
    device    = model_configs['model_args']['device']
    dpi       = model_configs['model_args']['pdf_dpi']

        
    layout_model = get_layout_model(model_configs)
    mfd_model    = get_batch_YOLO_model(model_configs) 
    deal_with_one_dataset("debug.jsonl", 
                          "debug.stage_1.jsonl", 
                          layout_model, mfd_model, inner_batch_size=16, batch_size=64,num_workers=16)
    # dataset    = PDFImageDataset("part-66210c190659-000035.jsonl",layout_model.predictor.aug,layout_model.predictor.input_format,mfd_pre_transform=None)
    # dataloader = DataLoader(dataset, batch_size=8,collate_fn=custom_collate_fn)  

    
    