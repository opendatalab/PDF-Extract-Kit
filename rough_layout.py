


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

UNIFIED_WIDTH  = 1472  # lets always make the oimage in such size
UNIFIED_HEIGHT = 1920  # lets always make the oimage in such size
def pad_image_to_ratio(image, output_width = UNIFIED_WIDTH,output_height=UNIFIED_HEIGHT, ):
    """
    Pads the given PIL.Image object to fit the specified width-height ratio
    by adding padding only to the bottom and right sides.

    :param image: PIL.Image object
    :param target_ratio: Desired width/height ratio (e.g., 16/9)
    :return: New PIL.Image object with the padding applied
    """
    # Original dimensions
    input_width, input_height = image.size
    height = min(input_height, output_height)
    width  = min(input_width,   output_width)

    if output_height == input_height and output_width == input_width:
        return image

    if input_height / output_height > input_width / output_width:
        # Resize to match height, width will be smaller than output_width
        height = output_height
        width = int(input_width * output_height / input_height)
    else:
        # Resize to match width, height will be smaller than output_height
        width = output_width
        height = int(input_height * output_width / input_width)
    image= image.resize((width, height), resample=3)
    # Create new image with target dimensions and a white background
    new_image = Image.new("RGB", (output_width, output_height), (255, 255, 255))
    new_image.paste(image, (0, 0))

    return new_image

def process_pdf_page_to_image(page, dpi):
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    if pix.width > 3000 or pix.height > 3000:
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    else:
        image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)

    image = pad_image_to_ratio(image, output_width = UNIFIED_WIDTH,output_height=UNIFIED_HEIGHT)
    
    image = np.array(image)[:,:,::-1]
    return image.copy()

class PDFImageDataset(IterableDataset):
    client = None
    #client = build_client()
    def __init__(self, metadata_filepath, aug, input_format, 
                 mfd_pre_transform,
                 return_original_image=False):
        super().__init__()
        self.metadata= self.smart_read_json(metadata_filepath)
        #self.pathlist= [t['path'] for t in self.metadata]
        self.last_read_pdf_buffer = {}
        self.dpi = 200
        self.aug = aug
        self.input_format = input_format
        self.mfd_pre_transform = mfd_pre_transform
        self.return_original_image = return_original_image
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

    def prepare_for_text_det(self,image):
        return self.text_det_transform(image)[0]

    def read_data_based_on_current_state(self):
        page  = self.get_pdf_by_index(self.current_pdf_index).load_page(self.current_page_index)
        oimage = process_pdf_page_to_image(page, self.dpi)
        original_image = oimage[:, :, ::-1] if self.input_format == "RGB" else oimage
        height, width = original_image.shape[:2]
        layout_image = self.aug.get_transform(original_image).apply_image(original_image)
        layout_image = torch.as_tensor(layout_image.astype("float32").transpose(2, 0, 1))[:,:1042,:800] ## it will be 1043x800 --> 1042:800
        ## lets make sure the image has correct size 
        # if layout_image.size(1) < 1042:
        #     layout_image = torch.nn.functional.pad(layout_image, (0, 0, 0, 1042-layout_image.size(1)))
        mfd_image=self.prepare_for_mfd_model(oimage)

        #print(self.current_pdf_index, self.current_page_index)
        return self.current_pdf_index, self.current_page_index, mfd_image, layout_image, height, width, oimage

    def check_should_skip(self):
        if self.current_pdf_index >= len(self.metadata):
            raise StopIteration
        
        if self.current_page_index >= self.get_pdf_by_index(self.current_pdf_index).page_count:
            worker_info = get_worker_info()
            step_for_pdf= 1 if worker_info is None else worker_info.num_workers
            self.current_pdf_index += step_for_pdf
            self.current_page_index = 0

            if self.current_pdf_index >= len(self.metadata):
                raise StopIteration

    def __next__(self):
        self.check_should_skip()
        output = self.read_data_based_on_current_state()
        self.current_page_index += 1
        # fail_times = 0
        # try:
        #     self.check_should_skip()
        #     output = self.read_data_based_on_current_state()
        #     self.current_page_index += 1
        #     fail_times = 0
        # except StopIteration:
        #     raise StopIteration
        # except:
        #     fail_times +=1
        #     if fail_times>10:
        #         raise StopIteration
        return output
        
    

       

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
    current_pdf_indexes = []
    current_page_indexes = []
    mfd_images = []
    layout_images = []
    heights = []
    widths = []
    oimages = []


    for current_pdf_index, current_page_index, mfd_image, layout_image, height, width, oimage in batch:
        current_pdf_indexes.append(current_pdf_index)
        current_page_indexes.append(current_page_index)
        mfd_images.append(mfd_image)
        layout_images.append(layout_image)
        heights.append(height)
        widths.append(width)
        oimages.append(oimage)
    
    mfd_images = torch.stack(mfd_images)
    layout_images = torch.stack(layout_images)
    heights = torch.tensor(heights)
    widths = torch.tensor(widths)
    current_pdf_indexes = torch.tensor(current_pdf_indexes)
    current_page_indexes = torch.tensor(current_page_indexes)
    return current_pdf_indexes, current_page_indexes, mfd_images, layout_images, heights, widths, oimages

def clean_layout_dets(layout_dets):
    rows = []
    for t in layout_dets:
        rows.append({
        "category_id":int(t['category_id']),
        "poly":[int(t) for t in t['poly']],
        "score":float(t['score'])
        })
        
    return rows
from ultralytics.utils import ops
import copy
from modules.self_modify import ModifiedPaddleOCR,update_det_boxes,sorted_boxes
from utils import *


def deal_with_one_dataset(pdf_path, result_path, layout_model, mfd_model, ocrmodel=None, inner_batch_size=4, batch_size=32,num_workers=8,timer=Timers(False)):
    dataset    = PDFImageDataset(pdf_path,layout_model.predictor.aug,layout_model.predictor.input_format,
                                mfd_pre_transform=mfd_process(mfd_model.predictor.args.imgsz,mfd_model.predictor.model.stride,mfd_model.predictor.model.pt))

    dataloader = DataLoader(dataset, batch_size=batch_size,collate_fn=custom_collate_fn, num_workers=num_workers)        
    featcher   = DataPrefetcher(dataloader,device='cuda')
    data_to_save = {}
    inner_batch_size = inner_batch_size
    pbar  = tqdm(total=len(dataset.metadata),position=2,desc="PDF Pages",leave=False)
    pdf_passed = set()
    batch = featcher.next()
    while batch is not None:
        # try:
        pdf_index_batch,page_ids_batch, mfd_images_batch,images_batch,heights_batch, widths_batch,oimage_list = batch
        pdf_index = set([t.item() for t in pdf_index_batch])
        new_pdf_processed = pdf_index - pdf_passed
        pdf_passed        = pdf_passed|pdf_index
        
        for j in tqdm(range(0, len(mfd_images_batch), inner_batch_size),position=3,leave=False,desc="mini-Batch"):
            pdf_index  = pdf_index_batch[j:j+inner_batch_size]
            page_ids   = page_ids_batch[j:j+inner_batch_size]
            mfd_images = mfd_images_batch[j:j+inner_batch_size]
            images     = images_batch[j:j+inner_batch_size]
            heights    = heights_batch[j:j+inner_batch_size]
            widths     = widths_batch[j:j+inner_batch_size]
            oimages    = oimage_list[j:j+inner_batch_size]
            with timer('get_layout'):
                layout_res = layout_model((images,heights, widths), ignore_catids=[])
            with timer('get_mfd'):
                mfd_res    = mfd_model.predict(mfd_images, imgsz=(1888,1472), conf=0.3, iou=0.5, verbose=False)
            
            with timer('combine_layout_mfd_result'):
                rough_layout_this_batch =[]
                ori_shape_list = []
                pdf_and_page_id_this_batch = []
                for pdf_id, page_id, layout_det, mfd_det, real_input_height, real_input_width in zip(pdf_index, page_ids, layout_res, mfd_res, heights, widths):
                    mfd_height,mfd_width = mfd_det.orig_shape
                    pdf_id = int(pdf_id)
                    page_id= int(page_id)
                    real_input_height = int(real_input_height)
                    real_input_width  = int(real_input_width)
                    pdf_path = dataset.metadata[pdf_id]['path']
                    if pdf_path not in data_to_save:
                        data_to_save[pdf_path] = {'height':real_input_height, 'width':real_input_width}
                    layout_dets = clean_layout_dets(layout_det['layout_dets'])
                    for xyxy, conf, cla in zip(mfd_det.boxes.xyxy.cpu(), 
                                            mfd_det.boxes.conf.cpu(), 
                                            mfd_det.boxes.cls.cpu()):
                        xyxy =  ops.scale_boxes(mfd_images.shape[2:], xyxy, (real_input_height, real_input_width))
                        xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                        new_item = {
                            'category_id': 13 + int(cla.item()),
                            'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                            'score': round(float(conf.item()), 2),
                            'latex': '',
                        }
                        layout_dets.append(new_item)
                    data_to_save[pdf_path][page_id] = layout_dets
                    ori_shape_list.append((real_input_height, real_input_width))
                    pdf_and_page_id_this_batch.append((pdf_id, page_id))
                    rough_layout_this_batch.append(layout_dets)
            
            if ocrmodel is not None:
                with timer('text_detection/collect_for_line_detect'):
                    canvas_tensor_this_batch, partition_per_batch,canvas_idxes_this_batch,single_page_mfdetrec_res_this_batch = collect_paragraph_image_and_its_coordinate(oimages, rough_layout_this_batch)
                with timer('text_detection/get_line_detect_result'):
                    dt_boxes_list = ocrmodel.batch_det_model.batch_process(canvas_tensor_this_batch,ori_shape_list=ori_shape_list)
                with timer('text_detection/collect_for_text_images'):
                    text_image_batch, text_image_position,text_line_bbox = collect_text_image_and_its_coordinate(single_page_mfdetrec_res_this_batch, partition_per_batch, canvas_tensor_this_batch,dt_boxes_list)
                with timer('text_detection/get_line_text'):
                    rec_res, elapse = ocrmodel.text_recognizer(text_image_batch)
                for line_box, rec_result,(partition_id,text_block_id, text_line_id) in zip(text_line_bbox, rec_res,text_image_position):
                    text, score = rec_result
                    pdf_id, page_id = pdf_and_page_id_this_batch[partition_id]
                    pdf_path = dataset.metadata[pdf_id]['path']
                    p1, p2, p3, p4 = line_box.tolist()
                    data_to_save[pdf_path][page_id].append(
                        {
                            'category_id': 15,
                            'poly': p1 + p2 + p3 + p4,
                            'score': round(score, 2),
                            'text': text,
                        }

                    )

        pbar.update(len(new_pdf_processed))
        # except:
        #     print("ERROR: Fail to process batch")
        timer.log()
        batch = featcher.next()

    ### next, we construct each result for each pdf in pdf wise and remove the page_id by the list position 
    pdf_to_metadata = {t['path']:t for t in dataset.metadata}

    new_data_to_save = []
    for pdf_path, layout_dets_per_page in data_to_save.items():
        new_pdf_dict = copy.deepcopy(pdf_to_metadata[pdf_path])
        new_pdf_dict['height'] = layout_dets_per_page.pop('height')
        new_pdf_dict['width'] = layout_dets_per_page.pop('width')
        pages = [t for t in layout_dets_per_page.keys()]
        pages.sort()
        new_pdf_dict["doc_layout_result"]=[]
        for page_id in range(max(pages)):
            if page_id not in layout_dets_per_page:
                print(f"WARNING: page {page_id} of PDF: {pdf_path} fail to parser!!! ")
                now_row = {"page_id": page_id, "status": "fail", "layout_dets":[]}
            else:
                now_row = {"page_id": page_id, "layout_dets":layout_dets_per_page[page_id]}
            new_pdf_dict["doc_layout_result"].append(now_row)
        new_data_to_save.append(new_pdf_dict)
    if dataset.client is None:dataset.client=build_client()
    write_jsonl_to_path(new_data_to_save, result_path, dataset.client)

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
    ocrmodel = None
    ocrmodel = ocr_model = ModifiedPaddleOCR(show_log=True)
    timer = Timers(True)
    deal_with_one_dataset("debug.jsonl", 
                          "debug.stage_1.jsonl", 
                          layout_model, mfd_model, ocrmodel=ocrmodel, inner_batch_size=3, batch_size=12,num_workers=4,timer=timer)
    # dataset    = PDFImageDataset("part-66210c190659-000035.jsonl",layout_model.predictor.aug,layout_model.predictor.input_format,mfd_pre_transform=None)
    # dataloader = DataLoader(dataset, batch_size=8,collate_fn=custom_collate_fn)  

    
    