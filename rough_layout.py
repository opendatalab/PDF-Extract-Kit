

import os 
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
from get_batch_yolo import mfd_process, get_batch_YOLO_model
from get_batch_layout_model import get_layout_model
from utils import *
from utils import Timers
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
import logging
logging.getLogger('detectron2').setLevel(logging.WARNING)

class PDFImageDataset(IterableDataset,DatasetUtils):
    #client = build_client()
    def __init__(self, metadata_filepath, aug, input_format, 
                 mfd_pre_transform, det_pre_transform=None,
                 return_original_image=False,timer=Timers(False)):
        super().__init__()
        self.metadata= self.smart_read_json(metadata_filepath)
        self.dpi = 200
        self.aug = aug
        self.input_format = input_format
        self.mfd_pre_transform = mfd_pre_transform
        self.return_original_image = return_original_image
        self.det_pre_transform = det_pre_transform
        self.timer = timer
    
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
        with self.timer("load_page"):
            page  = self.get_pdf_by_index(self.current_pdf_index).load_page(self.current_page_index)
        with self.timer("from_page_to_pimage"):
            oimage = process_pdf_page_to_image(page, self.dpi)
        original_image = oimage[:, :, ::-1] if self.input_format == "RGB" else oimage
        height, width = original_image.shape[:2]
        with self.timer("get_layout_image"):
            layout_image = self.aug.get_transform(original_image).apply_image(original_image)
            layout_image = torch.as_tensor(layout_image.astype("float32").transpose(2, 0, 1))[:,:1042,:800] ## it will be 1043x800 --> 1042:800
        ## lets make sure the image has correct size 
        # if layout_image.size(1) < 1042:
        #     layout_image = torch.nn.functional.pad(layout_image, (0, 0, 0, 1042-layout_image.size(1)))
        with self.timer("get_mfd_image"):
            mfd_image=self.prepare_for_mfd_model(oimage)
        with self.timer("get_det_image"):
            det_images = torch.from_numpy(self.det_pre_transform(original_image)[0])
        
        output= {"pdf_index":self.current_pdf_index, "page_index":self.current_page_index, "mfd_image":mfd_image, "layout_image":layout_image, "det_images":det_images, "height":height, "width":width}
        if self.return_original_image:
            output['oimage'] = original_image
        return output
    
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
        # self.check_should_skip()
        # with self.timer("read_data_based_on_current_state"):
        #     output = self.read_data_based_on_current_state()
        # self.current_page_index += 1
        fail_times = 0
        try:
            self.check_should_skip()
            output = self.read_data_based_on_current_state()
            self.current_page_index += 1
            fail_times = 0
        except StopIteration:
            raise StopIteration
        except:
            fail_times +=1
            self.current_page_index += 1
            if fail_times>10:
                raise StopIteration
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

def custom_collate_fn(batches):
    
    return_batch = {}
    for batch in batches:
        for key,val in batch.items():
            if key not in return_batch:
                return_batch[key] = []
            return_batch[key].append(val)
    
    keys = list(return_batch.keys())
    for key in keys:
        if key in ["pdf_index", "page_index","height", "width"]:
            return_batch[key] = torch.tensor(return_batch[key])
        elif key in ["mfd_image", "layout_image", "det_images"]:
            return_batch[key] = torch.stack(return_batch[key])
        elif key in ['oimage']:
            return_batch[key] = return_batch[key]
    return return_batch
    

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

def process_batch(inputs):
    preds, shape_list,ori_shape = inputs
    post_result = ocrmodel.batch_det_model.fast_postprocess(preds, shape_list)
    dt_boxes    = post_result[0]['points']
    dt_boxes    = ocrmodel.batch_det_model.filter_tag_det_res(dt_boxes, ori_shape)
    return dt_boxes
def deal_with_one_dataset(pdf_path, result_path, layout_model, mfd_model, 
                          ocrmodel=None, inner_batch_size=4, 
                          batch_size=32,num_workers=8,
                          do_text_det=False,
                          do_text_rec=False,
                          timer=Timers(False)):
    dataset    = PDFImageDataset(pdf_path,layout_model.predictor.aug,layout_model.predictor.input_format,
                                 
                                 mfd_pre_transform=mfd_process(mfd_model.predictor.args.imgsz,mfd_model.predictor.model.stride,mfd_model.predictor.model.pt),
                                 det_pre_transform=ocrmodel.batch_det_model.prepare_image,
                                 return_original_image=do_text_rec
                                 )
    collate_fn = custom_collate_fn if do_text_rec else None
    dataloader = DataLoader(dataset, batch_size=batch_size,collate_fn=collate_fn, 
                            num_workers=num_workers,pin_memory=True, pin_memory_device='cuda',
                            prefetch_factor=3)        
    featcher   = DataPrefetcher(dataloader,device='cuda')
    data_to_save = {}
    inner_batch_size = inner_batch_size
    pbar  = tqdm(total=len(dataset.metadata),position=2,desc="PDF Pages",leave=True)
    pdf_passed = set()
    batch = featcher.next()
    data_loading = []
    model_train  = []
    last_record_time = time.time()
    while batch is not None:
    #for batch in dataloader:
        try:
            data_loading.append(time.time() - last_record_time);last_record_time =time.time() 
            pbar.set_description(f"[Data][{np.mean(data_loading[-10:]):.2f}] [Model][{np.mean(model_train[-10:]):.2f}]")
            pdf_index_batch, page_ids_batch = batch["pdf_index"], batch["page_index"]
            mfd_layout_images_batch, layout_images_batch, det_layout_images_batch = batch["mfd_image"], batch["layout_image"], batch["det_images"]
            heights_batch, widths_batch = batch["height"], batch["width"]
            oimage_list = batch.get('oimage',None)
            pdf_index = set([t.item() for t in pdf_index_batch])
            new_pdf_processed = pdf_index - pdf_passed
            pdf_passed        = pdf_passed|pdf_index
            
            for j in tqdm(range(0, len(mfd_layout_images_batch), inner_batch_size),position=3,leave=False,desc="mini-Batch"):
                pdf_index  = pdf_index_batch[j:j+inner_batch_size]
                page_ids   = page_ids_batch[j:j+inner_batch_size]
                mfd_images = mfd_layout_images_batch[j:j+inner_batch_size]
                images     = layout_images_batch[j:j+inner_batch_size]
                heights    = heights_batch[j:j+inner_batch_size]
                widths     = widths_batch[j:j+inner_batch_size]
                oimages    = oimage_list[j:j+inner_batch_size] if oimage_list is not None else None
                detimages  = det_layout_images_batch[j:j+inner_batch_size]
    
                with timer('get_layout'):
                    if len(images)<inner_batch_size and layout_model.iscompiled:
                        images  = torch.nn.functional.pad(images,   (0,0,0,0,0,0,0, inner_batch_size-len(images)))
                        
                        heights = torch.nn.functional.pad(heights,  (0, inner_batch_size-len(heights)))
                        widths  = torch.nn.functional.pad(widths,   (0, inner_batch_size-len(widths)))

                    layout_res = layout_model((images,heights, widths), ignore_catids=[],dtype=torch.float16)
                    layout_res = layout_res[:len(pdf_index)]
                with timer('get_mfd'):
                    if len(mfd_images)<inner_batch_size:
                        mfd_images = torch.nn.functional.pad(mfd_images, (0,0,0,0,0,0,0, inner_batch_size-len(mfd_images)))
                        #print(mfd_images.shape)
                    mfd_res    = mfd_model.predict(mfd_images, imgsz=(1888,1472), conf=0.3, iou=0.5, verbose=False)
                    mfd_res = mfd_res[:len(pdf_index)]
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
                        assert real_input_height == 1920
                        assert real_input_width  == 1472
                
                if ocrmodel is not None:
                    if not do_text_det:continue
                    with timer('text_detection/collect_for_line_detect'):
                        canvas_tensor_this_batch, partition_per_batch,canvas_idxes_this_batch,single_page_mfdetrec_res_this_batch = collect_paragraph_image_and_its_coordinate(detimages, rough_layout_this_batch,2)
                    shape_list_batch = np.array([[(1920,1472, 0.5, 0.5)]]*len(canvas_tensor_this_batch)) 
                    ori_shape_list   = [(1920,1472)]*len(canvas_tensor_this_batch)
                    with torch.no_grad():
                        with timer('text_detection/stack'):
                            canvas_tensor_this_batch = torch.stack(canvas_tensor_this_batch)
                        with timer('text_detection/det_net'):
                            dt_boxaes_batch = ocrmodel.batch_det_model.net(canvas_tensor_this_batch)
                        # with timer('text_detection/discard_batch'):
                        #     pred_batch  = ocrmodel.batch_det_model.discard_batch(dt_boxaes_batch)
                        with timer('text_detection/batch_postprocess'):
                            #dt_boxes_list=ocrmodel.batch_det_model.batch_postprocess(pred_batch, shape_list_batch,ori_shape_list)
                            preds_list = dt_boxaes_batch
                            with timer('text_detection/batch_postprocess/postprocess_op'):
                                post_result = ocrmodel.batch_det_model.fast_postprocess(preds_list, shape_list_batch[:,0])
                            # dt_boxes    = post_result[0]['points']
                            with timer('text_detection/batch_postprocess/filter'):
                                dt_boxes_list = [ocrmodel.batch_det_model.filter_tag_det_res(dt_boxes['points'][0], ori_shape)
                                                for dt_boxes, ori_shape in zip(post_result,ori_shape_list )]
            
                    if oimages is not None and do_text_rec:
                        with timer('text_detection/collect_for_text_images'):
                            text_image_batch, text_image_position,text_line_bbox = collect_text_image_and_its_coordinate(single_page_mfdetrec_res_this_batch, partition_per_batch, oimages,dt_boxes_list)
                        with timer('text_detection/get_line_text_rec'):
                            rec_res, elapse = ocrmodel.text_recognizer(text_image_batch)
                        for line_box, rec_result,(partition_id,text_block_id, text_line_id) in zip(text_line_bbox, rec_res,text_image_position):
                            text, score = rec_result
                            pdf_id, page_id = pdf_and_page_id_this_batch[partition_id]
                            pdf_path = dataset.metadata[pdf_id]['path']
                            p1, p2, p3, p4 = line_box.tolist()
                            #print(line_box)
                            data_to_save[pdf_path][page_id].append(
                                {
                                    'category_id': 15,
                                    'poly': p1 + p2 + p3 + p4,
                                    'score': round(score, 2),
                                    'text': text,
                                }

                            )
                    else:
                        for partition_id in range(len(partition_per_batch)-1):
                            pdf_id, page_id = pdf_and_page_id_this_batch[partition_id]
                            pdf_path        = dataset.metadata[pdf_id]['path']
                            partition_start = partition_per_batch[partition_id]
                            partition_end   = partition_per_batch[partition_id+1]
                            dt_boxes_this_partition = dt_boxes_list[partition_start:partition_end]
                            for dt_boxes in dt_boxes_this_partition: #(1, 4, 2)
                                for line_box in dt_boxes:
                                    p1, p2, p3, p4 = line_box.tolist()
                                    data_to_save[pdf_path][page_id].append(
                                        {
                                            'category_id': 15,
                                            'poly': p1 + p2 + p3 + p4,
                                        }
                                    )
            pbar.update(len(new_pdf_processed))

        except:
            raise
            print("ERROR: Fail to process batch")
        timer.log()
        model_train.append(time.time() - last_record_time);last_record_time =time.time()
        pbar.set_description(f"[Data][{np.mean(data_loading[-10:]):.2f}] [Model][{np.mean(model_train[-10:]):.2f}]")
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

def test_dataset(pdf_path, layout_model, mfd_model, ocrmodel):
    timer = Timers(True)
    dataset    = PDFImageDataset(pdf_path,layout_model.predictor.aug,layout_model.predictor.input_format,
                                 
                                 mfd_pre_transform=mfd_process(mfd_model.predictor.args.imgsz,
                                                               mfd_model.predictor.model.stride,
                                                               mfd_model.predictor.model.pt),
                                 det_pre_transform=ocrmodel.batch_det_model.prepare_image,
                                 return_original_image=True, timer=timer,
                                 )
    print("======================================================")
    for _ in dataset:
        timer.log()

import threading
from flask import Flask, request, jsonify
import requests
from modules.batch_text_detector import fast_torch_postprocess_multiprocess, PostProcessConfig
app = Flask(__name__)

@app.route('/postprocess', methods=['POST'])
def postprocess():
    data = request.json
    preds_list       = data['preds_list']
    shape_list_batch = data['shape_list_batch']
    ori_shape_list   = data['ori_shape_list']
    
    post_result = fast_torch_postprocess_multiprocess(preds_list, shape_list_batch)
    dt_boxes_list = [
        ocrmodel.batch_det_model.filter_tag_det_res(dt_boxes['points'][0], ori_shape)
        for dt_boxes, ori_shape in zip(post_result, ori_shape_list)
    ]
    return jsonify(dt_boxes_list)

if __name__ == "__main__":

    with open('configs/model_configs.yaml') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)

    img_size  = model_configs['model_args']['img_size']
    conf_thres= model_configs['model_args']['conf_thres']
    iou_thres = model_configs['model_args']['iou_thres']
    device    = model_configs['model_args']['device']
    dpi       = model_configs['model_args']['pdf_dpi']

        
    layout_model = get_layout_model(model_configs)
    inner_batch_size= 16
    mfd_model    = get_batch_YOLO_model(model_configs,inner_batch_size) 
    ocrmodel = None
    ocrmodel = ocr_model = ModifiedPaddleOCR(show_log=True)
    timer = Timers(True,warmup=5)
    #test_dataset("debug.jsonl", layout_model, mfd_model, ocrmodel)
    deal_with_one_dataset("debug.jsonl", 
                          "debug.stage_1.jsonl", 
                          layout_model, mfd_model, ocrmodel=ocrmodel, 
                          inner_batch_size=inner_batch_size, batch_size=16,num_workers=4,
                          do_text_det = True,
                          do_text_rec = False,
                          timer=timer)
    
    
    