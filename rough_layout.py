

import os 
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import warnings
# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from get_batch_yolo import mfd_process, get_batch_YOLO_model
from get_batch_layout_model import get_layout_model
from modules.no_paddle_ocr import ModifiedPaddleOCR
from utils import *
from utils import Timers
import numpy as np
import torch
import numpy as np
from tqdm.auto import tqdm
import yaml
from dataaccelerate import DataPrefetcher 
from ultralytics.utils import ops
import copy
from get_data_utils import *
import traceback
import logging

from scihub_pdf_dataset import PDFImageDataset, PageInfoDataset, custom_collate_fn,DataLoader

def clean_layout_dets(layout_dets):
    rows = []
    for t in layout_dets:
        rows.append({
        "category_id":int(t['category_id']),
        "poly":[int(t) for t in t['poly']],
        "score":float(t['score'])
        })
        
    return rows

def inference_layout(layout_pair,layout_model,inner_batch_size):
    
    layout_images, heights, widths = layout_pair
    origin_length = len(layout_images)
    
    if len(layout_images)<inner_batch_size and layout_model.iscompiled:
        layout_images  = torch.nn.functional.pad(layout_images,   (0,0,0,0,0,0,0, inner_batch_size-len(layout_images)))
        heights = torch.nn.functional.pad(heights,  (0, inner_batch_size-len(heights)))
        widths  = torch.nn.functional.pad(widths,   (0, inner_batch_size-len(widths)))
    layout_res = layout_model((layout_images,heights, widths), ignore_catids=[],dtype=torch.float16)
    layout_res = layout_res[:origin_length]
    return layout_res

def inference_mfd(mfd_images,mfd_model,inner_batch_size):
    origin_length = len(mfd_images)
    
    if len(mfd_images)<inner_batch_size:
        mfd_images = torch.nn.functional.pad(mfd_images, (0,0,0,0,0,0,0, inner_batch_size-len(mfd_images)))
    mfd_res    = mfd_model.predict(mfd_images, imgsz=(1888,1472), conf=0.3, iou=0.5, verbose=False)
    mfd_res = mfd_res[:origin_length]
    return mfd_res

def combine_layout_mfd_result(layout_res, mfd_res, heights, widths):
    rough_layout_this_batch =[]
    ori_shape_list = []
    for layout_det, mfd_det, real_input_height, real_input_width in zip(layout_res, mfd_res, heights, widths):
        mfd_height,mfd_width = mfd_det.orig_shape
        real_input_height = int(real_input_height)
        real_input_width  = int(real_input_width)
        layout_dets = clean_layout_dets(layout_det['layout_dets'])
        for xyxy, conf, cla in zip(mfd_det.boxes.xyxy.cpu(), 
                                mfd_det.boxes.conf.cpu(), 
                                mfd_det.boxes.cls.cpu()):

            xyxy =  ops.scale_boxes((mfd_height,mfd_width), xyxy, (real_input_height, real_input_width))
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
            new_item = {
                'category_id': 13 + int(cla.item()),
                'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                'score': round(float(conf.item()), 2),
                'latex': '',
            }
            layout_dets.append(new_item)
        ori_shape_list.append((real_input_height, real_input_width))
        rough_layout_this_batch.append(layout_dets)
        assert real_input_height == 1920
        assert real_input_width  == 1472
    return rough_layout_this_batch, ori_shape_list

def inference_det(canvas_tensor_this_batch,det_model,det_inner_batch_size):
    with torch.no_grad():
        ### do inner_batch_size batch forward
        dt_boxaes_batch_list = []
        for i in range(0, len(canvas_tensor_this_batch), det_inner_batch_size):
            dt_boxaes_batch = det_model(canvas_tensor_this_batch[i:i+det_inner_batch_size])
            dt_boxaes_batch = dt_boxaes_batch['maps'].cpu()[:,0]
            dt_boxaes_batch_list.append(dt_boxaes_batch) 
        dt_boxaes_batch_list= torch.cat(dt_boxaes_batch_list)
    return dt_boxaes_batch_list

def det_postprocess(dt_boxaes_batch,ocrmodel):
    dt_boxaes_batch=dt_boxaes_batch.numpy()
    post_result = ocrmodel.batch_det_model.fast_postprocess(dt_boxaes_batch, np.array([(1920,1472, 0.5, 0.5)]*len(dt_boxaes_batch)) )
    # print(post_result[0]['points'].shape)
    # raise
    dt_boxes_list = [ocrmodel.batch_det_model.filter_tag_det_res(dt_boxes['points'][0], (1920,1472)) for dt_boxes in post_result]
    return dt_boxes_list


def deal_with_one_dataset(pdf_path, result_path, layout_model, mfd_model, 
                          ocrmodel=None, inner_batch_size=4, 
                          batch_size=32,num_workers=8,
                          do_text_det=False,
                          do_text_rec=False,
                          timer=Timers(False),
                          partion_num = 1,
                          partion_idx = 0):
    dataset    = PDFImageDataset(pdf_path,layout_model.predictor.aug,layout_model.predictor.input_format,
                                 
                                 mfd_pre_transform=mfd_process(mfd_model.predictor.args.imgsz,mfd_model.predictor.model.stride,mfd_model.predictor.model.pt),
                                 det_pre_transform=ocrmodel.batch_det_model.prepare_image,
                                 return_original_image=do_text_rec,
                                 partion_num = partion_num,
                                 partion_idx = partion_idx
                                 )
    collate_fn = custom_collate_fn if do_text_rec else None
    num_workers=min(num_workers,len(dataset.metadata))
    dataloader = DataLoader(dataset, batch_size=batch_size,collate_fn=collate_fn, 
                            num_workers=num_workers,pin_memory=True, pin_memory_device='cuda',
                            prefetch_factor=3 if num_workers>0 else None)        
    
    data_to_save = {}
    inner_batch_size = inner_batch_size
    #pbar  = tqdm(total=len(dataset.metadata),position=2,desc="PDF Pages",leave=False)
    pbar  = None
    pdf_passed = set()
    featcher   = DataPrefetcher(dataloader,device='cuda')
    batch = featcher.next()
    data_loading = []
    model_train  = []
    last_record_time = time.time()
    while batch is not None:
    #for batch in dataloader:
        try:
            data_loading.append(time.time() - last_record_time);last_record_time =time.time() 
            if pbar:pbar.set_description(f"[Data][{np.mean(data_loading[-10:]):.2f}] [Model][{np.mean(model_train[-10:]):.2f}]")
            pdf_index_batch, page_ids_batch = batch["pdf_index"], batch["page_index"]
            mfd_layout_images_batch, layout_images_batch, det_layout_images_batch = batch["mfd_image"], batch["layout_image"], batch["det_images"]
            heights_batch, widths_batch = batch["height"], batch["width"]
            oimage_list = batch.get('oimage',None)
            pdf_index = set([t.item() for t in pdf_index_batch])
            new_pdf_processed = pdf_index - pdf_passed
            pdf_passed        = pdf_passed|pdf_index
            
            iterater = tqdm(range(0, len(mfd_layout_images_batch), inner_batch_size),position=3,leave=False,desc="mini-Batch") if len(mfd_layout_images_batch)>inner_batch_size else range(0, len(mfd_layout_images_batch), inner_batch_size)

            for j in iterater:
                pdf_index  = pdf_index_batch[j:j+inner_batch_size]
                page_ids   = page_ids_batch[j:j+inner_batch_size]
                mfd_images = mfd_layout_images_batch[j:j+inner_batch_size]
                layout_images = layout_images_batch[j:j+inner_batch_size]
                heights    = heights_batch[j:j+inner_batch_size]
                widths     = widths_batch[j:j+inner_batch_size]
                oimages    = oimage_list[j:j+inner_batch_size] if oimage_list is not None else None
                detimages  = det_layout_images_batch[j:j+inner_batch_size]
                pdf_paths  = [dataset.metadata[pdf_index]['path'] for pdf_index in pdf_index]
                with timer('get_layout'):
                    layout_res = inference_layout((layout_images,heights, widths),layout_model,inner_batch_size)
                with timer('get_mfd'):
                    mfd_res    = inference_mfd(mfd_images,mfd_model,inner_batch_size)
                with timer('combine_layout_mfd_result'):
                    rough_layout_this_batch, ori_shape_list = combine_layout_mfd_result(layout_res, mfd_res, heights, widths)
                pdf_and_page_id_this_batch=[]
                for pdf_path, page_id, layout_dets,ori_shape in zip(pdf_paths, page_ids, rough_layout_this_batch,ori_shape_list):
                    page_id = int(page_id)
                    if pdf_path not in data_to_save:
                        data_to_save[pdf_path] = {'height':ori_shape[0], 'width':ori_shape[1]}
                    data_to_save[pdf_path][page_id] = layout_dets
                    pdf_and_page_id_this_batch.append((pdf_path, page_id))
                
                
                if ocrmodel is not None:
                    if not do_text_det:continue
                    with timer('text_detection/collect_for_line_detect'):
                        det_height, det_width = detimages.shape[2:]
                        scale_height = int(heights[0])/int(det_height)
                        scale_width  = int(widths[0])/int(det_width)
                        assert scale_height == scale_width
                        assert scale_height == 2
                        canvas_tensor_this_batch, partition_per_batch,_,_ = collect_paragraph_image_and_its_coordinate(detimages, rough_layout_this_batch,scale_height) # 2 is the scale between detiamge and box_images
                    if len(canvas_tensor_this_batch)==0:
                        tqdm.write("WARNING: no text line to detect")
                        continue
                    with timer('text_detection/stack'):
                        canvas_tensor_this_batch = torch.stack(canvas_tensor_this_batch)
                    with timer('text_detection/det_net'):
                        dt_boxaes_batch = inference_det(canvas_tensor_this_batch,ocrmodel.batch_det_model.net,128)
                    with timer('text_detection/det_postprocess'):
                        dt_boxes_list = det_postprocess(dt_boxaes_batch,ocrmodel)
                    
                    
                    if do_text_rec:
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
                            pdf_path, page_id = pdf_and_page_id_this_batch[partition_id]
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
            

        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            print("ERROR: Fail to process batch")
        update_seq = len(new_pdf_processed)
        if pbar:pbar.update(update_seq)
        timer.log()
        model_train.append(time.time() - last_record_time);last_record_time =time.time()
        if pbar:pbar.set_description(f"[Data][{np.mean(data_loading[-10:]):.2f}] [Model][{np.mean(model_train[-10:]):.2f}]")
        batch = featcher.next()
        if pbar is None:
            pbar = tqdm(total=len(dataset.metadata)-update_seq,position=2,desc="PDF Pages",leave=False, bar_format='{l_bar}{bar}{r_bar}')

    ### next, we construct each result for each pdf in pdf wise and remove the page_id by the list position 
    save_result(data_to_save,dataset,result_path)


def deal_with_page_info_dataset(pdf_path, result_path, layout_model, mfd_model, 
                          ocrmodel=None, inner_batch_size=4, 
                          batch_size=32,num_workers=8,
                          do_text_det=False,
                          do_text_rec=False,
                          timer=Timers(False),
                          partion_num = 1,
                          partion_idx = 0,page_num_for_name=None):
    dataset    = PageInfoDataset(pdf_path,layout_model.predictor.aug,layout_model.predictor.input_format,
                                 
                                 mfd_pre_transform=mfd_process(mfd_model.predictor.args.imgsz,mfd_model.predictor.model.stride,mfd_model.predictor.model.pt),
                                 det_pre_transform=ocrmodel.batch_det_model.prepare_image,
                                 return_original_image=do_text_rec,
                                 partion_num = partion_num,
                                 partion_idx = partion_idx,page_num_for_name=page_num_for_name
                                 )
    print(f"current dataset size={len(dataset)} images")
    collate_fn = custom_collate_fn if do_text_rec else None
    num_workers=min(num_workers,len(dataset.metadata))
    dataloader = DataLoader(dataset, batch_size=batch_size,collate_fn=collate_fn, 
                            num_workers=num_workers,pin_memory=True, pin_memory_device='cuda',
                            prefetch_factor=2 if num_workers>0 else None)        
    
    data_to_save = {}
    inner_batch_size = inner_batch_size
    #pbar  = tqdm(total=len(dataset.metadata),position=2,desc="PDF Pages",leave=False)
    pbar  = None
    pdf_passed = set()
    featcher   = DataPrefetcher(dataloader,device='cuda')
    batch = featcher.next()
    data_loading = []
    model_train  = []
    last_record_time = time.time()
    while batch is not None:
        data_loading.append(time.time() - last_record_time);last_record_time =time.time() 
        if pbar:pbar.set_description(f"[Data][{np.mean(data_loading[-10:]):.2f}] [Model][{np.mean(model_train[-10:]):.2f}]")
        pdf_index_batch, page_ids_batch = batch["pdf_index"], batch["page_index"]
        mfd_layout_images_batch, layout_images_batch, det_layout_images_batch = batch["mfd_image"], batch["layout_image"], batch["det_images"]
        heights_batch, widths_batch = batch["height"], batch["width"]
        oimage_list = batch.get('oimage',None)
        pdf_index = set([t.item() for t in pdf_index_batch])

        
        iterater = tqdm(range(0, len(mfd_layout_images_batch), inner_batch_size),position=3,leave=False,desc="mini-Batch") if len(mfd_layout_images_batch)>inner_batch_size else range(0, len(mfd_layout_images_batch), inner_batch_size)

        for j in iterater:
            pdf_index  = pdf_index_batch[j:j+inner_batch_size]
            page_ids   = page_ids_batch[j:j+inner_batch_size]
            mfd_images = mfd_layout_images_batch[j:j+inner_batch_size]
            layout_images = layout_images_batch[j:j+inner_batch_size]
            heights    = heights_batch[j:j+inner_batch_size]
            widths     = widths_batch[j:j+inner_batch_size]
            oimages    = oimage_list[j:j+inner_batch_size] if oimage_list is not None else None
            detimages  = det_layout_images_batch[j:j+inner_batch_size]
            pdf_paths  = [dataset.metadata[pdf_index]['path'] for pdf_index in pdf_index]
            with timer('get_layout'):
                layout_res = inference_layout((layout_images,heights, widths),layout_model,inner_batch_size)
            with timer('get_mfd'):
                mfd_res    = inference_mfd(mfd_images,mfd_model,inner_batch_size)
            with timer('combine_layout_mfd_result'):
                rough_layout_this_batch, ori_shape_list = combine_layout_mfd_result(layout_res, mfd_res, heights, widths)
            pdf_and_page_id_this_batch=[]
            for pdf_path, page_id, layout_dets,ori_shape in zip(pdf_paths, page_ids, rough_layout_this_batch,ori_shape_list):
                page_id = int(page_id)
                if pdf_path not in data_to_save:
                    data_to_save[pdf_path] = {'height':ori_shape[0], 'width':ori_shape[1]}
                data_to_save[pdf_path][page_id] = layout_dets
                pdf_and_page_id_this_batch.append((pdf_path, page_id))
            
            
            if ocrmodel is not None:
                if not do_text_det:continue
                with timer('text_detection/collect_for_line_detect'):
                    det_height, det_width = detimages.shape[2:]
                    scale_height = int(heights[0])/int(det_height)
                    scale_width  = int(widths[0])/int(det_width)
                    assert scale_height == scale_width
                    assert scale_height == 2
                    canvas_tensor_this_batch, partition_per_batch,_,_ = collect_paragraph_image_and_its_coordinate(detimages, rough_layout_this_batch,scale_height) # 2 is the scale between detiamge and box_images
                if len(canvas_tensor_this_batch)==0:
                    tqdm.write("WARNING: no text line to detect")
                    continue
                with timer('text_detection/stack'):
                    canvas_tensor_this_batch = torch.stack(canvas_tensor_this_batch)
                with timer('text_detection/det_net'):
                    dt_boxaes_batch = inference_det(canvas_tensor_this_batch,ocrmodel.batch_det_model.net,128)
                with timer('text_detection/det_postprocess'):
                    dt_boxes_list = det_postprocess(dt_boxaes_batch,ocrmodel)
                
                
                if do_text_rec:
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
                        pdf_path, page_id = pdf_and_page_id_this_batch[partition_id]
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
            

        # except KeyboardInterrupt:
        #     raise
        # except:
        #     traceback.print_exc()
        #     print("ERROR: Fail to process batch")
        update_seq = len(page_ids_batch)
        if pbar:pbar.update(update_seq)
        timer.log()
        model_train.append(time.time() - last_record_time);last_record_time =time.time()
        if pbar:pbar.set_description(f"[Data][{np.mean(data_loading[-10:]):.2f}] [Model][{np.mean(model_train[-10:]):.2f}]")
        batch = featcher.next()
        if pbar is None:
            pbar = tqdm(total=len(dataset),position=2,desc="PDF Pages",leave=False, bar_format='{l_bar}{bar}{r_bar}')

    ### next, we construct each result for each pdf in pdf wise and remove the page_id by the list position 
    save_result(data_to_save,dataset,result_path)
    

def save_result(data_to_save,dataset,result_path):
    pdf_to_metadata = {t['path']:t for t in dataset.metadata}
 
    new_data_to_save = []
    for pdf_path, layout_dets_per_page in data_to_save.items():
        
        new_pdf_dict = copy.deepcopy(pdf_to_metadata[pdf_path])
        new_pdf_dict['height'] = layout_dets_per_page.pop('height')
        new_pdf_dict['width'] = layout_dets_per_page.pop('width')
        pages = [t for t in layout_dets_per_page.keys()]
        pages.sort()
        #print(pages)
 
        new_pdf_dict["doc_layout_result"]=[]
        for page_id in range(max(pages)+1): ### those , before, we may lost whole the last page for layoutV1-5 result
            if page_id not in layout_dets_per_page:
                print(f"WARNING: page {page_id} of PDF: {pdf_path} fail to parser!!! ")
                now_row = {"page_id": page_id, "status": "fail", "layout_dets":[]}
            else:
                now_row = {"page_id": page_id, "layout_dets":layout_dets_per_page[page_id]}
            new_pdf_dict["doc_layout_result"].append(now_row)
        new_data_to_save.append(new_pdf_dict)
    if "s3:" in new_data_to_save and dataset.client is None:dataset.client=build_client()

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

if __name__ == "__main__":
    
    with open('configs/model_configs.yaml') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)

    img_size  = model_configs['model_args']['img_size']
    conf_thres= model_configs['model_args']['conf_thres']
    iou_thres = model_configs['model_args']['iou_thres']
    device    = model_configs['model_args']['device']
    dpi       = model_configs['model_args']['pdf_dpi']

    accelerated = True
    layout_model = get_layout_model(model_configs,accelerated)
    
    total_memory = get_gpu_memory()
    if total_memory > 60:
        inner_batch_size = 16
    elif total_memory > 30:
        inner_batch_size = 8
    else:
        inner_batch_size = 2
    print(f"totally gpu memory is {total_memory} we use inner batch size {inner_batch_size}")
    mfd_model    = get_batch_YOLO_model(model_configs,inner_batch_size) 
    ocrmodel = None
    ocrmodel = ocr_model = ModifiedPaddleOCR(show_log=True)
    timer = Timers(False,warmup=5)
    #test_dataset("debug.jsonl", layout_model, mfd_model, ocrmodel)
    #page_num_map_whole = get_page_num_map_whole()
    page_num_map_whole = None
    deal_with_page_info_dataset("part-66210c190659-000026.jsonl", 
                                "part-66210c190659-000026.jsonl.stage_1.jsonl", 
                                layout_model, mfd_model, ocrmodel=ocrmodel, 
                                inner_batch_size=inner_batch_size, batch_size=inner_batch_size,num_workers=8,
                                do_text_det = True,
                                do_text_rec = False,
                                timer=timer,page_num_for_name=page_num_map_whole)
    
    
    