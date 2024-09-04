
import os,sys,warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from get_data_utils import *
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset, TensorDataset, DataLoader
from dataaccelerate import DataPrefetcher ,sendall2gpu, DataSimfetcher
import torch
from task_layout.rough_layout import ModifiedPaddleOCR,inference_det,det_postprocess,save_result
from scihub_pdf_dataset import DetPageInfoImageDataset, DetImageDataset,concat_collate_fn,tuple_list_collate_fn
from utils import collect_paragraph_image_and_its_coordinate
try:
    client=build_client()
except:
    client=None
eps=1e-7
import math


from typing import List, Dict
import time
def deal_with_one_dataset(pdf_path, result_path, text_detector,det_pre_transform,
                          pdf_batch_size  =32,
                          image_batch_size=256,
                          num_workers=8,
                          partion_num = 1,
                          partion_idx = 0):

    images_dataset = DetPageInfoImageDataset(pdf_path,det_pre_transform,partion_num = partion_num, partion_idx = partion_idx)
    data_to_save =  fast_deal_with_one_dataset(images_dataset,text_detector, pdf_batch_size  =pdf_batch_size, 
                                                image_batch_size=image_batch_size,num_workers=num_workers)
    save_result(data_to_save, images_dataset,result_path)


class DetDataPrefetcher(DataPrefetcher):
    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            tensor, information = self.batch
            if tensor is not None:
                self.batch = (sendall2gpu(tensor,self.device), information)
            else:
                self.batch = None, information

def gpu_inference(canvas_tensor_this_batch,text_detector,det_inner_batch_size,oriheight,oriwidth):
    det_model = text_detector.batch_det_model.net
    with torch.inference_mode():
        ### do inner_batch_size batch forward
        dt_boxes_result_list = []
        for i in tqdm(range(0, len(canvas_tensor_this_batch), det_inner_batch_size), position=3, desc="inner_batch", leave=False):
            data = canvas_tensor_this_batch[i:i+det_inner_batch_size]
            if isinstance(data, list): data = torch.stack(data)
            data=data.cuda()
            dt_boxaes_batch = det_model(data)
            dt_boxaes_batch = dt_boxaes_batch['maps'].cpu()[:,0]
            dt_boxes_result   = det_postprocess(dt_boxaes_batch,text_detector,oriheight,oriwidth)
            dt_boxes_result_list.extend(dt_boxes_result) 
        return dt_boxes_result_list
    

def fast_deal_with_one_dataset(images_dataset:DetImageDataset,text_detector:ModifiedPaddleOCR,
                          pdf_batch_size  =32,
                          image_batch_size=256,
                          num_workers=8):
    
    dataloader = DataLoader(images_dataset, batch_size=pdf_batch_size,collate_fn=tuple_list_collate_fn, 
                            num_workers=num_workers,pin_memory=False,
                            prefetch_factor=2)  
    featcher   = DetDataPrefetcher(dataloader,device='cuda')
    batch = featcher.next()
    data_loading = []
    model_train  = []
    last_record_time = time.time()
    data_to_save = {}
    pbar  = None
    oriheight = 1920 # used for correct giving boxing and postprocessing
    oriwidth  = 1472 # used for correct giving boxing and postprocessing
    while batch is not None:
    #for batch in dataloader:
        ########## format data ################
        detimages, rough_layout_this_batch = batch
        if detimages is not None and len(detimages)>0:
            canvas_tensor_this_batch, partition_per_batch,_,_ = collect_paragraph_image_and_its_coordinate(detimages, rough_layout_this_batch,2)
            location = []
            for global_page_id in range(len(partition_per_batch)-1):
                start_id = partition_per_batch[global_page_id]
                end_id   = partition_per_batch[global_page_id+1]
                pdf_path = rough_layout_this_batch[global_page_id][0]['pdf_path']
                page_id  = rough_layout_this_batch[global_page_id][0]['page_id']
                for image_id in range(0, end_id-start_id):
                    location.append((pdf_path, page_id, image_id))
            assert len(location) == len(canvas_tensor_this_batch)
            data_loading.append(time.time() - last_record_time);last_record_time =time.time() 
            ########## format computing ################
            dt_boxes_list   = gpu_inference(canvas_tensor_this_batch,text_detector,image_batch_size,oriheight,oriwidth)
            if pbar:pbar.set_description(f"[Data][{np.mean(data_loading[-10:]):.2f}] [Model][{np.mean(model_train[-10:]):.2f}]")
            for dt_boxes, (pdf_path, page_id, line_image_id) in zip(dt_boxes_list, location):
                page_id= int(page_id)
                line_image_id=int(line_image_id)
                for line_box in dt_boxes:
                    p1, p2, p3, p4 = line_box.tolist()
                    if pdf_path not in data_to_save:
                        data_to_save[pdf_path] = {'height':oriheight, 'width':oriwidth}
                    if page_id not in data_to_save[pdf_path]:
                        data_to_save[pdf_path][page_id] = []
                    data_to_save[pdf_path][page_id].append(
                        {
                            'category_id': 15,
                            'poly': p1 + p2 + p3 + p4,
                        }
                    )
            model_train.append(time.time() - last_record_time)
        last_record_time =time.time()
        update_seq_len = len(detimages)
        if pbar:
            pbar.update(update_seq_len)
            pbar.set_description(f"[Data][{np.mean(data_loading[-10:]):.2f}] [Model][{np.mean(model_train[-10:]):.2f}]")
        if pbar is None:
            pbar = tqdm(total=len(images_dataset)-update_seq_len,position=2,desc="pages",leave=False, bar_format='{l_bar}{bar}{r_bar}')
        batch = featcher.next()
    return data_to_save

if __name__ == "__main__":
    
    ocr_mode = 'batch'
    batch_size = 128
    num_workers= 8
    metadata_filepath = "part-66210c190659-012745.jsonl"
    text_detector     = ModifiedPaddleOCR()
    images_dataset    = DetPageInfoImageDataset(metadata_filepath,det_pre_transform=text_detector.batch_det_model.prepare_image)
    data_to_save      = fast_deal_with_one_dataset(images_dataset,text_detector,pdf_batch_size=32, image_batch_size=128 ,num_workers=num_workers)
    #print(data_to_save)
    save_result(data_to_save, images_dataset, "test_result/result.det.jsonl")
