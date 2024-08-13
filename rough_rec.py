
from get_data_utils import *
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool
from functools import partial
import cv2
from torch.utils.data import Dataset, TensorDataset, DataLoader
from dataaccelerate import DataPrefetcher 
from modules.batch_text_rec import TextRecognizer, rec_args
import torch
from scihub_pdf_dataset import RecImageDataset,rec_collate_fn,deal_with_one_pdf,none_collate_fn
try:
    client=build_client()
except:
    client=None
   
import math

# def rec_preprocessing(text_recognizer, img_list):
#     norm_img_batch = []
    
#     resize_norm_img_func = partial(resize_norm_img,
#                                max_wh_ratio=max_wh_ratio,
#                                rec_image_shape  =text_recognizer.rec_image_shape,
#                                limited_max_width=text_recognizer.limited_max_width,
#                                limited_min_width=text_recognizer.limited_min_width)
#     for img_now in tqdm(img_list, desc="resize and normlized image"):
#         norm_img = resize_norm_img_func(img_now)
#         norm_img = norm_img[np.newaxis, :]
#         norm_img_batch.append(norm_img)
#     norm_img_batch = np.concatenate(norm_img_batch)
#     # norm_img_batch = norm_img_batch.copy()
#     return norm_img_batch

def resize_norm_img(img, max_wh_ratio=None,rec_image_shape=None,limited_max_width=None,limited_min_width=None):
    imgC, imgH, imgW = rec_image_shape
    assert imgC == img.shape[2]
    max_wh_ratio = max(max_wh_ratio, imgW / imgH)
    imgW = int((imgH * max_wh_ratio))
    imgW = max(min(imgW, limited_max_width), limited_min_width)
    h, w = img.shape[:2]
    ratio = w / float(h)
    ratio_imgH = math.ceil(imgH * ratio)
    ratio_imgH = max(ratio_imgH, limited_min_width)
    if ratio_imgH > imgW:
        resized_w = imgW
    else:
        resized_w = int(ratio_imgH)
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im

class UnifiedResizedDataset(Dataset):
    def __init__(self, img_list,rec_image_shape,limited_max_width,limited_min_width):
        max_wh_ratio = 0
        for img_now in img_list:
            # h, w = img_list[ino].shape[0:2]
            h, w = img_now.shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        self.max_wh_ratio = max_wh_ratio
        self.image_list   = img_list
        self.rec_image_shape =rec_image_shape
        self.limited_max_width =limited_max_width
        self.limited_min_width =limited_min_width
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        return resize_norm_img(self.image_list[idx], self.max_wh_ratio, self.rec_image_shape, self.limited_max_width, self.limited_min_width)

def postprocess(self,preds, label=None):
    preds_prob,preds_idx  = preds.max(axis=2)
    text = self.decode(preds_idx.cpu().numpy(), preds_prob.cpu().numpy(), is_remove_duplicate=True)

    if label is None:return text
    label = self.decode(label)
    return text, label

def gpu_inference(batch, tex_recognizer):
    inp = batch
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16): ### tested, fp16 only influence the result for last end sign like `.` or similar symbol like `0`` and `O`
            prob_out = tex_recognizer.net(inp)
    rec_result = postprocess(tex_recognizer.postprocess_op,prob_out)
    return rec_result


def calculate_dimensions(bbox):
        x_coords = bbox[::2]
        y_coords = bbox[1::2]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        return width, height


def build_bbox_group(metadatas):
    width_range = 100
    height_range= 100
    grouped_bboxes = {}
    location2group = {}
    location2boxes = {}
    for pdf_index, pdf_metadata in enumerate(tqdm(metadatas,desc="building group")):
        pdf_path = pdf_metadata['path']
        for pdf_page_metadata in pdf_metadata['doc_layout_result']:
            page_id = pdf_page_metadata['page_id']
            bbox_id = 0
            
            for bbox_metadata in pdf_page_metadata['layout_dets']:
                if bbox_metadata['category_id']!=15:continue
                location= (pdf_path,page_id,bbox_id)
                bbox_id+=1
                bbox = bbox_metadata['poly']
                width, height = calculate_dimensions(bbox)
                width_group   = int(width  // width_range)
                height_group  = int(height // height_range)
                group_key     = (width_group, height_group)
                if group_key not in grouped_bboxes:
                    grouped_bboxes[group_key] = []
                grouped_bboxes[group_key].append(location)
                location2group[location] = group_key
                location2boxes[location] = bbox
    return grouped_bboxes, location2group, location2boxes

from typing import List, Dict
def obtain_data_from_pool_list(pool_list, key):
    for pool in pool_list:
        if key in pool:
            return pool[key]
    return None


if __name__ == "__main__":
    from modules.self_modify import ModifiedPaddleOCR
    ocr_mode = 'batch'
    batch_size = 128
    num_workers= 8
    metadata_filepath = "part-66210c190659-012553.jsonl"
    images_dataset           = RecImageDataset(metadata_filepath)
    
    if ocr_mode == 'batch':
        # metadatas = read_json_from_path(metadata_filepath, client)
        _,location2group,location2boxes = build_bbox_group(images_dataset.metadata)
        # processes_num = min(8, len(metadatas))
        # with Pool(processes=processes_num) as pool:
        #     image_pool_list = list(tqdm(pool.imap(deal_with_one_pdf, metadatas), total=len(metadatas), desc="Reading whole text image into memory"))
        # image_pool_list = [deal_with_one_pdf(t) for t in tqdm(metadatas, desc="Reading whole text image into memory")]
        image_collecter   = DataLoader(images_dataset, batch_size=20,collate_fn=none_collate_fn, 
                            num_workers=8,pin_memory=False,
                            prefetch_factor=2)  
        tex_recognizer = TextRecognizer(rec_args)
        tex_recognizer.rec_batch_num = batch_size
        location_to_rec = {}
        for image_pool_list in tqdm(image_collecter,position=0,leave=True,desc="Images batch"):
            no_image_pdf_list = []
            image_pool = {}
            current_group_bboxes = {}
            for idx,(pdf_path, image_dict) in enumerate(image_pool_list):
                if len(image_dict)==0:
                    no_image_pdf_list.append(pdf_path)
                    #print(f"pdf {pdf_path} has no text image")
                    continue
                for key,val in image_dict.items():
                    image_pool[key]=val
                    group = location2group[key]
                    if group not in current_group_bboxes:
                        current_group_bboxes[group] = []
                    current_group_bboxes[group].append((key,location2boxes[key]))
            tqdm.write(f"we have {len(no_image_pdf_list)} pdfs has no text image and {len(image_pool)} text images")
            if len(image_pool) == 0:continue
            
            
            #### next step, lets do normlized the bbox to the same size

            
            pbar_whole_images  = tqdm(total=len(image_pool),position=1,leave=False,desc="Group batch")
            for group_key, location_and_bbox in current_group_bboxes.items():
                if len(location_and_bbox) == 0:continue
                
                img_list_group = [image_pool[location] for location, bbox in location_and_bbox]
                rec_list_group = []
                dataset  = UnifiedResizedDataset(img_list_group, tex_recognizer.rec_image_shape, tex_recognizer.limited_max_width, tex_recognizer.limited_min_width)
                dataloader_group = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, pin_memory_device='cuda')
                featcher   = DataPrefetcher(dataloader_group,device='cuda')
                pbar  = tqdm(total=len(dataloader_group),position=2,leave=False,desc="GPU batch")
                batch = featcher.next()
                while batch is not None:
                    rec_result = gpu_inference(batch, tex_recognizer)
                    rec_list_group.extend(rec_result)
                    pbar.update(1)
                    batch = featcher.next()
                assert len(location_and_bbox) == len(rec_list_group)
                for (location, bbox), rec_res in zip(location_and_bbox, rec_list_group):
                    location_to_rec[location] = rec_res
                    print(rec_res[0])
                raise
                pbar_whole_images.update(len(img_list_group))

        patch_metadata_list = []
        for pdf_index, pdf_metadata in enumerate(tqdm(images_dataset.metadata)):
            pdf_path = pdf_metadata['path']
            
            patch_metadata = {'path':pdf_path,'doc_layout_result':[]}
            for pdf_page_metadata in pdf_metadata['doc_layout_result']:
                page_id = pdf_page_metadata['page_id']
                bbox_id = 0
                this_line_pool = {'page_id':page_id, 'layout_dets':[]}
                for bbox_metadata in pdf_page_metadata['layout_dets']:
                    if bbox_metadata['category_id']!=15:continue
                    
                    location= (pdf_path,page_id,bbox_id)
                    bbox_id+=1
                    text, score = location_to_rec[location]
                    this_line_pool['layout_dets'].append({'category_id':15, 'text':text, 'score':score})
                patch_metadata['doc_layout_result'].append(this_line_pool)
            patch_metadata_list.append(patch_metadata)
    else:
        
        
        dataset           = RecImageDataset(metadata_filepath)
        image_collecter   = DataLoader(dataset, batch_size=8,collate_fn=rec_collate_fn, 
                                num_workers=num_workers,pin_memory=False, pin_memory_device='cuda',
                                prefetch_factor=2 if num_workers>0 else None)  
    
        ocr_model = ModifiedPaddleOCR(show_log=True)
        tex_recognizer=ocr_model.text_recognizer
        # tex_recognizer = TextRecognizer(rec_args)
        tex_recognizer.rec_batch_num = batch_size
        for location_abs_list, image_list in tqdm(image_collecter,position=0,leave=False,desc="Do Rec"):
            if len(image_list) == 0:continue
            tqdm.write(f"Now deal with B={len(image_list)}")
            rec_result = tex_recognizer(image_list)
            
    
    
    
    
    # #### next step, lets do normlized the bbox to the same size

    # location_to_rec = {}
    # pbar_whole_images  = tqdm(total=len(image_pool),position=1,leave=False)
    # for group_key, location_and_bbox in grouped_bboxes.items():
    #     if len(location_and_bbox) == 0:continue
        
    #     img_list_group = [image_pool[location] for location, bbox in location_and_bbox]
    #     rec_list_group = []
    #     dataset  = UnifiedResizedDataset(img_list_group, tex_recognizer.rec_image_shape, tex_recognizer.limited_max_width, tex_recognizer.limited_min_width)
    #     dataloader_group = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, pin_memory_device='cuda')
    #     featcher   = DataPrefetcher(dataloader_group,device='cuda')
    #     pbar  = tqdm(total=len(dataloader_group),position=2,leave=False)
    #     batch = featcher.next()
    #     while batch is not None:
    #         rec_result = gpu_inference(batch, tex_recognizer)
    #         rec_list_group.extend(rec_result)
    #         pbar.update(1)
    #         batch = featcher.next()
    #     assert len(location_and_bbox) == len(rec_list_group)
    #     for (location, bbox), rec_res in zip(location_and_bbox, rec_list_group):
    #         location_to_rec[location] = rec_res
    #     pbar_whole_images.update(len(img_list_group))

    # patch_metadata_list = []
    # for pdf_index, pdf_metadata in enumerate(tqdm(metadatas)):
    #     pdf_path = pdf_metadata['path']
        
    #     patch_metadata = {'path':pdf_path,'doc_layout_result':[]}
    #     for pdf_page_metadata in pdf_metadata['doc_layout_result']:
    #         page_id = pdf_page_metadata['page_id']
    #         bbox_id = 0
    #         this_line_pool = {'page_id':page_id, 'layout_dets':[]}
    #         for bbox_metadata in pdf_page_metadata['layout_dets']:
    #             if bbox_metadata['category_id']!=15:continue
                
    #             location= (pdf_path,page_id,bbox_id)
    #             bbox_id+=1
    #             text, score = location_to_rec[location]
    #             this_line_pool['layout_dets'].append({'category_id':15, 'text':text, 'score':score})
    #         patch_metadata['doc_layout_result'].append(this_line_pool)
    #     patch_metadata_list.append(patch_metadata)
    
    # write_json_to_path(patch_metadata_list, metadata_filepath.replace('.jsonl','.patch.rec_result.jsonl'), client)

    # deal_with_one_dataset("debug.jsonl", 
    #                       "debug.stage_1.jsonl", 
    #                       layout_model, mfd_model, ocrmodel=ocrmodel, 
    #                       inner_batch_size=2, batch_size=4,num_workers=4,
    #                       do_text_det = True,
    #                       do_text_rec = True,
    #                       timer=timer)
    # dataset    = PDFImageDataset("part-66210c190659-000035.jsonl",layout_model.predictor.aug,layout_model.predictor.input_format,mfd_pre_transform=None)
    # dataloader = DataLoader(dataset, batch_size=8,collate_fn=custom_collate_fn)  

    
    