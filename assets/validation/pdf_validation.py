# coding: utf-8
import json
import os
import os.path as osp
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from mmeval import COCODetection
import cv2


# category_id
class_dict = {
 0: 'title',              # Title
 1: 'plain text',         # Text
 2: 'abandon',            # Includes headers, footers, page numbers, and page annotations
 3: 'figure',             # Image
 4: 'figure_caption',     # Image caption
 5: 'table',              # Table
 6: 'table_caption',      # Table caption
 7: 'table_footnote',     # Table footnote
 8: 'isolate_formula',    # Display formula (this is a layout display formula, lower priority than 14)
 9: 'formula_caption',    # Display formula label

 13: 'inline_formula',    # Inline formula
 14: 'isolated_formula',  # Display formula
 15: 'ocr_text'}          # OCR result


def reformat_gt_and_pred(labels, det_res, label_classes):
    preds = []
    gts = []
    
    for idx, (ann, pred) in enumerate(zip(labels, det_res)):
        gt_bboxes = []
        gt_labels = []
        for item in ann['layout_dets']:
            class_name = class_dict.get(item['category_id'])
            if not class_name:
                class_name = 'unknown'

            if class_name in label_classes:
                L = item['poly'][0]
                U = item['poly'][1]
                R = item['poly'][4]
                D = item['poly'][5]
                L, R = min(L, R), max(L, R)
                U, D = min(U, D), max(U, D)
                gt_bboxes.append([L, U, R, D])
                gt_labels.append(label_classes.index(class_name))
        
        gts.append({
            'img_id': idx,
            'width': ann['page_info']['width'],
            'height': ann['page_info']['height'],
            'bboxes': np.array(gt_bboxes),
            'labels': np.array(gt_labels),
            'ignore_flags': [False]*len(gt_labels),
        })
        
        bboxes = []
        labels = []
        scores = []

        for item in pred['layout_dets']:
            class_name = class_dict.get(item['category_id'])
            if not class_name:
                class_name = 'unknown'

            if class_name in label_classes:
                L = item['poly'][0]
                U = item['poly'][1]
                R = item['poly'][4]
                D = item['poly'][5]
                L, R = min(L, R), max(L, R)
                U, D = min(U, D), max(U, D)
                bboxes.append([L, U, R, D])
                labels.append(label_classes.index(class_name))
                scores.append(item['score'])
        
        preds.append({
            'img_id': idx,
            'bboxes': np.array(bboxes),
            'scores': np.array(scores),
            'labels': np.array(labels),
        })
    
    return gts, preds


def validation(gt_path, pred_path, label_classes):
    labels = []
    det_res = []
    pred_dict = {}

    for single_pdf_result in os.listdir(pred_path):
        
        with open(os.path.join(pred_path, single_pdf_result), 'r') as f:
            preds_sample = json.load(f)

        basename = single_pdf_result[:-5]
        for pred in preds_sample:
            page_num = pred['page_info']['page_no']
            pred_dict[f'{basename}_{page_num}'] = pred

    for single_pdf_gt in os.listdir(gt_path):
        
        with open(os.path.join(gt_path, single_pdf_gt), 'r') as f:
            gt_sample = json.load(f)

        basename = single_pdf_gt[:-5]
        for gt in gt_sample:
            page_num = gt['page_info']['page_no']
            sample_name = f'{basename}_{page_num}'
            if pred_dict.get(sample_name):
                det_res.append(pred_dict[sample_name])
                labels.append(gt)
            else:
                print(f'No matching prediction for {sample_name}.')

            
    meta={'CLASSES':tuple(label_classes)}
    coco_det_metric = COCODetection(dataset_meta=meta, metric=['bbox'], classwise=True)

    gts, preds = reformat_gt_and_pred(labels, det_res, label_classes)

  
    detect_matrix = coco_det_metric(predictions=preds, groundtruths=gts)
 
    print('detect_matrix', detect_matrix)

    
if __name__ == "__main__":
    """
        gt_path: the folder of ground truth
        pred_path: the folder of prediction
        label_classes: The classes to validate
    """
    gt_path = "./gt"
    pred_path = "./pred"
    label_classes = ["title", "plain text", "abandon", "figure", "figure_caption", "table"]

    validation(gt_path, pred_path, label_classes)
    print("=> process done!")
