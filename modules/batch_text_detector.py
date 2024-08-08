import os
import sys
import copy
import cv2
import numpy as np
import time
import json
import torch
from shapely.geometry import Polygon
import pyclipper

from .pytorchocr.base_ocr_v20 import BaseOCRV20
from .pytorchocr.utils.utility import get_image_file_list, check_and_read_gif
from .pytorchocr.data import create_operators, transform
from .pytorchocr.postprocess import build_post_process
from .pytorchocr import pytorchocr_utility as utility
from dataclasses import dataclass 


class TextDetector(BaseOCRV20):
    def __init__(self, args, **kwargs):
        self.args = args
        self.det_algorithm = args.det_algorithm
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': args.det_limit_side_len,
                'limit_type': args.det_limit_type,
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {}
        if self.det_algorithm == "DB":
            postprocess_params['name'] = 'DBPostProcess'
            postprocess_params["thresh"] = args.det_db_thresh
            postprocess_params["box_thresh"] = args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
            postprocess_params["use_dilation"] = args.use_dilation
            postprocess_params["score_mode"] = args.det_db_score_mode
        elif self.det_algorithm == "DB++":
            postprocess_params['name'] = 'DBPostProcess'
            postprocess_params["thresh"] = args.det_db_thresh
            postprocess_params["box_thresh"] = args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
            postprocess_params["use_dilation"] = args.use_dilation
            postprocess_params["score_mode"] = args.det_db_score_mode
            pre_process_list[1] = {
                'NormalizeImage': {
                    'std': [1.0, 1.0, 1.0],
                    'mean':
                        [0.48109378172549, 0.45752457890196, 0.40787054090196],
                    'scale': '1./255.',
                    'order': 'hwc'
                }
            }
        elif self.det_algorithm == "EAST":
            postprocess_params['name'] = 'EASTPostProcess'
            postprocess_params["score_thresh"] = args.det_east_score_thresh
            postprocess_params["cover_thresh"] = args.det_east_cover_thresh
            postprocess_params["nms_thresh"] = args.det_east_nms_thresh
        elif self.det_algorithm == "SAST":
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'resize_long': args.det_limit_side_len
                }
            }
            postprocess_params['name'] = 'SASTPostProcess'
            postprocess_params["score_thresh"] = args.det_sast_score_thresh
            postprocess_params["nms_thresh"] = args.det_sast_nms_thresh
            self.det_sast_polygon = args.det_sast_polygon
            if self.det_sast_polygon:
                postprocess_params["sample_pts_num"] = 6
                postprocess_params["expand_scale"] = 1.2
                postprocess_params["shrink_ratio_of_width"] = 0.2
            else:
                postprocess_params["sample_pts_num"] = 2
                postprocess_params["expand_scale"] = 1.0
                postprocess_params["shrink_ratio_of_width"] = 0.3
        elif self.det_algorithm == "PSE":
            postprocess_params['name'] = 'PSEPostProcess'
            postprocess_params["thresh"] = args.det_pse_thresh
            postprocess_params["box_thresh"] = args.det_pse_box_thresh
            postprocess_params["min_area"] = args.det_pse_min_area
            postprocess_params["box_type"] = args.det_pse_box_type
            postprocess_params["scale"] = args.det_pse_scale
            self.det_pse_box_type = args.det_pse_box_type
        elif self.det_algorithm == "FCE":
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'rescale_img': [1080, 736]
                }
            }
            postprocess_params['name'] = 'FCEPostProcess'
            postprocess_params["scales"] = args.scales
            postprocess_params["alpha"] = args.alpha
            postprocess_params["beta"] = args.beta
            postprocess_params["fourier_degree"] = args.fourier_degree
            postprocess_params["box_type"] = args.det_fce_box_type
        else:
            print("unknown det_algorithm:{}".format(self.det_algorithm))
            sys.exit(0)

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)

        use_gpu = args.use_gpu
        self.use_gpu = torch.cuda.is_available() and use_gpu

        self.weights_path = args.det_model_path
        self.yaml_path = args.det_yaml_path
        network_config = utility.AnalysisConfig(self.weights_path, self.yaml_path)
        super(TextDetector, self).__init__(network_config, **kwargs)
        self.load_pytorch_weights(self.weights_path)
        self.net.eval()
        if self.use_gpu:
            self.net.cuda()

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def clip_det_res_batch(self, points, img_height, img_width):
        # Clip the points to the image borders
        points[:, :, 0] = np.clip(points[:, :, 0], 0, img_width - 1)
        points[:, :, 1] = np.clip(points[:, :, 1], 0, img_height - 1)
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def order_points_clockwise_batch(self, pts_batch):
        """
        Orders a batch of points in a clockwise manner.
        
        Args:
            pts_batch (numpy.ndarray): Array of shape (N, 4, 2) containing N sets of four points.
            
        Returns:
            numpy.ndarray: Array of shape (N, 4, 2) with points ordered as top-left, top-right, 
                           bottom-right, bottom-left.
        """
        # Sort points in each set by x-coordinates
        xSorted = np.sort(pts_batch, axis=1, order=['x'])

        # Separate left-most and right-most points
        leftMost = xSorted[:, :2, :]
        rightMost = xSorted[:, 2:, :]

        # Sort left-most points by y-coordinates
        leftMost = leftMost[np.argsort(leftMost[:, :, 1], axis=1)]
        tl = leftMost[:, 0, :]
        bl = leftMost[:, 1, :]

        # Sort right-most points by y-coordinates
        rightMost = rightMost[np.argsort(rightMost[:, :, 1], axis=1)]
        tr = rightMost[:, 0, :]
        br = rightMost[:, 1, :]

        # Combine the points into the ordered rectangle
        rect = np.stack((tl, tr, br, bl), axis=1)
        return rect

    def filter_tag_det_res_new(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        
        # Order points clockwise and clip them
        ordered_boxes = self.order_points_clockwise_batch(dt_boxes)
        clipped_boxes = self.clip_det_res_batch(ordered_boxes,img_height, img_width)
        
        # Calculate widths and heights
        widths  = np.linalg.norm(clipped_boxes[:, 0] - clipped_boxes[:, 1], axis=1).astype(int)
        heights = np.linalg.norm(clipped_boxes[:, 0] - clipped_boxes[:, 3], axis=1).astype(int)
        
        # Filter out boxes with width or height <= 3
        valid_indices = (widths > 3) & (heights > 3)
        dt_boxes_new = clipped_boxes[valid_indices]

        return dt_boxes_new

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def prepare_image(self, img):
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        return data
    def preprocess(self,img):
        
        img, shape_list = self.prepare_image(img)
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        inp = torch.from_numpy(img)
        if self.use_gpu:
            inp = inp.cuda()
        return inp, shape_list

    def postprocess(self, outputs,shape_list,ori_shape):
        preds = {}
        if self.det_algorithm == "EAST":
            preds['f_geo'] = outputs['f_geo'].cpu().numpy()
            preds['f_score'] = outputs['f_score'].cpu().numpy()
        elif self.det_algorithm == 'SAST':
            preds['f_border'] = outputs['f_border'].cpu().numpy()
            preds['f_score'] = outputs['f_score'].cpu().numpy()
            preds['f_tco'] = outputs['f_tco'].cpu().numpy()
            preds['f_tvo'] = outputs['f_tvo'].cpu().numpy()
        elif self.det_algorithm in ['DB', 'PSE', 'DB++']:
            preds['maps'] = outputs['maps'].cpu().numpy()
        elif self.det_algorithm == 'FCE':
            for i, (k, output) in enumerate(outputs.items()):
                preds['level_{}'.format(i)] = output
        else:
            raise NotImplementedError
    
        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        if (self.det_algorithm == "SAST" and
            self.det_sast_polygon) or (self.det_algorithm in ["PSE", "FCE"] and
                                       self.postprocess_op.box_type == 'poly'):
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_shape)
        return dt_boxes
    
    def __call__(self, img):
        ori_shape = img.shape
        inp,shape_list = self.preprocess(img)
        starttime = time.time()
        with torch.no_grad():
            outputs = self.net(inp)
        dt_boxes = self.postprocess(outputs,shape_list,ori_shape)
        elapse = time.time() - starttime
        return dt_boxes, elapse
    

fast_config = {'use_gpu': True,
 'gpu_mem': 500,
 'warmup': False,
 'image_dir': './doc/imgs/1.jpg',
 'det_algorithm': 'DB',
 'det_model_path': 'weights/en_ptocr_v3_det_infer.pth',
 'det_limit_side_len': 960,
 'det_limit_type': 'max',
 'det_db_thresh': 0.3,
 'det_db_box_thresh': 0.6,
 'det_db_unclip_ratio': 1.5,
 'max_batch_size': 10,
 'use_dilation': False,
 'det_db_score_mode': 'fast',
 'det_east_score_thresh': 0.8,
 'det_east_cover_thresh': 0.1,
 'det_east_nms_thresh': 0.2,
 'det_sast_score_thresh': 0.5,
 'det_sast_nms_thresh': 0.2,
 'det_sast_polygon': False,
 'det_pse_thresh': 0,
 'det_pse_box_thresh': 0.85,
 'det_pse_min_area': 16,
 'det_pse_box_type': 'box',
 'det_pse_scale': 1,
 'scales': [8, 16, 32],
 'alpha': 1.0,
 'beta': 1.0,
 'fourier_degree': 5,
 'det_fce_box_type': 'poly',
 'rec_algorithm': 'CRNN',
 'rec_model_path': 'weights/en_ptocr_v4_rec_infer.pth',
 'rec_image_inverse': True,
 'rec_image_shape': '3, 32, 320',
 'rec_char_type': 'ch',
 'rec_batch_num': 6,
 'max_text_length': 25,
 'use_space_char': True,
 'drop_score': 0.5,
 'limited_max_width': 1280,
 'limited_min_width': 16,
 'vis_font_path': '/mnt/data/zhangtianning/projects/doc/fonts/simfang.ttf',
 'rec_char_dict_path': './pytorchocr/utils/en_dict.txt',
 'use_angle_cls': False,
 'cls_model_path': None,
 'cls_image_shape': '3, 48, 192',
 'label_list': ['0', '180'],
 'cls_batch_num': 6,
 'cls_thresh': 0.9,
 'enable_mkldnn': False,
 'use_pdserving': False,
 'e2e_algorithm': 'PGNet',
 'e2e_model_path': None,
 'e2e_limit_side_len': 768,
 'e2e_limit_type': 'max',
 'e2e_pgnet_score_thresh': 0.5,
 'e2e_char_dict_path': '/mnt/data/zhangtianning/projects/pytorchocr/utils/ic15_dict.txt',
 'e2e_pgnet_valid_set': 'totaltext',
 'e2e_pgnet_polygon': True,
 'e2e_pgnet_mode': 'fast',
 'sr_model_path': None,
 'sr_image_shape': '3, 32, 128',
 'sr_batch_num': 1,
 'det_yaml_path': 'configs/det/det_ppocr_v3.yml',
 'rec_yaml_path': './configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml',
 'cls_yaml_path': None,
 'e2e_yaml_path': None,
 'sr_yaml_path': None,
 'use_mp': False,
 'total_process_num': 1,
 'process_id': 0,
 'benchmark': False,
 'save_log_path': './log_output/',
 'show_log': True}
from argparse import Namespace
@dataclass
class PostProcessConfig:
    thresh:float
    unclip_ratio:float 
    max_candidates:int 
    min_size:int 
    box_thresh:float 


class BatchTextDetector(TextDetector):
    def __init__(self, **kwargs):
        args = Namespace(**fast_config)
        super().__init__(args, **kwargs)

    def batch_forward(self, _input_image_batch,shape_list_batch,ori_shape_list):
        with torch.no_grad():
            dt_boxaes_batch = self.net(_input_image_batch)
            pred_batch  = self.discard_batch(dt_boxaes_batch)
            dt_boxes_list=self.batch_postprocess(pred_batch, shape_list_batch,ori_shape_list)
        return dt_boxes_list
    def batch_process(self, img_batch, ori_shape_list):
        _input_image_batch = []
        shape_list_batch   = []
        for img in img_batch:
            _input_image, shape_list = self.preprocess(img)
            _input_image_batch.append(_input_image)
            shape_list_batch.append(shape_list)
        _input_image_batch = torch.cat(_input_image_batch)
        shape_list_batch   = np.stack(shape_list_batch)
        return self.batch_forward(self, _input_image_batch,shape_list_batch,ori_shape_list)
    
    def discard_batch(self, outputs):
        preds = {}
        if self.det_algorithm == "EAST":
            raise NotImplementedError
            preds['f_geo']      =  outputs['f_geo'].cpu().numpy()
            preds['f_score']    =  outputs['f_score'].cpu().numpy()
        elif self.det_algorithm == 'SAST':
            raise NotImplementedError
            preds['f_border']   =  outputs['f_border'].cpu().numpy()
            preds['f_score']    =  outputs['f_score'].cpu().numpy()
            preds['f_tco']      =  outputs['f_tco'].cpu().numpy()
            preds['f_tvo']      =  outputs['f_tvo'].cpu().numpy()
        elif self.det_algorithm in ['DB', 'PSE', 'DB++']:
            preds = [{'maps':outputs['maps'][j:j+1]} for j in range(len(outputs['maps']))]
        elif self.det_algorithm == 'FCE':
            for i, (k, output) in enumerate(outputs.items()):
                preds['level_{}'.format(i)] = output
        else:
            raise NotImplementedError
        return preds
    
    def fast_postprocess(self,preds, shape_list ):
        #return fast_torch_postprocess(self.postprocess_op,preds, shape_list)
        config = PostProcessConfig(thresh=self.postprocess_op.thresh, 
                                   unclip_ratio=self.postprocess_op.unclip_ratio, 
                                   max_candidates=self.postprocess_op.max_candidates, 
                                   min_size=self.postprocess_op.min_size, 
                                   box_thresh=self.postprocess_op.box_thresh)
    
        if len(shape_list) == 1:
            return fast_torch_postprocess(preds, shape_list,config)
        else:
            return fast_torch_postprocess_multiprocess(preds, shape_list,config)
        

    def batch_postprocess(self, preds_list, shape_list_list,ori_shape_list):
        dt_boxes_list=[]
        for preds, shape_list,ori_shape in zip(preds_list, shape_list_list,ori_shape_list):
            post_result = self.fast_postprocess(preds, shape_list)
            dt_boxes = post_result[0]['points']
            if (self.det_algorithm == "SAST" and self.det_sast_polygon) or (self.det_algorithm in ["PSE", "FCE"] and self.postprocess_op.box_type == 'poly'):
                raise NotImplementedError
                dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_shape)
            else:
                dt_boxes = self.filter_tag_det_res(dt_boxes, ori_shape)
            dt_boxes_list.append(dt_boxes)
        return dt_boxes_list

def fast_torch_postprocess(self, outs_dict, shape_list):
    """
    Accelerate below 
    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu().numpy()
    

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(np.array(segmentation[batch_index]).astype(np.uint8),self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,src_w, src_h)

            boxes_batch.append({'points': boxes})
        return boxes_batch
    """
    pred_batch = outs_dict['maps'][:, 0, :, :]
    pred_batch = pred_batch
    if isinstance(pred_batch, torch.Tensor):pred_batch= pred_batch.cpu().numpy()
    segmentation_batch = pred_batch > self.thresh
    if isinstance(segmentation_batch, torch.Tensor):segmentation_batch = segmentation_batch.cpu().numpy()
    config = PostProcessConfig(thresh=self.thresh, unclip_ratio=self.unclip_ratio, max_candidates=self.max_candidates, min_size=self.min_size, box_thresh=self.box_thresh)
    
    boxes_batch = []
    for batch_index in range(pred_batch.shape[0]):
        src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
        boxes, scores = boxes_from_bitmap(pred_batch[batch_index], segmentation_batch[batch_index],src_w, src_h,config)
        #boxes = boxes_from_bitmap_without_score(self,segmentation_batch[batch_index],src_w, src_h)
        boxes_batch.append({'points': boxes})
    return boxes_batch

def get_contours_multiprocess(segmentation_mask):
    """Process a single segmentation batch and find contours."""
    outs= cv2.findContours((segmentation_mask * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(outs) == 3:
        img, contours, _ = outs[0], outs[1], outs[2]
    elif len(outs) == 2:
        contours, _ = outs[0], outs[1]
    return contours
from concurrent.futures import ThreadPoolExecutor
def fast_torch_postprocess_multiprocess(outs_dict, shape_list, config):
    """
    Accelerate below 
    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu().numpy()
    

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(np.array(segmentation[batch_index]).astype(np.uint8),self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,src_w, src_h)

            boxes_batch.append({'points': boxes})
        return boxes_batch
    """
    if isinstance(outs_dict, dict):
        pred_batch = outs_dict['maps'][:, 0, :, :]
        pred_batch = pred_batch
    else:
        pred_batch = outs_dict
    if isinstance(pred_batch, torch.Tensor):pred_batch= pred_batch.cpu().numpy()
    segmentation_batch = pred_batch > config.thresh
    if isinstance(segmentation_batch, torch.Tensor):segmentation_batch = segmentation_batch.cpu().numpy()
    
    
    num_threads = min(8, len(segmentation_batch))
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        contours_batch = list(executor.map(get_contours_multiprocess, segmentation_batch))

    boxes_batch = []
    for batch_index in range(pred_batch.shape[0]):
        src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
        #boxes, scores = boxes_from_bitmap(self,pred_batch[batch_index], segmentation_batch[batch_index],src_w, src_h)
        boxes = boxes_from_contours(pred_batch[batch_index],contours_batch[batch_index],src_w, src_h, config)
        boxes_batch.append({'points': boxes})

    # def boxes_from_bitmap_wrapper(args):
    #     pred_now,segmentation_now,src_w, src_h, config = args
    #     boxes = boxes_from_bitmap(pred_now,segmentation_now,src_w, src_h, config)
    #     return {'points': boxes}
    
    # with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     src_h_list=[src_h for src_h, src_w, ratio_h, ratio_w in shape_list]
    #     src_w_list=[src_w for src_h, src_w, ratio_h, ratio_w in shape_list]
    #     configlist=[config]*len(shape_list)
    #     boxes_batch = list(executor.map(boxes_from_bitmap_wrapper, zip(pred_batch,segmentation_batch,src_w_list,src_h_list,configlist)))
    return boxes_batch

def boxes_from_contours(pred, contours, dest_width, dest_height,config):
    '''
    _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
    '''

    height, width = pred.shape
    num_contours = min(len(contours), config.max_candidates)

    boxes = []
    scores = []
    for index in range(num_contours):
        contour = contours[index]
        result  = deal_with_on_contours(contour, pred, height, width, dest_height, dest_width, config)
        if result is None:continue
        box, score = result
        boxes.append(box)
        scores.append(score)
    return np.array(boxes), scores

def deal_with_on_contours(contour, score_table, height, width, dest_height, dest_width, config):
    points, sside = get_mini_boxes(contour)
    if sside < config.min_size:return
    points =np.array(points)
    score  = box_score_fast(score_table, points.reshape(-1, 2))
    if config.box_thresh > score:return
    box = unclip(points,config.unclip_ratio).reshape(-1, 1, 2)
    box, sside = get_mini_boxes(box)
    if sside < config.min_size + 2:return
    box = np.array(box)
    box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
    box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
    return box, score

def boxes_from_bitmap(pred, _bitmap, dest_width, dest_height,config:PostProcessConfig):
    '''
    _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
    '''

    bitmap = _bitmap
    height, width = bitmap.shape

    outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(outs) == 3:
        img, contours, _ = outs[0], outs[1], outs[2]
    elif len(outs) == 2:
        contours, _ = outs[0], outs[1]
    num_contours = min(len(contours), config.max_candidates)

    boxes = []
    scores = []
    for index in range(num_contours):
        contour = contours[index]
        result  = deal_with_on_contours(contour, pred, height, width, dest_height, dest_width, config)
        if result is None:continue
        box, score = result
        boxes.append(box)
        scores.append(score)
    return np.array(boxes), scores

def boxes_from_bitmap_without_score(self, _bitmap, dest_width, dest_height):
    '''
    _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
    '''

    bitmap = _bitmap
    height, width = bitmap.shape

    outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(outs) == 3:
        img, contours, _ = outs[0], outs[1], outs[2]
    elif len(outs) == 2:
        contours, _ = outs[0], outs[1]

    num_contours = min(len(contours), self.max_candidates)

    boxes = []
    for index in range(num_contours):
        contour = contours[index]
        points, sside = get_mini_boxes(contour)
        if sside < self.min_size:continue
        points =np.array(points)
        box = unclip(points).reshape(-1, 1, 2)
        box, sside = get_mini_boxes(box)
        if sside < self.min_size + 2:continue
        box = np.array(box)
        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.astype(np.int16))
    return np.array(boxes, dtype=np.int16)

def obtain_score_mask(_box, h, w):
    #h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return xmin, xmax, ymin, ymax, mask

def box_score_fast(bitmap, _box):
    '''
    box_score_fast: use bbox mean score as the mean score
    '''
    h, w = bitmap.shape[:2]
    xmin, xmax, ymin, ymax, mask = obtain_score_mask(_box,h, w)
    crop = bitmap[ymin:ymax + 1, xmin:xmax + 1]
    mask = torch.BoolTensor(mask)
    return crop[mask].mean().item()
    
def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box, min(bounding_box[1])

def unclip(box, unclip_ratio):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded

def deal_with_on_contours_without_score(contour, height, width, dest_height, dest_width, config):
    points, sside = get_mini_boxes(contour)
    if sside < config.min_size:return
    points =np.array(points)
    box = unclip(points,config.unclip_ratio).reshape(-1, 1, 2)
    box, sside = get_mini_boxes(box)
    if sside < config.min_size + 2:return
    box = np.array(box)
    box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
    box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
    return box