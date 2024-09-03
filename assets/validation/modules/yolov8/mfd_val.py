# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import cv2
import json
import warnings
import numpy as np
import torch
from pathlib import Path
from threading import Thread
from mmeval import COCODetection

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops, TQDM, yaml_load, emojis
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.checks import check_requirements, check_imgsz, check_file
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.nn.autobackend import AutoBackend


class MFDValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import MFDValidator

        args = dict(model='yolov8n.pt', data='coco8.yaml')
        validator = MFDValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'detect'
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255
        for k in ['batch_idx', 'cls', 'bboxes']:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch['img'].shape[2:]
            nb = len(batch['img'])
            bboxes = batch['bboxes'] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch['cls'][batch['batch_idx'] == i], bboxes[batch['batch_idx'] == i]], dim=-1)
                for i in range(nb)] if self.args.save_hybrid else []  # for autolabelling

        return batch

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None, vis_box=True):
        """Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
        gets priority).
        """
        # label_classes = ['inline_formula', "formula"]
        # label_classes = ['footnote', 'footer', 'header']
        file = check_file(self.args.data)
        data = yaml_load(file, append_filename=True)
        label_classes = data['names']

        model = AutoBackend(model or self.args.model,
                            device=select_device(self.args.device, self.args.batch),
                            dnn=self.args.dnn,
                            data=self.args.data,
                            fp16=self.args.half)
        # self.model = model
        self.device = model.device  # update device
        self.args.half = model.fp16  # update half
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_imgsz(self.args.imgsz, stride=stride)
        if engine:
            self.args.batch = model.batch_size
        elif not pt and not jit:
            self.args.batch = 1  # export.py models default to batch-size 1
            LOGGER.info(f'Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        if str(self.args.data).split('.')[-1] in ('yaml', 'yml'):
            self.data = check_det_dataset(self.args.data)
        else:
            raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

        if self.device.type in ('cpu', 'mps'):
            self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
        if not pt:
            self.args.rect = False
        self.stride = model.stride  # used in get_dataloader() for padding
        self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

        model.eval()
        model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        
        img_idx = 0
        all_images, all_gts, all_preds = [], [], []
        for batch_i, batch in enumerate(bar):
            self.batch_i = batch_i
            batch = self.preprocess(batch)
            preds = model(batch['img'], augment=False)
            preds = self.postprocess(preds)
            for im_path, ori_shape, re_shape, pred in zip(batch['im_file'], batch['ori_shape'], batch['resized_shape'], preds):
                pred[:, :4] = ops.scale_boxes(re_shape, pred[:, :4], ori_shape)
                all_images.append(im_path)
                bboxes, labels, scores = [], [], []
                for *xyxy, conf, cla in reversed(pred):
                    bboxes.append([p.item() for p in xyxy])
                    labels.append(int(cla.item()))
                    scores.append(float(conf.item()))
                all_preds.append({
                    'img_id': img_idx,
                    'bboxes': np.array(bboxes),
                    'scores': np.array(scores),
                    'labels': np.array(labels),
                })
                # GroundTruths
                gt_boxes, gt_labels, gt_ignore = [], [], []
                label_path = im_path.replace("/images/", "/labeling_anns/")[0:-4] + ".json"
                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        ann = json.load(f)
                    W, H = ann['width'], ann['height']
                    for item in ann['step_1']['result']:
                        if item['attribute'] in label_classes:
                            gt_boxes.append([item['x'], item['y'], item['x']+item['width'], item['y']+item['height']])
                            gt_labels.append(label_classes.index(item['attribute']))
                            gt_ignore.append(False)
                        else:
                            gt_boxes.append([item['x'], item['y'], item['x']+item['width'], item['y']+item['height']])
                            gt_labels.append(0)
                            gt_ignore.append(True)
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        label_path = im_path.replace("/images/", "/labels/")[0:-4] + ".txt"
                        gt_lines = np.genfromtxt(label_path, usecols=(0,1,2,3,4,5,6,7,8),)
                        H, W = ori_shape
                        if len(gt_lines) == 0:
                            pass
                        else:
                            if len(gt_lines.shape) == 1: gt_lines = [gt_lines]
                            for l in gt_lines:
                                x1 = int(l[1] * W)
                                y1 = int(l[2] * H)
                                x2 = int(l[5] * W)
                                y2 = int(l[6] * H)
                                gt_boxes.append([x1, y1, x2, y2])
                                gt_labels.append(int(l[0]))
                                gt_ignore.append(False)
                                
                all_gts.append({
                    'img_id': img_idx,
                    'width': W,
                    'height': H,
                    'bboxes': np.array(gt_boxes),
                    'labels': np.array(gt_labels),
                    'ignore_flags': gt_ignore,
                })                
                img_idx += 1
                if vis_box:
                    Thread(target=self.visulize, args=(all_images[-1], all_gts[-1], all_preds[-1], label_classes, 0.25, self.save_dir), daemon=True).start()
        ## coco metric
        meta={'CLASSES':tuple(label_classes)}
        coco_det_metric = COCODetection(dataset_meta=meta, metric=['bbox'], classwise=True) 
        res = coco_det_metric(predictions=all_preds, groundtruths=all_gts)
        return res

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(preds,
                                       self.args.conf,
                                       self.args.iou,
                                       labels=self.lb,
                                       multi_label=True,
                                       agnostic=self.args.single_cls,
                                       max_det=self.args.max_det)


    def build_dataset(self, img_path, mode='val', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode='val')
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader
    
    
    
    def plot_bbox(self, img, bbox, text_info, text_location=0, mask=None, color=(0,200,0), thickness=2):
        c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(img, c1, c2, color, thickness)
        if text_info:
            # t_size=cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, 0.5 , 1)[0]
            t_size=cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, 1 , 1)[0]
            if text_location == 0:
                cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1]*1.6)), color, -1)
                # cv2.putText(img, text_info, (c1[0], c1[1] + int(t_size[1]*1.6)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, [255,255,255], 1)
                cv2.putText(img, text_info, (c1[0], c1[1] + int(t_size[1]*1.6)), cv2.FONT_HERSHEY_TRIPLEX, 1, [255,255,255], 1)
            else:
                cv2.rectangle(img, (c2[0] - int(t_size[0]), c1[1]), (c2[0], c1[1] + int(t_size[1]*1.6)), color, -1)
                # cv2.putText(img, text_info, (c2[0] - int(t_size[0]), c1[1] + int(t_size[1]*1.6)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, [255,255,255], 1)
                cv2.putText(img, text_info, (c2[0] - int(t_size[0]), c1[1] + int(t_size[1]*1.6)), cv2.FONT_HERSHEY_TRIPLEX, 1, [255,255,255], 1)
        return img

    def visulize(self, image, gt, pred, label_classes, thres=0.45, save_dir="vis"):
        img = cv2.imread(image)
        if gt["bboxes"].shape[0] == 0 and pred["bboxes"].shape[0] == 0:
            return 0
        for idx, (bbox, score, label) in enumerate(zip(pred['bboxes'], pred['scores'], pred['labels'])):
            if score > thres:
                bbox = [bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2]
                img = self.plot_bbox(img, bbox, str(label), text_location=0, color=(0,0,200), thickness=1)
                # img = plot_bbox(img, bbox, str(round(score, 2)), color=(0,0,200), thickness=1)
        for idx ,(bbox, label, ignore) in enumerate(zip(gt['bboxes'], gt['labels'], gt['ignore_flags'])):
            if ignore:
                color = (125,125,125)
            else:
                color = (0,200,0)
            img = self.plot_bbox(img, bbox, str(label), text_location=1, color=color, thickness=1)
        cv2.imwrite(os.path.join(save_dir, os.path.basename(image)), img)
