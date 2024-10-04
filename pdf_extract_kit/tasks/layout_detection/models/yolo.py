import os
import cv2
import math
import random
import numpy as np
from pathlib import Path
from copy import deepcopy
from PIL import Image, ImageOps
from multiprocessing.pool import ThreadPool

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from pdf_extract_kit.registry import MODEL_REGISTRY
from pdf_extract_kit.utils.visualization import  visualize_bbox
from pdf_extract_kit.dataset.dataset import ImageDataset

from ultralytics import YOLOv10
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, NUM_THREADS, TQDM
from ultralytics.data import build_dataloader
from ultralytics.data.augment import Compose, Instances
from ultralytics.data.utils import IMG_FORMATS, exif_size
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils import ops


# TODO: technically this is not an augmentation, maybe we should put this to another files
class Format:
    """
    Formats image annotations for object detection, instance segmentation, and pose estimation tasks. The class
    standardizes the image and instance annotations to be used by the `collate_fn` in PyTorch DataLoader.

    Attributes:
        bbox_format (str): Format for bounding boxes. Default is 'xywh'.
        normalize (bool): Whether to normalize bounding boxes. Default is True.
        return_mask (bool): Return instance masks for segmentation. Default is False.
        return_keypoint (bool): Return keypoints for pose estimation. Default is False.
        mask_ratio (int): Downsample ratio for masks. Default is 4.
        mask_overlap (bool): Whether to overlap masks. Default is True.
        batch_idx (bool): Keep batch indexes. Default is True.
        bgr (float): The probability to return BGR images. Default is 0.0.
    """

    def __init__(
        self,
        bbox_format="xywh",
        normalize=True,
        return_mask=False,
        return_keypoint=False,
        return_obb=False,
        mask_ratio=4,
        mask_overlap=True,
        batch_idx=True,
        bgr=0.0,
    ):
        """Initializes the Format class with given parameters."""
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask  # set False when training detection only
        self.return_keypoint = return_keypoint
        self.return_obb = return_obb
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx  # keep the batch indexes
        self.bgr = bgr

    def __call__(self, labels):
        """Return formatted image, classes, bounding boxes & keypoints to be used by 'collate_fn'."""
        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        
        if "instances" in labels:
            instances = labels.pop("instances")
            instances.convert_bbox(format=self.bbox_format)
            instances.denormalize(w, h)
            nl = len(instances)
        else:
            nl = 0

        if self.normalize and "instances" in labels:
            instances.normalize(w, h)
        labels["img"] = self._format_img(img)
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        
        # Then we can use collate_fn
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        return labels

    def _format_img(self, img):
        """Format the image for YOLO from Numpy array to PyTorch tensor."""
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr else img)
        img = torch.from_numpy(img)
        return img


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        if "instances" in labels:
            labels["instances"].convert_bbox(format="xyxy")
            labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
            labels["instances"].scale(*ratio)
            labels["instances"].add_padding(padw, padh)
        return labels

class DetectionYOLODataset(Dataset):
    def __init__(self, image_list, imgsz, batch_size, rect=True, stride=32, pad=0.5):
        super().__init__()
        self.imgsz = imgsz
        self.transforms = self.build_transforms()
        self.im_files = self.get_img_files(image_list)
        self.labels = self.cache_labels()
        self.ni = len(self.labels)    # number of images
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        self.rect = rect
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()
        
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                batch_idx=True,
            )
        )
        return transforms
        
    def verify_image(self, im_file):
        nc, msg = 0, ""
        try:
            im = Image.open(im_file)
            im.verify()  # PIL verify
            shape = exif_size(im)  # image size
            shape = (shape[1], shape[0])  # hw
            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
            if im.format.lower() in ("jpg", "jpeg"):
                with open(im_file, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                        msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
            return [im, im_file, shape, nc, msg]
        except Exception as e:
            nc = 1
            msg = f"WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
            return [None, None, nc, msg]
        
        return im, im_file, shape, nc, msg
        
    def cache_labels(self):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nc, msgs = 0, []  # number missing, found, empty, corrupt, messages
        desc = f"Scanning ..."
        total = len(self.im_files)
        nkpt, ndim = (0, 0)
        
        verify_func = self.verify_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_func,
                iterable=self.im_files,
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, im_path, shape, nc_f, msg in pbar:
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        dict(
                            im_path=im_path,
                            shape=shape,
                            cls=None,  # n, 1
                            bboxes=None,  # n, 4
                            segments=None,
                            keypoints=None,
                            normalized=True,
                            bbox_format="xywh",
                        )
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {total} images, {nc} corrupt"
            pbar.close()

        return x["labels"]
        
    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image
        
    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        f = self.im_files[i]
        im = cv2.imread(f)  # BGR
        if im is None:
            raise FileNotFoundError(f"Image Not Found {f}")

        h0, w0 = im.shape[:2]  # orig hw
        if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
            r = self.imgsz / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        return im, (h0, w0), im.shape[:2]
        
    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = []
        keypoints = None
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")
        if bboxes is not None:
            label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label
    
    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p).resolve()  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file() and 'txt' in str(p):  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                elif p.is_file() and not ('txt' in str(p)):
                    parent = str(p.parent) + os.sep
                    f.append(str(p).replace("./", parent) if str(p).startswith("./") else str(p))
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            assert im_files, f"{self.prefix}No images found in {img_path}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}") from e
        return im_files
    
    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        image_and_label = self.transforms(self.get_image_and_label(index))
        return image_and_label
    
    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)
    
    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k == "im_path":
                value = list(value)
            if k in ["masks", "keypoints", "bboxes", "cls", "segments", "obb"]:
                if value[0] is not None:
                    value = torch.cat(value, 0)
                else:
                    value = None
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

@MODEL_REGISTRY.register('layout_detection_yolo')
class LayoutDetectionYOLO:
    def __init__(self, config):
        """
        Initialize the LayoutDetectionYOLO class.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        # Mapping from class IDs to class names
        self.id_to_names = {
            0: 'title', 
            1: 'plain text',
            2: 'abandon', 
            3: 'figure', 
            4: 'figure_caption', 
            5: 'table', 
            6: 'table_caption', 
            7: 'table_footnote', 
            8: 'isolate_formula', 
            9: 'formula_caption'
        }
        
        # Set model parameters
        self.img_size = config.get('img_size', 1280)
        self.pdf_dpi = config.get('pdf_dpi', 200)
        self.conf_thres = config.get('conf_thres', 0.25)
        self.iou_thres = config.get('iou_thres', 0.45)
        self.visualize = config.get('visualize', False)
        self.rect = config.get('rect', True)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = config.get('batch_size', 1)
        self.max_det = config.get('max_det', 300)
        self.nc = config.get('nc', 10)
        self.workers = config.get('workers', 8)
        
        # Load the YOLO model from the specified path
        self.cur_device = select_device(self.device, self.batch_size)
        self.model = AutoBackend(
            weights=config['model_path'],
            device=self.cur_device,
            dnn=False,
            data=None,
            fp16=False,
        )

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["processed_img"] = batch["img"].to(self.cur_device, non_blocking=True)
        batch["processed_img"] = batch["processed_img"].float() / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.cur_device)
        return batch
        
    def postprocess(self, preds, conf=None):
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        preds = preds.transpose(-1, -2)
        boxes, scores, labels = ops.v10postprocess(preds, self.max_det, self.nc)
        bboxes = ops.xywh2xyxy(boxes)
        
        preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
        if preds.shape[-1] == 6 and conf is not None:  # end-to-end model (BNC, i.e. 1,300,6)
            preds = [pred[pred[:, 4] > conf] for pred in preds]
        return preds
        
    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        return dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn
        
    def predict(self, images, result_path, image_ids=None):
        """
        Predict layouts in images.

        Args:
            images (list): List of images to be predicted.
            result_path (str): Path to save the prediction results.
            image_ids (list, optional): List of image IDs corresponding to the images.

        Returns:
            list: List of prediction results.
        """
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        self.dataset = DetectionYOLODataset(
            image_list = images,
            imgsz = self.img_size,
            batch_size = self.batch_size,
            rect = self.rect,
        )
        self.dataloader = build_dataloader(self.dataset, self.batch_size, self.workers, shuffle=False, rank=-1)
        
        bar = TQDM(self.dataloader, desc="Batch Inferencing ...", total=len(self.dataloader))
        results = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            # Preprocess
            batch = self.preprocess(batch)
            # Inference
            preds = self.model(batch["processed_img"], augment=False)
            preds = self.postprocess(preds, conf=self.conf_thres)

            # visualize
            if self.visualize:
                for si, pred in enumerate(preds):
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)
                        
                    pbatch = self._prepare_batch(si, batch)
                    predn = self._prepare_pred(pred, pbatch)
                        
                    boxes = predn[:,:4].cpu().numpy()
                    scores = predn[:,4].cpu().numpy()
                    classes = predn[:,-1].cpu().numpy()
                    
                    vis_result = visualize_bbox(batch["im_path"][si], boxes, classes, scores, self.id_to_names)

                    # Determine the base name of the image
                    base_name = os.path.basename(batch['im_path'][si])
                    result_name = f"{base_name}_layout.png"

                    # Save the visualized result                
                    cv2.imwrite(os.path.join(result_path, result_name), vis_result)
                    
                    # append result
                    results.append({
                        "im_path": batch['im_path'][si],
                        "boxes": boxes,
                        "scores": scores,
                        "classes": classes,
                    })
        return results