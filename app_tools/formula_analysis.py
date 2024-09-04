import os, gc
import time
import argparse
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ultralytics import YOLO
from unimernet.common.config import Config
import unimernet.tasks as tasks
from unimernet.processors import load_processor  ## TODO: WARNING 'load_processor' is not declared in __all__
from modules.post_process import get_croped_image, latex_rm_whitespace

from app_tools.config import load_config, setup_logging

logger = setup_logging('formula_analysis')

class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx >= len(self.image_paths) or idx < 0:
            raise IndexError("Index out of range")

        # if not pil image, then convert to pil image
        if isinstance(self.image_paths[idx], str):
            try:
                raw_image = Image.open(self.image_paths[idx])
            except IOError as e:
                raise IOError(f"Error opening image: {self.image_paths[idx]}") from e
        else:
            raw_image = self.image_paths[idx]

        # apply transformation if any
        if self.transform:
            return self.transform(raw_image)
        return raw_image


class FormulaProcessor:
    """
    The FormulaProcessor class is designed to handle formula detection and recognition in images.
    It is initialized with an optional configuration file and provides methods to detect and recognize formulas in images.

    """
    def __init__(self, config_path: str = None):
        """
        It initializes the mfd_model, mfr_model, mfr_transform, latex_filling_list, and mf_image_list properties.

        Attributes:
            - config: The configuration loaded from the config_path or default configuration.
            - mfd_model: The model used for mfd transformation.
            - mfr_model: The model used for mfr transformation.
            - mfr_transform: The transformation object used for mfr transformation.
            - latex_filling_list: A list to store latex filling data.
            - mf_image_list: A list to store MF images.

        Methods:
            - __init__: Initializes the developer object with the provided config_path or default configuration.

        Parameters:
            - config_path (str): The path to the configuration file (optional).

        Note:
            - The config_path parameter is optional. If not provided, the default configuration will be used.
            - The load_config function is used internally to load the configuration from the provided path or default configuration.
            - The _init_mfd_model, _init_mfr_model, and _init_mfr_transform methods are used internally to initialize the mfd_model, mfr_model, and mfr_transform properties respectively.
            - The latex_filling_list and mf_image_list properties are empty lists to start with.
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.mfd_model = self._init_mfd_model()
        self.mfr_model, self.mfr_transform = self._init_mfr_model()
        self.latex_filling_list = []
        self.mf_image_list = []

    def _init_mfd_model(self):
        """
        Initializes the MFD (Multiple Feature Detection) model.

        This method initializes the MFD model by setting the weight of the model from the configuration file and creating a new instance of the YOLO class.

        Returns:
            mfd_model (YOLO): The initialized MFD model.

        """
        weight = self.config['model_args']['mfd_weight']
        mfd_model = YOLO(weight)
        return mfd_model

    def _init_mfr_model(self) -> Tuple[torch.nn.Module, transforms.Compose]:
        """
        Initializes the MFR model by loading the weights and setting the device.

        Returns:
            Tuple: A tuple containing the MFR model (`torch.nn.Module`) and the transformation (`transforms.Compose`).
        """
        weight_dir = self.config['model_args']['mfr_weight']
        device = self.config['model_args']['device']

        args = argparse.Namespace(cfg_path="modules/UniMERNet/configs/demo.yaml", options=None)
        cfg = Config(args)
        cfg.config.model.pretrained = os.path.join(weight_dir, "pytorch_model.bin")
        cfg.config.model.model_config.model_name = weight_dir
        cfg.config.model.tokenizer_config.path = weight_dir
        task = tasks.setup_task(cfg)
        model = task.build_model(cfg)
        model = model.to(device)
        vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
        mfr_transform = transforms.Compose([vis_processor])
        return model, mfr_transform

    def detect_formulas(self, img_list: List, doc_layout_result: List[dict]) -> List[dict]:
        """
        Detect formulas in the given list of images and update the document layout result with the detected formulas.

        Parameters:
        - img_list: A list of images to detect formulas from.
        - doc_layout_result: A list of dictionaries representing the document layout result.
                             Each dictionary contains the layout details of a single page.

        Returns:
        - A list of dictionaries representing the updated document layout result with the detected formulas.
        """
        img_size = self.config['model_args']['img_size']
        conf_thres = self.config['model_args']['conf_thres']
        iou_thres = self.config['model_args']['iou_thres']

        logger.debug('Formula detection - init')
        start = time.time()

        for idx, image in enumerate(img_list):
            mfd_res = self.mfd_model.predict(image, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=True)[0]

            for xyxy, conf, cla in zip(mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()):
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                new_item = {
                    'category_id': 13 + int(cla.item()),
                    'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                    'score': round(float(conf.item()), 2),
                    'latex': '',
                }
                doc_layout_result[idx]['layout_dets'].append(new_item)
                self.latex_filling_list.append(new_item)
                bbox_img = get_croped_image(Image.fromarray(image), [xmin, ymin, xmax, ymax])
                self.mf_image_list.append(bbox_img)

            del mfd_res
            torch.cuda.empty_cache()
            gc.collect()

        logger.debug(f'Formula detection done in {round(time.time() - start, 2)}s!')

        return doc_layout_result

    def recognize_formulas(self, batch_size: int = 128):
        """
        This method is used to recognize formulas in a batch of images.

        This method performs formula recognition by iterating over a dataset of images.
        It uses a pre-trained model to generate predictions for each image in the dataset.
        The recognized formulas are then stored in a list.
        Finally, the method logs the number of formulas recognized and the time taken for formula recognition.

        The method takes an optional argument `batch_size` that specifies the number of images to process in each batch.
        This can be useful for managing memory usage. The default batch size is 128.

        Parameters:
        - batch_size: An integer specifying the batch size. Default is 128.

        Returns:
            None

        Note:
            - The method assumes that the pre-trained model and the image dataset have already been initialized and assigned to the appropriate instance variables.
        """
        device = self.config['model_args']['device']

        logger.debug('Formula recognition')
        start = time.time()

        dataset = MathDataset(self.mf_image_list, transform=self.mfr_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=32)
        mfr_res = []
        for imgs in dataloader:
            imgs = imgs.to(device)
            output = self.mfr_model.generate({'image': imgs})
            mfr_res.extend(output['pred_str'])
        for res, latex in zip(self.latex_filling_list, mfr_res):
            res['latex'] = latex_rm_whitespace(latex)

        logger.info(f'Formula nums: {len(self.mf_image_list)} mfr time: {round(time.time() - start, 2)}')

    def detect_recognize_formulas(self, img_list: List, doc_layout_result: List[dict], batch_size: int = 128):
        """
        Detect and recognize formulas in document layout results.

        This method takes a list of images, a list of document layout results, and an optional batch size.
        It detects formulas in the document layout results by calling the `detect_formulas` method.
        Then, it recognizes the detected formulas using the `recognize_formulas` method.
        Finally, it returns the updated document layout results.

        Parameters:
        - `img_list` (List): A list of images.
        - `doc_layout_result` (List[dict]): A list of document layout results.
        - `batch_size` (int, optional): The batch size for recognition. Defaults to 128.

        Returns:
        - `doc_layout_result` (List[dict]): The updated document layout results.

        """
        doc_layout_result = self.detect_formulas(img_list, doc_layout_result)
        self.recognize_formulas(batch_size)
        return doc_layout_result

    def clear_memory(self):
        """
        Clears the models from memory, freeing up resources.
        """
        logger.info('Clearing models from memory.')
        if self.mfd_model is not None:
            del self.mfd_model
            self.mfd_model = None

        if self.mfr_model is not None:
            del self.mfr_model
            self.mfr_model = None

        torch.cuda.empty_cache()
        gc.collect()
        logger.info('Models successfully cleared from memory.')
