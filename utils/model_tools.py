import os
import argparse

from torchvision import transforms
from ultralytics import YOLO
from unimernet.common.config import Config
import unimernet.tasks as tasks
from unimernet.processors import load_processor
from struct_eqtable import build_model

from modules.layoutlmv3.model_init import Layoutlmv3_Predictor
from utils.config import load_config


def mfd_model_init():
    model_configs = load_config()
    weight = model_configs['model_args']['mfd_weight']
    mfd_model = YOLO(weight)
    return mfd_model


def mfr_model_init():
    model_configs = load_config()
    weight_dir = model_configs['model_args']['mfr_weight']
    device = model_configs['model_args']['device']

    args = argparse.Namespace(cfg_path="modules/UniMERNet/configs/demo.yaml", options=None)
    cfg = Config(args)
    cfg.config.model.pretrained = os.path.join(weight_dir, "pytorch_model.bin")
    cfg.config.model.model_config.model_name = weight_dir
    cfg.config.model.tokenizer_config.path = weight_dir
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    model = model.to(device)
    vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
    mfr_transform = transforms.Compose([vis_processor, ])
    return model, mfr_transform


def layout_model_init():
    model_configs = load_config()
    weight = model_configs['model_args']['layout_weight']

    model = Layoutlmv3_Predictor(weight)
    return model


def tr_model_init():
    model_configs = load_config()
    weight = model_configs['model_args']['tr_weight']
    max_time = model_configs['model_args']['table_max_time']
    device = model_configs['model_args']['device']

    tr_model = build_model(weight, max_new_tokens=4096, max_time=max_time)
    if device == 'cuda':
        tr_model = tr_model.cuda()
    return tr_model



