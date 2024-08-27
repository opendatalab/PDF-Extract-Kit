import os
import cv2
import json
import yaml
import time
import pytz
import datetime
import argparse
import shutil
import torch
import numpy as np
import gc

from paddleocr import draw_ocr
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from unimernet.common.config import Config
import unimernet.tasks as tasks
from unimernet.processors import load_processor
from struct_eqtable import build_model

from modules.latex2png import tex2pil, zhtext2pil
from modules.extract_pdf import load_pdf_fitz
from modules.layoutlmv3.model_init import Layoutlmv3_Predictor
from modules.self_modify import ModifiedPaddleOCR
from modules.post_process import get_croped_image, latex_rm_whitespace


def mfd_model_init(weight):
    mfd_model = YOLO(weight)
    return mfd_model


def mfr_model_init(weight_dir, device='cpu'):
    args = argparse.Namespace(cfg_path="modules/UniMERNet/configs/demo.yaml", options=None)
    cfg = Config(args)
    cfg.config.model.pretrained = os.path.join(weight_dir, "pytorch_model.bin")
    cfg.config.model.model_config.model_name = weight_dir
    cfg.config.model.tokenizer_config.path = weight_dir
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    model = model.to(device)
    vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
    return model, vis_processor

def layout_model_init(weight):
    model = Layoutlmv3_Predictor(weight)
    return model

def tr_model_init(weight, max_time, device='cuda'):
    tr_model = build_model(weight, max_new_tokens=4096, max_time=max_time)
    if device == 'cuda':
        tr_model = tr_model.cuda()
    return tr_model

class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # if not pil image, then convert to pil image
        if isinstance(self.image_paths[idx], str):
            raw_image = Image.open(self.image_paths[idx])
        else:
            raw_image = self.image_paths[idx]
        if self.transform:
            image = self.transform(raw_image)
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=str)
    parser.add_argument('--output', type=str, default="output")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    print(args)
    
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.datetime.now(tz)
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Started!')
    
    ## ======== model init ========##
    with open('configs/model_configs.yaml') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    img_size = model_configs['model_args']['img_size']
    conf_thres = model_configs['model_args']['conf_thres']
    iou_thres = model_configs['model_args']['iou_thres']
    device = model_configs['model_args']['device']
    dpi = model_configs['model_args']['pdf_dpi']
    mfd_model = mfd_model_init(model_configs['model_args']['mfd_weight'])
    mfr_model, mfr_vis_processors = mfr_model_init(model_configs['model_args']['mfr_weight'], device=device)
    mfr_transform = transforms.Compose([mfr_vis_processors, ])
    tr_model = tr_model_init(model_configs['model_args']['tr_weight'], max_time=model_configs['model_args']['table_max_time'], device=device)
    layout_model = layout_model_init(model_configs['model_args']['layout_weight'])
    ocr_model = ModifiedPaddleOCR(show_log=True)
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Model init done!')
    ## ======== model init ========##
    
    start = time.time()
    if os.path.isdir(args.pdf):
        all_pdfs = [os.path.join(args.pdf, name) for name in os.listdir(args.pdf)]
    else:
        all_pdfs = [args.pdf]
    print("total files:", len(all_pdfs))
    for idx, single_pdf in enumerate(all_pdfs):
        try:
            img_list = load_pdf_fitz(single_pdf, dpi=dpi)
        except:
            img_list = None
            print("unexpected pdf file:", single_pdf)
        if img_list is None:
            continue
        print("pdf index:", idx, "pages:", len(img_list))
        # layout detection and formula detection
        doc_layout_result = []
        latex_filling_list = []
        mf_image_list = []
        for idx, image in enumerate(img_list):
            img_H, img_W = image.shape[0], image.shape[1]
            layout_res = layout_model(image, ignore_catids=[])
            mfd_res = mfd_model.predict(image, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=True)[0]
            for xyxy, conf, cla in zip(mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()):
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                new_item = {
                    'category_id': 13 + int(cla.item()),
                    'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                    'score': round(float(conf.item()), 2),
                    'latex': '',
                }
                layout_res['layout_dets'].append(new_item)
                latex_filling_list.append(new_item)
                bbox_img = get_croped_image(Image.fromarray(image), [xmin, ymin, xmax, ymax])
                mf_image_list.append(bbox_img)
                
            layout_res['page_info'] = dict(
                page_no = idx,
                height = img_H,
                width = img_W
            )
            doc_layout_result.append(layout_res)

            del mfd_res
            torch.cuda.empty_cache()
            gc.collect()
            
        # Formula recognition, collect all formula images in whole pdf file, then batch infer them.
        a = time.time()  
        dataset = MathDataset(mf_image_list, transform=mfr_transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=32)
        mfr_res = []
        for imgs in dataloader:
            imgs = imgs.to(device)
            output = mfr_model.generate({'image': imgs})
            mfr_res.extend(output['pred_str'])
        for res, latex in zip(latex_filling_list, mfr_res):
            res['latex'] = latex_rm_whitespace(latex)
        b = time.time()
        print("formula nums:", len(mf_image_list), "mfr time:", round(b-a, 2))
            
        # ocr and table recognition
        for idx, image in enumerate(img_list):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            single_page_res = doc_layout_result[idx]['layout_dets']
            single_page_mfdetrec_res = []
            for res in single_page_res:
                if int(res['category_id']) in [13, 14]:
                    xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                    xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                    single_page_mfdetrec_res.append({
                        "bbox": [xmin, ymin, xmax, ymax],
                    })
            for res in single_page_res:
                if int(res['category_id']) in [0, 1, 2, 4, 6, 7]:  #categories that need to do ocr
                    xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                    xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                    crop_box = [xmin, ymin, xmax, ymax]
                    cropped_img = Image.new('RGB', pil_img.size, 'white')
                    cropped_img.paste(pil_img.crop(crop_box), crop_box)
                    cropped_img = cv2.cvtColor(np.asarray(cropped_img), cv2.COLOR_RGB2BGR)
                    ocr_res = ocr_model.ocr(cropped_img, mfd_res=single_page_mfdetrec_res)[0]
                    if ocr_res:
                        for box_ocr_res in ocr_res:
                            p1, p2, p3, p4 = box_ocr_res[0]
                            text, score = box_ocr_res[1]
                            doc_layout_result[idx]['layout_dets'].append({
                                'category_id': 15,
                                'poly': p1 + p2 + p3 + p4,
                                'score': round(score, 2),
                                'text': text,
                            })
                elif int(res['category_id']) == 5: # do table recognition
                    xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                    xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                    crop_box = [xmin, ymin, xmax, ymax]
                    cropped_img = pil_img.convert("RGB").crop(crop_box)
                    start = time.time()
                    with torch.no_grad():
                        output = tr_model(cropped_img)
                    end = time.time()
                    if (end-start) > model_configs['model_args']['table_max_time']:
                        res["timeout"] = True
                    res["latex"] = output[0]


        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(single_pdf)[0:-4]
        with open(os.path.join(output_dir, f'{basename}.json'), 'w') as f:
            json.dump(doc_layout_result, f)
        
        if args.vis:
            color_palette = [
                (255,64,255),(255,255,0),(0,255,255),(255,215,135),(215,0,95),(100,0,48),(0,175,0),(95,0,95),(175,95,0),(95,95,0),
                (95,95,255),(95,175,135),(215,95,0),(0,0,255),(0,255,0),(255,0,0),(0,95,215),(0,0,0),(0,0,0),(0,0,0)
            ]
            id2names = ["title", "plain_text", "abandon", "figure", "figure_caption", "table", "table_caption", "table_footnote", 
                        "isolate_formula", "formula_caption", " ", " ", " ", "inline_formula", "isolated_formula", "ocr_text"]
            vis_pdf_result = []
            for idx, image in enumerate(img_list):
                single_page_res = doc_layout_result[idx]['layout_dets']
                vis_img = Image.new('RGB', Image.fromarray(image).size, 'white') if args.render else Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                draw = ImageDraw.Draw(vis_img)
                for res in single_page_res:
                    label = int(res['category_id'])
                    if label > 15:     # categories that do not need visualize
                        continue
                    label_name = id2names[label]
                    x_min, y_min = int(res['poly'][0]), int(res['poly'][1])
                    x_max, y_max = int(res['poly'][4]), int(res['poly'][5])
                    if args.render and label in [13, 14, 15]:
                        try:
                            if label in [13, 14]:  # render formula
                                window_img = tex2pil(res['latex'])[0]
                            else:
                                if True:           # render chinese
                                    window_img = zhtext2pil(res['text'])
                                else:              # render english
                                    window_img = tex2pil([res['text']], tex_type="text")[0]
                            ratio = min((x_max - x_min) / window_img.width, (y_max - y_min) / window_img.height) - 0.05
                            window_img = window_img.resize((int(window_img.width * ratio), int(window_img.height * ratio)))
                            vis_img.paste(window_img, (int(x_min + (x_max-x_min-window_img.width) / 2), int(y_min + (y_max-y_min-window_img.height) / 2)))
                        except Exception as e:
                            print(f"got exception on {text}, error info: {e}")
                    draw.rectangle([x_min, y_min, x_max, y_max], fill=None, outline=color_palette[label], width=1)
                    fontText = ImageFont.truetype("assets/fonts/simhei.ttf", 15, encoding="utf-8")
                    draw.text((x_min, y_min), label_name, color_palette[label], font=fontText)
                
                width, height = vis_img.size
                width, height = int(0.75*width), int(0.75*height)
                vis_img = vis_img.resize((width, height))
                vis_pdf_result.append(vis_img)
            
            first_page = vis_pdf_result.pop(0)
            first_page.save(os.path.join(output_dir, f'{basename}.pdf'), 'PDF', resolution=100, save_all=True, append_images=vis_pdf_result)
            try:
                shutil.rmtree('./temp')
            except:
                pass
            
    now = datetime.datetime.now(tz)
    end = time.time()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Finished! time cost:', int(end-start), 's')