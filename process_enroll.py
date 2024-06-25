import os
import io
import cv2
import json
import yaml
import pytz
import time
import datetime
import argparse
import shutil
import numpy as np

from modules.s3_utils import *
from paddleocr import draw_ocr
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from unimernet.common.config import Config
import unimernet.tasks as tasks
from unimernet.processors import load_processor

from modules.latex2png import tex2pil, zhtext2pil
from modules.extract_pdf import load_pdf_fitz
from modules.layoutlmv3.model_init import Layoutlmv3_Predictor
from modules.self_modify import ModifiedPaddleOCR
from modules.post_process import get_croped_image, latex_rm_whitespace

from modules.faiss_model.faiss_run import Faiss_Index

import logging
# logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)

def mfd_model_init(checkpoint):
    mfd_model = YOLO(checkpoint)
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
    parser.add_argument("--input", help = "input enroll jsonl file", type = str)
    parser.add_argument("--output", help = "output enroll jsonl file", type = str)
    args = parser.parse_args()
    print(args)
    
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.datetime.now(tz)
    start = time.time()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("Started!")
    
    ## ======== model init ========##
    with open('global_args.yaml') as f:
        global_args = yaml.load(f, Loader=yaml.FullLoader)

    img_size = global_args['model_args']['img_size']
    conf_thres = global_args['model_args']['conf_thres']
    iou_thres = global_args['model_args']['iou_thres']
    device = global_args['model_args']['device']
    dpi = global_args['model_args']['pdf_dpi']
    mfd_model = mfd_model_init(global_args['model_args']['mfd_weight'])
    mfr_model, mfr_vis_processors = mfr_model_init(global_args['model_args']['mfr_weight'], device=device)
    mfr_transform = transforms.Compose([mfr_vis_processors, ])
    layout_model = layout_model_init(global_args['model_args']['layout_weight'])
    ocr_model = ModifiedPaddleOCR(show_log=False)
    faiss_model = Faiss_Index(global_args['model_args']['faiss_img_list'], global_args['model_args']['img_ap_ar'])
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Model init done!')
    ## ======== model init ========##

    ## ======== parse s3 path ========##
    input_s3_dir = global_args['s3_args']['input_s3_dir']
    output_s3_dir = global_args['s3_args']['output_s3_dir']
    s3_load_cfg = get_s3_cfg_by_bucket(input_s3_dir)
    s3_load_client = get_s3_client('', s3_load_cfg)
    s3_save_cfg = get_s3_cfg_by_bucket(output_s3_dir)
    s3_save_client = get_s3_client('', s3_save_cfg)
    ## ======== parse s3 path ========##
    
    content = read_s3_object_content(s3_load_client, args.input)
    all_datas = io.BytesIO(content)
    save_content = ""
    for i, line in enumerate(all_datas):
        if not line:
            continue
        pdf_info = json.loads(line)
        if pdf_info.get("doc_layout_result", False):
            print("=> pdf already processed.")
            save_content += json.dumps(pdf_info, ensure_ascii=False) + "\n"
            continue
        temp_path = ""
        
        try:
            pdf_s3_cfg = get_s3_cfg_by_bucket(pdf_info['path'])
            print("=> processing s3 pdf:", pdf_info['path'])
            temp_path = download_s3_asset(pdf_info['path'], pdf_s3_cfg)
            img_list = load_pdf_fitz(temp_path, dpi=dpi)
            print("=> pdf pages:", len(img_list))
            doc_layout_result = []
            latex_filling_list = []
            mf_image_list = []
            for idx, image in enumerate(img_list):
                img_H, img_W = image.shape[0], image.shape[1]
                layout_res = layout_model(image, ignore_catids=[])
                
                # 单页面检索
                check_img = faiss_model.trans_img(image)
                D, I = faiss_model.index.search(check_img, 10)
                ap_list = faiss_model.get_retrival_ap_list(I, D)
                search_judge, cannot_find = faiss_model.low_ap_percentage(ap_list)
                score_judge = faiss_model.score_judge(layout_res['layout_dets'])  # 仅计算了layout检测的score
                final_judge = score_judge and search_judge
                layout_res['judge'] = {'final_judge': final_judge, 'search_judge': search_judge, 'score_judge': score_judge, 'cannot_find': cannot_find, 'search_list': ap_list}

                mfd_res = mfd_model.predict(image, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=False)[0]
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
                
            # 公式识别，因为识别速度较慢，为了提速，把单个pdf的所有公式裁剪完，一起批量做识别。    
            dataset = MathDataset(mf_image_list, transform=mfr_transform)
            dataloader = DataLoader(dataset, batch_size=128, num_workers=32)
            mfr_res = []
            for imgs in dataloader:
                imgs = imgs.to(device)
                output = mfr_model.generate({'image': imgs})
                mfr_res.extend(output['pred_str'])
            for res, latex in zip(latex_filling_list, mfr_res):
                res['latex'] = latex_rm_whitespace(latex)
                
            # ocr识别
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
                    if int(res['category_id']) in [0, 1, 2, 4, 6, 7]:  #需要进行ocr的类别
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
            if doc_layout_result:
                print("=> process done, add to line info.")
                pdf_info["doc_layout_result"] = doc_layout_result
        except Exception as e:
            print(f"got exception: {e}")
            
        save_content += json.dumps(pdf_info, ensure_ascii=False) + "\n"
        if os.path.exists(temp_path):
            print("=> remove temp pdf path:", temp_path)
            os.remove(temp_path)

    print("Save jsonline to ceph.")
    write_s3_object_content(s3_save_client, args.output, save_content.encode('utf-8'))
    now = datetime.datetime.now(tz)
    end = time.time()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("Finished! total time cost:", int(end-start), "s")
        