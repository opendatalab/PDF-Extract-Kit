import time
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from modules.post_process import latex_rm_whitespace

from utils.config import load_config, setup_logging

# Apply the logging configuration
logger = setup_logging('recognition')


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


def formula_recognition(mf_image_list, latex_filling_list, mfr_model, mfr_transform, batch_size: int = 128):
    # Formula recognition, collect all formula images in whole pdf file, then batch infer them.
    model_configs = load_config()
    device = model_configs['model_args']['device']

    logger.debug('Formula recognition')
    start = time.time()

    dataset = MathDataset(mf_image_list, transform=mfr_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=32)
    mfr_res = []
    for imgs in dataloader:
        imgs = imgs.to(device)
        output = mfr_model.generate({'image': imgs})
        mfr_res.extend(output['pred_str'])
    for res, latex in zip(latex_filling_list, mfr_res):
        res['latex'] = latex_rm_whitespace(latex)

    logger.info(f'formula nums: {len(mf_image_list)} mfr time: {round(time.time() - start, 2)}')


def ocr_table_recognition(img_list: list, doc_layout_result, ocr_model, tr_model):
    model_configs = load_config()
    max_time = model_configs['model_args']['table_max_time']

    logger.debug('ocr & table recognition')
    start = time.time()

    for idx, image in enumerate(img_list):
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        single_page_res = doc_layout_result[idx]['layout_dets']
        single_page_mfdetrec_res = []
        for res in single_page_res:
            if int(res['category_id']) in [13, 14]:  # categories formula
                xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                single_page_mfdetrec_res.append({
                    "bbox": [xmin, ymin, xmax, ymax],
                })
        for res in single_page_res:
            if int(res['category_id']) in [0, 1, 2, 4, 6, 7]:  # categories that need to do ocr
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
            elif int(res['category_id']) == 5:  # do table recognition
                xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                crop_box = [xmin, ymin, xmax, ymax]
                cropped_img = pil_img.convert("RGB").crop(crop_box)

                start = time.time()
                with torch.no_grad():
                    output = tr_model(cropped_img)
                end = time.time()
                if (end - start) > max_time:
                    res["timeout"] = True
                res["latex"] = output[0]

    logger.info(f'ocr and table recognition done in: {round(time.time() - start, 2)}')

    return doc_layout_result


def ocr_recognition(img_list: list, doc_layout_result, ocr_model):
    logger.debug('ocr recognition')
    start = time.time()

    for idx, image in enumerate(img_list):
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        single_page_res = doc_layout_result[idx]['layout_dets']
        single_page_mfdetrec_res = []
        for res in single_page_res:
            if int(res['category_id']) in [13, 14]:  # categories formula
                xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                single_page_mfdetrec_res.append({
                    "bbox": [xmin, ymin, xmax, ymax],
                })
        for res in single_page_res:
            if int(res['category_id']) in [0, 1, 2, 4, 6, 7]:  # categories that need to do ocr
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

    logger.info(f'ocr recognition done in: {round(time.time() - start, 2)}')

    return doc_layout_result


def table_recognition(img_list, doc_layout_result, tr_model):
    model_configs = load_config()
    max_time = model_configs['model_args']['table_max_time']

    logger.debug('table recognition')
    start_0 = time.time()

    for idx, image in enumerate(img_list):
        pil_img = Image.fromarray(image)
        single_page_res = doc_layout_result[idx]['layout_dets']

        for jdx, res in enumerate(single_page_res):
            if int(res['category_id']) == 5:  # do table recognition
                xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                crop_box = [xmin, ymin, xmax, ymax]
                cropped_img = pil_img.crop(crop_box)

                start = time.time()
                with torch.no_grad():
                    start_1 = time.time()
                    output = tr_model(cropped_img) # It takes a lot of time
                    logger.debug(f'{idx} - {jdx} tr_model generate in: {time.time() - start_1}s')

                if (time.time() - start) > max_time:
                    res["timeout"] = True
                res["latex"] = output[0]

    logger.info(f'table recognition done in: {round(time.time() - start_0, 2)}')

    return doc_layout_result