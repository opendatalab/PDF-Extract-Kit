import torch
import gc
import time
from PIL import Image

from modules.post_process import get_croped_image
from utils.config import load_config, setup_logging

# Apply the logging configuration
logger = setup_logging('detection')


# layout detection and formula detection
def layout_detection_and_formula(img_list, layout_model, mfd_model):
    # img_list es similar a pag_pdf

    model_configs = load_config()
    img_size = model_configs['model_args']['img_size']
    conf_thres = model_configs['model_args']['conf_thres']
    iou_thres = model_configs['model_args']['iou_thres']

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
            page_no=idx,
            height=img_H,
            width=img_W
        )
        doc_layout_result.append(layout_res)

        del mfd_res
        torch.cuda.empty_cache()
        gc.collect()

    return doc_layout_result, latex_filling_list, mf_image_list


def layout_detection(img_list, layout_model):
    doc_layout_result = []

    logger.debug('layout detection - init')
    start = time.time()

    for idx, image in enumerate(img_list):
        img_H, img_W = image.shape[0], image.shape[1]

        layout_res = layout_model(image, ignore_catids=[])

        layout_res['page_info'] = dict(
            page_no=idx,
            height=img_H,
            width=img_W
        )
        doc_layout_result.append(layout_res)

        del layout_res
        torch.cuda.empty_cache()
        gc.collect()

    logger.debug(f'Layout detection done in {round(time.time() - start, 2)}s!')

    return doc_layout_result


def formula_detection(img_list, doc_layout_result, mfd_model):
    model_configs = load_config()
    img_size = model_configs['model_args']['img_size']
    conf_thres = model_configs['model_args']['conf_thres']
    iou_thres = model_configs['model_args']['iou_thres']

    logger.debug('formula detection - init')
    start = time.time()

    latex_filling_list = []
    mf_image_list = []
    for idx, image in enumerate(img_list):

        mfd_res = mfd_model.predict(image, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=True)[0]

        for xyxy, conf, cla in zip(mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()):
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
            new_item = {
                'category_id': 13 + int(cla.item()),
                'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                'score': round(float(conf.item()), 2),
                'latex': '',
            }
            doc_layout_result[idx]['layout_dets'].append(new_item)
            latex_filling_list.append(new_item)
            bbox_img = get_croped_image(Image.fromarray(image), [xmin, ymin, xmax, ymax])
            mf_image_list.append(bbox_img)

        del mfd_res
        torch.cuda.empty_cache()
        gc.collect()

    logger.debug(f'Formula detection done in {round(time.time() - start, 2)}s!')

    return doc_layout_result, latex_filling_list, mf_image_list
