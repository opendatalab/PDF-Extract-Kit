import os
import re
import gc
import sys
import time
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from pdf_extract_kit.utils.data_preprocess import load_pdf
from pdf_extract_kit.tasks.ocr.task import OCRTask
from pdf_extract_kit.dataset.dataset import MathDataset
from pdf_extract_kit.registry.registry import TASK_REGISTRY
from pdf_extract_kit.utils.merge_blocks_and_spans import (
    fill_spans_in_blocks,
    fix_block_spans,
    merge_para_with_text
)


def latex_rm_whitespace(s: str):
    """Remove unnecessary whitespace from LaTeX code.
    """
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', s)
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == s:
            break
    return s

def crop_img(input_res, input_pil_img, padding_x=0, padding_y=0):
    crop_xmin, crop_ymin = int(input_res['poly'][0]), int(input_res['poly'][1])
    crop_xmax, crop_ymax = int(input_res['poly'][4]), int(input_res['poly'][5])
    # Create a white background with an additional width and height of 50
    crop_new_width = crop_xmax - crop_xmin + padding_x * 2
    crop_new_height = crop_ymax - crop_ymin + padding_y * 2
    return_image = Image.new('RGB', (crop_new_width, crop_new_height), 'white')

    # Crop image
    crop_box = (crop_xmin, crop_ymin, crop_xmax, crop_ymax)
    cropped_img = input_pil_img.crop(crop_box)
    return_image.paste(cropped_img, (padding_x, padding_y))
    return_list = [padding_x, padding_y, crop_xmin, crop_ymin, crop_xmax, crop_ymax, crop_new_width, crop_new_height]
    return return_image, return_list

@TASK_REGISTRY.register("pdf2markdown")
class PDF2MARKDOWN(OCRTask):
    def __init__(self, layout_model, mfd_model, mfr_model, ocr_model):
        self.layout_model = layout_model
        self.mfd_model = mfd_model
        self.mfr_model = mfr_model
        self.ocr_model = ocr_model
        if self.mfr_model is not None:
            assert self.mfd_model is not None, "formula recognition based on formula detection, mfd_model can not be None."
            self.mfr_transform = transforms.Compose([self.mfr_model.vis_processor, ])
            
        self.color_palette  = {
            'title': (255, 64, 255),
            'plain text': (255, 255, 0),
            'abandon': (0, 255, 255),
            'figure': (255, 215, 135),
            'figure_caption': (215, 0, 95),
            'table': (100, 0, 48),
            'table_caption': (0, 175, 0),
            'table_footnote': (95, 0, 95),
            'isolate_formula': (175, 95, 0),
            'formula_caption': (95, 95, 0),
            'inline': (0, 0, 255),
            'isolated': (0, 255, 0),
            'text': (255, 0, 0)
        }

    def convert_format(self, yolo_res, id_to_names, ):
        """
        convert yolo format to pdf-extract format.
        """
        res_list = []
        for xyxy, conf, cla in zip(yolo_res.boxes.xyxy.cpu(), yolo_res.boxes.conf.cpu(), yolo_res.boxes.cls.cpu()):
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
            new_item = {
                'category_type': id_to_names[int(cla.item())],
                'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                'score': round(float(conf.item()), 2),
            }
            res_list.append(new_item)
        return res_list
    
    
    def process_single_pdf(self, image_list):
        """predict on one image, reture text detection and recognition results.
        
        Args:
            image_list: List[PIL.Image.Image]
            
        Returns:
            List[dict]: list of PDF extract results
            
        Return example:
            [
                {
                    "layout_dets": [
                        {
                            "category_type": "text",
                            "poly": [
                                380.6792698635707,
                                159.85058512958923,
                                765.1419999999998,
                                159.85058512958923,
                                765.1419999999998,
                                192.51073013642917,
                                380.6792698635707,
                                192.51073013642917
                            ],
                            "text": "this is an example text",
                            "score": 0.97
                        },
                        ...
                    ], 
                    "page_info": {
                        "page_no": 0,
                        "height": 2339,
                        "width": 1654,
                    }
                },
                ...
            ]
        """
        pdf_extract_res = []
        mf_image_list = []
        latex_filling_list = []
        for idx, image in enumerate(image_list):
            img_W, img_H = image.size
            if self.layout_model is not None:
                ori_layout_res = self.layout_model.predict([image], "")[0]
                layout_res = self.convert_format(ori_layout_res, self.layout_model.id_to_names)
            else:
                layout_res = []
            single_page_res = {'layout_dets': layout_res}
            single_page_res['page_info'] = dict(
                page_no = idx,
                height = img_H,
                width = img_W
            )
            if self.mfd_model is not None:
                mfd_res = self.mfd_model.predict([image], "")[0]
                for xyxy, conf, cla in zip(mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()):
                    xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                    new_item = {
                        'category_type': self.mfd_model.id_to_names[int(cla.item())],
                        'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                        'score': round(float(conf.item()), 2),
                        'latex': '',
                    }
                    single_page_res['layout_dets'].append(new_item)
                    if self.mfr_model is not None:
                        latex_filling_list.append(new_item)
                        bbox_img = image.crop((xmin, ymin, xmax, ymax))
                        mf_image_list.append(bbox_img)
                    
                pdf_extract_res.append(single_page_res)
                
                del mfd_res
                torch.cuda.empty_cache()
                gc.collect()
            
        # Formula recognition, collect all formula images in whole pdf file, then batch infer them.
        if self.mfr_model is not None:
            a = time.time()
            dataset = MathDataset(mf_image_list, transform=self.mfr_transform)
            dataloader = DataLoader(dataset, batch_size=self.mfr_model.batch_size, num_workers=0)

            mfr_res = []
            for imgs in dataloader:
                imgs = imgs.to(self.mfr_model.device)
                output = self.mfr_model.model.generate({'image': imgs})
                mfr_res.extend(output['pred_str'])
            for res, latex in zip(latex_filling_list, mfr_res):
                res['latex'] = latex_rm_whitespace(latex)
            b = time.time()
            print("formula nums:", len(mf_image_list), "mfr time:", round(b-a, 2))
        
        # ocr_res = self.ocr_model.predict(image)
            
        # ocr and table recognition
        for idx, image in enumerate(image_list):
            layout_res = pdf_extract_res[idx]['layout_dets']
            pil_img = image.copy()

            ocr_res_list = []
            table_res_list = []
            single_page_mfdetrec_res = []

            for res in layout_res:
                if res['category_type'] in self.mfd_model.id_to_names.values():
                    single_page_mfdetrec_res.append({
                        "bbox": [int(res['poly'][0]), int(res['poly'][1]),
                                 int(res['poly'][4]), int(res['poly'][5])],
                    })
                elif res['category_type'] in [self.layout_model.id_to_names[cid] for cid in [0, 1, 2, 4, 6, 7]]:
                    ocr_res_list.append(res)
                elif res['category_type'] in [self.layout_model.id_to_names[5]]:
                    table_res_list.append(res)

            ocr_start = time.time()
            # Process each area that requires OCR processing
            for res in ocr_res_list:
                new_image, useful_list = crop_img(res, pil_img, padding_x=25, padding_y=25)
                paste_x, paste_y, xmin, ymin, xmax, ymax, new_width, new_height = useful_list
                # Adjust the coordinates of the formula area
                adjusted_mfdetrec_res = []
                for mf_res in single_page_mfdetrec_res:
                    mf_xmin, mf_ymin, mf_xmax, mf_ymax = mf_res["bbox"]
                    # Adjust the coordinates of the formula area to the coordinates relative to the cropping area
                    x0 = mf_xmin - xmin + paste_x
                    y0 = mf_ymin - ymin + paste_y
                    x1 = mf_xmax - xmin + paste_x
                    y1 = mf_ymax - ymin + paste_y
                    # Filter formula blocks outside the graph
                    if any([x1 < 0, y1 < 0]) or any([x0 > new_width, y0 > new_height]):
                        continue
                    else:
                        adjusted_mfdetrec_res.append({
                            "bbox": [x0, y0, x1, y1],
                        })

                # OCR recognition
                ocr_res = self.ocr_model.ocr(new_image, mfd_res=adjusted_mfdetrec_res)[0]

                # Integration results
                if ocr_res:
                    for box_ocr_res in ocr_res:
                        p1, p2, p3, p4 = box_ocr_res[0]
                        text, score = box_ocr_res[1]

                        # Convert the coordinates back to the original coordinate system
                        p1 = [p1[0] - paste_x + xmin, p1[1] - paste_y + ymin]
                        p2 = [p2[0] - paste_x + xmin, p2[1] - paste_y + ymin]
                        p3 = [p3[0] - paste_x + xmin, p3[1] - paste_y + ymin]
                        p4 = [p4[0] - paste_x + xmin, p4[1] - paste_y + ymin]

                        layout_res.append({
                            'category_type': 'text',
                            'poly': p1 + p2 + p3 + p4,
                            'score': round(score, 2),
                            'text': text,
                        })

            ocr_cost = round(time.time() - ocr_start, 2)
            print(f"ocr cost: {ocr_cost}")
        return pdf_extract_res
    
    def order_blocks(self, blocks):
        def calculate_oder(poly):
            xmin, ymin, _, _, xmax, ymax, _, _ = poly
            return ymin*3000 + xmin
        return sorted(blocks, key=lambda item: calculate_oder(item['poly']))
                 
    def convert2md(self, extract_res):
        blocks = []
        spans = []

        for item in extract_res['layout_dets']:
            if item['category_type'] in ['inline', 'text', 'isolated']:
                text_key = 'text' if item['category_type'] == 'text' else 'latex'
                xmin, ymin, _, _, xmax, ymax, _, _ = item['poly']
                spans.append(
                    {
                        "type": item['category_type'],
                        "bbox": [xmin, ymin, xmax, ymax],
                        "content": item[text_key]
                    }
                )
                if item['category_type'] == "isolated":
                    item['category_type'] = "isolate_formula"
                    blocks.append(item)
            else:
                blocks.append(item)
                
        blocks_types = ["title", "plain text", "figure_caption", "table_caption", "table_footnote", "isolate_formula", "formula_caption"]

        need_fix_bbox = []
        final_block = []
        for block in blocks:
            block_type = block["category_type"]
            if block_type in blocks_types:
                need_fix_bbox.append(block)
            else:
                final_block.append(block)
                
        block_with_spans, spans = fill_spans_in_blocks(need_fix_bbox, spans, 0.6)
        
        fix_blocks = fix_block_spans(block_with_spans)
        for para_block in fix_blocks:
            result = merge_para_with_text(para_block)
            if para_block['type'] == "isolate_formula":
                para_block['saved_info']['latex'] = result
            else:
                para_block['saved_info']['text'] = result
            final_block.append(para_block['saved_info'])
            
        final_block = self.order_blocks(final_block)
        md_text = ""
        for block in final_block:
            if block['category_type'] == "title":
                md_text += "\n# "+block['text'] +"\n"
            elif block['category_type'] in ["isolate_formula"]:
                md_text += "\n"+block['latex']+"\n"
            elif block['category_type'] in ["plain text", "figure_caption", "table_caption"]:
                md_text += " "+block['text']+" "
            elif block['category_type'] in ["figure", "table"]:
                continue
            else:
                continue
        return md_text
        
    def process(self, input_path, save_dir=None, visualize=False, merge2markdown=False):
        file_list = self.prepare_input_files(input_path)
        res_list = []
        for fpath in file_list:
            basename = os.path.basename(fpath)[:-4]
            if fpath.endswith(".pdf") or fpath.endswith(".PDF"):
                images = load_pdf(fpath)
            else:
                images = [Image.open(fpath)]
            pdf_extract_res = self.process_single_pdf(images)
            res_list.append(pdf_extract_res)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                self.save_json_result(pdf_extract_res, os.path.join(save_dir, f"{basename}.json"))
                
                if merge2markdown:
                    md_content = []
                    for extract_res in pdf_extract_res:
                        md_text = self.convert2md(extract_res)
                        md_content.append(md_text)
                    with open(os.path.join(save_dir, f"{basename}.md"), "w") as f:
                        f.write("\n\n".join(md_content))
                        
                if visualize:
                    for image, page_res in zip(images, pdf_extract_res):
                        self.visualize_image(image, page_res['layout_dets'], cate2color=self.color_palette)
                    if fpath.endswith(".pdf") or fpath.endswith(".PDF"):
                        first_page = images.pop(0)
                        first_page.save(os.path.join(save_dir, f'{basename}.pdf'), 'PDF', resolution=100, save_all=True, append_images=images)
                    else:
                        images[0].save(os.path.join(save_dir, f"{basename}.png"))

        return res_list
        
        
        
        