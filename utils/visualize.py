import os
import shutil
import cv2
from PIL import Image, ImageDraw, ImageFont

from modules.latex2png import tex2pil, zhtext2pil
from utils.config import setup_logging

# Apply the logging configuration
logger = setup_logging('visualize')

color_palette = [
    (255, 64, 255), (255, 255, 0), (0, 255, 255), (255, 215, 135), (215, 0, 95), (100, 0, 48), (0, 175, 0),
    (95, 0, 95), (175, 95, 0), (95, 95, 0),
    (95, 95, 255), (95, 175, 135), (215, 95, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 95, 215),
    (0, 0, 0), (0, 0, 0), (0, 0, 0)
]
id2names = ["title", "plain_text", "abandon", "figure", "figure_caption", "table", "table_caption",
            "table_footnote",
            "isolate_formula", "formula_caption", " ", " ", " ", "inline_formula", "isolated_formula",
            "ocr_text"]

def get_visualize(img_list: list, doc_layout_result, render: bool, output_dir, basename):

    vis_pdf_result = []
    for idx, image in enumerate(img_list):
        single_page_res = doc_layout_result[idx]['layout_dets']
        vis_img = Image.new('RGB', Image.fromarray(image).size, 'white') if render else Image.fromarray(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        draw = ImageDraw.Draw(vis_img)
        for res in single_page_res:
            label = int(res['category_id'])
            if label > 15:  # categories that do not need visualize
                continue
            label_name = id2names[label]
            x_min, y_min = int(res['poly'][0]), int(res['poly'][1])
            x_max, y_max = int(res['poly'][4]), int(res['poly'][5])
            if render and label in [13, 14, 15]:
                try:
                    if label in [13, 14]:  # render formula
                        window_img = tex2pil(res['latex'])[0]
                    else:
                        if True:  # render chinese
                            window_img = zhtext2pil(res['text'])
                        else:  # render english
                            window_img = tex2pil([res['text']], tex_type="text")[0]
                    ratio = min((x_max - x_min) / window_img.width, (y_max - y_min) / window_img.height) - 0.05
                    window_img = window_img.resize(
                        (int(window_img.width * ratio), int(window_img.height * ratio)))
                    vis_img.paste(window_img, (int(x_min + (x_max - x_min - window_img.width) / 2),
                                               int(y_min + (y_max - y_min - window_img.height) / 2)))
                except Exception as e:
                    logger.error(f"got exception on {text}, error info: {e}")

            draw.rectangle([x_min, y_min, x_max, y_max], fill=None, outline=color_palette[label], width=1)
            fontText = ImageFont.truetype("assets/fonts/simhei.ttf", 15, encoding="utf-8")
            draw.text((x_min, y_min), label_name, color_palette[label], font=fontText)

        width, height = vis_img.size
        width, height = int(0.75 * width), int(0.75 * height)
        vis_img = vis_img.resize((width, height))
        vis_pdf_result.append(vis_img)

    first_page = vis_pdf_result.pop(0)
    first_page.save(
        fp=os.path.join(output_dir, f'{basename}.pdf'),
        format='PDF',
        resolution=100,
        save_all=True,
        append_images=vis_pdf_result
    )
    try:
        shutil.rmtree('./temp')
    except Exception as e:
        logger.error(f"got exception on shutil.rmtree, error info: {e}")
        pass