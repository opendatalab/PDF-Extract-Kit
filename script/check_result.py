import json
from rough_layout import *
from PIL import ImageDraw

with open("snap.json", 'r') as f:
    result = json.load(f)


pdf_path = result["path"]
if pdf_path.startswith("s3"): 
    pdf_path = "opendata:"+ pdf_path
client = build_client() if "s3" in pdf_path else None
pdf = read_pdf_from_path(pdf_path, client)

for chekc_idx in range(len(result['doc_layout_result'])):
    page_id   = result['doc_layout_result'][chekc_idx]['page_id']
    layout    = result['doc_layout_result'][chekc_idx]['layout_dets']
    page = pdf.load_page(page_id)
    oimage = process_pdf_page_to_image(page, 200)
    imagesdata = oimage
    image = Image.fromarray(imagesdata)
    draw = ImageDraw.Draw(image)
    for box in layout:
        poly = box['poly']
        poly_points = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
        draw.polygon(poly_points, outline='red' if box['category_id'] < 13 else 'green')
    # for box in rect_boxes:
    #     x_min, y_min, x_max, y_max = box
    #     # Draw the rectangle
    #     draw.rectangle([x_min, y_min, x_max, y_max], outline='green')    
    image.save(f"test_images/test_{chekc_idx}.png")