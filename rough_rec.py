
from get_data_utils import *
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool
from functools import partial
import cv2
from torch.utils.data import Dataset, TensorDataset, DataLoader
from dataaccelerate import DataPrefetcher 
from modules.batch_text_rec import TextRecognizer, rec_args
import torch

try:
    client=build_client()
except:
    client=None

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def get_rotate_crop_image(img, points, padding=10):
    """
    Extracts a rotated and cropped image patch defined by the quadrilateral `points`
    with an additional padding.
    
    Args:
        img (numpy.ndarray): The input image.
        points (numpy.ndarray): A (4, 2) array containing the coordinates of the quadrilateral.
        padding (int): The number of pixels to expand the bounding box on each side.

    Returns:
        numpy.ndarray: The cropped and rotated image patch.
    """
    assert len(points) == 4, "shape of points must be 4*2"
    
    # Calculate the bounding box with padding
    img_height, img_width = img.shape[0:2]
    left = max(0, int(np.min(points[:, 0])) - padding)
    right = min(img_width, int(np.max(points[:, 0])) + padding)
    top = max(0, int(np.min(points[:, 1])) - padding)
    bottom = min(img_height, int(np.max(points[:, 1])) + padding)
    
    # Crop the image with padding
    img_crop = img[top:bottom, left:right, :].copy()
    
    # Adjust points to the new cropped region
    points[:, 0] -= left
    points[:, 1] -= top

    # Calculate the width and height of the rotated crop
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), 
            np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), 
            np.linalg.norm(points[1] - points[2])
        )
    )

    # Define the destination points for perspective transformation
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    
    # Perform the perspective transformation
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img_crop,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    
    # Rotate the image if the height/width ratio is >= 1.5
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    
    return dst_img
      
def build_bbox_group(metadatas):
    width_range = 100
    height_range= 100
    grouped_bboxes = {}
    for pdf_index, pdf_metadata in enumerate(tqdm(metadatas)):
        pdf_path = pdf_metadata['path']
        for pdf_page_metadata in pdf_metadata['doc_layout_result']:
            page_id = pdf_page_metadata['page_id']
            bbox_id = 0
            
            for bbox_metadata in pdf_page_metadata['layout_dets']:
                if bbox_metadata['category_id']!=15:continue
                location= (pdf_path,page_id,bbox_id)
                bbox_id+=1
                bbox = bbox_metadata['poly']
                width, height = calculate_dimensions(bbox)
                width_group   = int(width  // width_range)
                height_group  = int(height // height_range)
                group_key     = (width_group, height_group)
                if group_key not in grouped_bboxes:
                    grouped_bboxes[group_key] = []
                grouped_bboxes[group_key].append((location,bbox))
    return grouped_bboxes

def calculate_dimensions(bbox):
        x_coords = bbox[::2]
        y_coords = bbox[1::2]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        return width, height

def deal_with_one_pdf(pdf_metadata):
    
    images_pool = {}
    pdf_path = pdf_metadata['path']
    try:
        with read_pdf_from_path(pdf_path, client) as pdf:
            for pdf_page_metadata in pdf_metadata['doc_layout_result']:
                page_id = pdf_page_metadata['page_id']
                page    = pdf.load_page(page_id)
                ori_im  = process_pdf_page_to_image(page, 200)     
                bbox_id = 0
                for bbox_metadata in pdf_page_metadata['layout_dets']:
                    if bbox_metadata['category_id']!=15:continue
                    location= (pdf_path,page_id,bbox_id)
                    tmp_box  = np.array(bbox_metadata['poly']).reshape(-1, 2)
                    tmp_box  = sorted_boxes(tmp_box[None])[0].astype('float32')
                    img_crop = get_rotate_crop_image(ori_im, tmp_box, padding=10)
                    bbox_id+=1
                    images_pool[location] = img_crop

        return (pdf_path, images_pool)
    except KeyboardInterrupt:
        raise
    except:
        return (pdf_path, {})
    

def rec_preprocessing(text_recognizer, img_list):
    norm_img_batch = []
    
    resize_norm_img_func = partial(resize_norm_img,
                               max_wh_ratio=max_wh_ratio,
                               rec_image_shape  =text_recognizer.rec_image_shape,
                               limited_max_width=text_recognizer.limited_max_width,
                               limited_min_width=text_recognizer.limited_min_width)
    for img_now in tqdm(img_list, desc="resize and normlized image"):
        norm_img = resize_norm_img_func(img_now)
        norm_img = norm_img[np.newaxis, :]
        norm_img_batch.append(norm_img)
    norm_img_batch = np.concatenate(norm_img_batch)
    # norm_img_batch = norm_img_batch.copy()
    return norm_img_batch

def resize_norm_img(img, max_wh_ratio=None,rec_image_shape=None,limited_max_width=None,limited_min_width=None):
    imgC, imgH, imgW = rec_image_shape
    assert imgC == img.shape[2]
    max_wh_ratio = max(max_wh_ratio, imgW / imgH)
    imgW = int((imgH * max_wh_ratio))
    imgW = max(min(imgW, limited_max_width), limited_min_width)
    h, w = img.shape[:2]
    ratio = w / float(h)
    ratio_imgH = math.ceil(imgH * ratio)
    ratio_imgH = max(ratio_imgH, limited_min_width)
    if ratio_imgH > imgW:
        resized_w = imgW
    else:
        resized_w = int(ratio_imgH)
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im

class UnifiedResizedDataset(Dataset):
    def __init__(self, img_list,rec_image_shape,limited_max_width,limited_min_width):
        max_wh_ratio = 0
        for img_now in img_list:
            # h, w = img_list[ino].shape[0:2]
            h, w = img_now.shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        self.max_wh_ratio = max_wh_ratio
        self.image_list   = img_list
        self.rec_image_shape =rec_image_shape
        self.limited_max_width =limited_max_width
        self.limited_min_width =limited_min_width
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        return resize_norm_img(self.image_list[idx], self.max_wh_ratio, self.rec_image_shape, self.limited_max_width, self.limited_min_width)
def postprocess(self,preds, label=None):
    preds_prob,preds_idx  = preds.max(axis=2)
    text = self.decode(preds_idx.cpu().numpy(), preds_prob.cpu().numpy(), is_remove_duplicate=True)

    if label is None:return text
    label = self.decode(label)
    return text, label

if __name__ == "__main__":
    batch_size = 128
    #dataset    = RecImageDataset("debug.jsonl",tex_recognizer)
    metadata_filepath = "debug.jsonl"
    metadatas = read_json_from_path(metadata_filepath, client)
    print("we are going to processing only the text recognition")
    processes_num = min(64, len(metadatas))
    with Pool(processes=processes_num) as pool:
        image_pool_list = list(tqdm(pool.imap(deal_with_one_pdf, metadatas), total=len(metadatas), desc="Reading whole text image into memory"))
    # image_pool_list = [deal_with_one_pdf(t) for t in tqdm(metadatas, desc="Reading whole text image into memory")]
    no_image_pdf_list = []
    image_pool = {}
    for idx,(pdf_path, image_dict) in enumerate(image_pool_list):
        if len(image_dict)==0:
            no_image_pdf_list.append(pdf_path)
            #print(f"pdf {pdf_path} has no text image")
            continue
        for key,val in image_dict.items():
            image_pool[key]=val
    print(f"we have {len(no_image_pdf_list)} pdfs has no text image")
    print(f"we have {len(image_pool)} text images")
    grouped_bboxes = build_bbox_group(metadatas)
    
    
    tex_recognizer = TextRecognizer(rec_args)
    tex_recognizer.rec_batch_num = batch_size
    #### next step, lets do normlized the bbox to the same size

    location_to_rec = {}
    pbar_whole_images  = tqdm(total=len(image_pool),position=1,leave=False)
    for group_key, location_and_bbox in grouped_bboxes.items():
        if len(location_and_bbox) == 0:continue
        
        img_list_group = [image_pool[location] for location, bbox in location_and_bbox]
        rec_list_group = []
        dataset  = UnifiedResizedDataset(img_list_group, tex_recognizer.rec_image_shape, tex_recognizer.limited_max_width, tex_recognizer.limited_min_width)
        dataloader_group = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, pin_memory_device='cuda')
        featcher   = DataPrefetcher(dataloader_group,device='cuda')
        pbar  = tqdm(total=len(dataloader_group),position=2,leave=False)
        batch = featcher.next()
        while batch is not None:
            inp = batch
            with torch.no_grad():
                prob_out = tex_recognizer.net(inp)
            rec_result = postprocess(tex_recognizer.postprocess_op,prob_out)
            rec_list_group.extend(rec_result)
            pbar.update(1)
            batch = featcher.next()
        assert len(location_and_bbox) == len(rec_list_group)
        for (location, bbox), rec_res in zip(location_and_bbox, rec_list_group):
            location_to_rec[location] = rec_res
        pbar_whole_images.update(len(img_list_group))

    patch_metadata_list = []
    for pdf_index, pdf_metadata in enumerate(tqdm(metadatas)):
        pdf_path = pdf_metadata['path']
        
        patch_metadata = {'path':pdf_path,'doc_layout_result':[]}
        for pdf_page_metadata in pdf_metadata['doc_layout_result']:
            page_id = pdf_page_metadata['page_id']
            bbox_id = 0
            this_line_pool = {'page_id':page_id, 'layout_dets':[]}
            for bbox_metadata in pdf_page_metadata['layout_dets']:
                if bbox_metadata['category_id']!=15:continue
                
                location= (pdf_path,page_id,bbox_id)
                bbox_id+=1
                text, score = location_to_rec[location]
                this_line_pool['layout_dets'].append({'category_id':15, 'text':text, 'score':score})
            patch_metadata['doc_layout_result'].append(this_line_pool)
        patch_metadata_list.append(patch_metadata)
    
    write_json_to_path(patch_metadata_list, metadata_filepath.replace('.jsonl','.patch.rec_result.jsonl'), client)

    # deal_with_one_dataset("debug.jsonl", 
    #                       "debug.stage_1.jsonl", 
    #                       layout_model, mfd_model, ocrmodel=ocrmodel, 
    #                       inner_batch_size=2, batch_size=4,num_workers=4,
    #                       do_text_det = True,
    #                       do_text_rec = True,
    #                       timer=timer)
    # dataset    = PDFImageDataset("part-66210c190659-000035.jsonl",layout_model.predictor.aug,layout_model.predictor.input_format,mfd_pre_transform=None)
    # dataloader = DataLoader(dataset, batch_size=8,collate_fn=custom_collate_fn)  

    
    