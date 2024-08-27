
from get_data_utils import *
from torch.utils.data import IterableDataset,get_worker_info,DataLoader, Dataset
from utils import Timers
import torch

def clean_pdf_path(pdf_path):
    return pdf_path[len("opendata:"):] if pdf_path.startswith("opendata:") else pdf_path


class ImageTransformersUtils:
    
    def prepare_for_mfd_model(self, im:np.ndarray):
        if self.mfd_pre_transform is None :return im
        assert im.ndim==3
        im = [im]
        im = np.stack(self.mfd_pre_transform(im))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = im.astype('float')/255
        im = torch.from_numpy(im)
        return im[0]

    def prepare_for_text_det(self,image):
        return self.text_det_transform(image)[0]

class PDFImageDataset(IterableDataset,DatasetUtils,ImageTransformersUtils):
    #client = build_client()
    def __init__(self, metadata_filepath, aug, input_format, 
                 mfd_pre_transform, det_pre_transform=None,
                 return_original_image=False,timer=Timers(False),
                 partion_num = 1,
                 partion_idx = 0):
        super().__init__()
        self.metadata= self.smart_read_json(metadata_filepath)
        self.metadata= np.array_split(self.metadata, partion_num)[partion_idx]
        self.dpi = 200
        self.aug = aug
        self.input_format = input_format
        self.mfd_pre_transform = mfd_pre_transform
        self.return_original_image = return_original_image
        self.det_pre_transform = det_pre_transform
        self.timer = timer
    
    def get_pdf_by_index(self,index):
        pdf_path  = self.metadata[index]['path']
        return self.get_pdf_buffer(pdf_path)

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            self.current_pdf_index = 0
            self.current_page_index = 0
        else:  # in a worker process
            # split workload
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            self.current_pdf_index = worker_id
            self.current_page_index = 0

        self.pdf = self.get_pdf_by_index(self.current_pdf_index)
        return self
    
    @property
    def current_doc(self):
        if len(self.last_read_pdf_buffer)==0:return None
        return list(self.last_read_pdf_buffer.values())[0]

    def prepare_for_text_det(self,image):
        return self.text_det_transform(image)[0]

    def read_data_based_on_current_state(self):
        #print(f"read image from current_page_index={self.current_page_index} ")

        with self.timer("load_page"):
            page  = self.get_pdf_by_index(self.current_pdf_index).load_page(self.current_page_index)
        with self.timer("from_page_to_pimage"):
            oimage = process_pdf_page_to_image(page, self.dpi)
        original_image = oimage[:, :, ::-1] if self.input_format == "RGB" else oimage
        height, width = original_image.shape[:2]
        with self.timer("get_layout_image"):
            layout_image = self.aug.get_transform(original_image).apply_image(original_image)
            layout_image = torch.as_tensor(layout_image.astype("float32").transpose(2, 0, 1))[:,:1042,:800] ## it will be 1043x800 --> 1042:800
        ## lets make sure the image has correct size 
        # if layout_image.size(1) < 1042:
        #     layout_image = torch.nn.functional.pad(layout_image, (0, 0, 0, 1042-layout_image.size(1)))
        with self.timer("get_mfd_image"):
            mfd_image=self.prepare_for_mfd_model(oimage)
        with self.timer("get_det_image"):
            det_images = torch.from_numpy(self.det_pre_transform(original_image)[0])
        
        output= {"pdf_index":self.current_pdf_index, "page_index":self.current_page_index, "mfd_image":mfd_image, "layout_image":layout_image, "det_images":det_images, "height":height, "width":width}
        if self.return_original_image:
            output['oimage'] = original_image
        return output
    
    def go_to_next_pdf(self):
        worker_info = get_worker_info()
        step_for_pdf= 1 if worker_info is None else worker_info.num_workers
         
        self.current_pdf_index += step_for_pdf
        # pdf_path  = self.metadata[self.current_pdf_index]['path']
        # error_count = 0
        # while (not self.check_path_exists(pdf_path) or self.get_pdf_buffer(pdf_path) is None) and error_count<10 :
        #     tqdm.write(f"[Error]: {pdf_path}")
        #     self.current_pdf_index += step_for_pdf
        #     pdf_path  = self.metadata[self.current_pdf_index]['path']
        #     error_count+=1
        # if pdf_path is None:
        #     raise NotImplementedError(f"Seem you use a very bad dataset that we can't find any pdf file, anymore")
        self.current_page_index = 0
        if self.current_pdf_index >= len(self.metadata):
            raise StopIteration
    
    def check_should_skip(self):
        pdf_now = self.get_pdf_by_index(self.current_pdf_index)
        error_count = 0
        while pdf_now is None and error_count<10:
            self.go_to_next_pdf()
            pdf_now = self.get_pdf_by_index(self.current_pdf_index)
            error_count+=1
        if error_count>=10:
            raise NotImplementedError(f"Seem you use a very bad dataset that we can't find any pdf file, anymore")
        current_pdf_page_num = pdf_now.page_count
        if self.current_page_index >= current_pdf_page_num:
            self.go_to_next_pdf()

    def __next__(self):
        
        fail_times = 0
        output = None
        while output is None and fail_times<=10:
            self.check_should_skip()
            try:
                output = self.read_data_based_on_current_state()
                self.current_page_index += 1
                fail_times = 0
            except StopIteration:
                self.clean_pdf_buffer()
                raise StopIteration
            except:
                fail_times +=1
        if fail_times>10 or output is None:
            self.clean_pdf_buffer()
            raise StopIteration
        return output
        
class RecImageDataset(Dataset, DatasetUtils,ImageTransformersUtils):
    error_count=0
    def __init__(self, metadata_filepath,
                 partion_num = 1,
                 partion_idx = 0):
        super().__init__()
        self.metadata= self.smart_read_json(metadata_filepath)
        self.metadata= np.array_split(self.metadata, partion_num)[partion_idx]
        self.dpi = 200
        self.timer = Timers(False)
        self.client = build_client()
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index) :
        pdf_metadata = self.metadata[index]
        return deal_with_one_pdf(pdf_metadata, self.client)

class PageInfoDataset(Dataset,DatasetUtils,ImageTransformersUtils):
    error_count=0
    #client = build_client()
    def __init__(self, metadata_filepath, aug, input_format, mfd_pre_transform, det_pre_transform=None,
                 return_original_image=False,timer=Timers(False),
                 partion_num = 1,
                 partion_idx = 0,
                 page_num_for_name=None):
        super().__init__()
        if page_num_for_name is None:
            filename = metadata_filepath.split("/")[-1].replace('.jsonl','.json')
            page_num_for_name_path = f"opendata:s3://llm-pdf-text/pdf_gpu_output/scihub_shared/page_num_map/{filename}"
            page_num_for_name_list = self.smart_read_json(page_num_for_name_path)
            page_num_for_name={}
            for pdf_path, page_num in page_num_for_name_list:
                if pdf_path.startswith("s3:"): pdf_path = "opendata:"+ pdf_path
                page_num_for_name[pdf_path] = page_num
            tqdm.write(f"we load page_num_for_name from {page_num_for_name_path}")
        metadata= self.smart_read_json(metadata_filepath)
        metadata= np.array_split(metadata, partion_num)[partion_idx]
        tqdm.write("we filte out good metadata")
        self.metadata   = []
        self.pdf_id_and_page_id_pair = []
        for row in metadata:
            if row['path'].startswith("s3:"): row['path'] = "opendata:"+ row['path']
            if row['path'] not in page_num_for_name:continue
            if page_num_for_name[row['path']]<=0:continue
            
            path     = row['path']

            page_num = page_num_for_name[path]
            row['page_num'] = page_num_for_name[path]
            for page_id in range(page_num):
                self.pdf_id_and_page_id_pair.append((len(self.metadata), page_id)) 
            self.metadata.append(row)
        
        self.dpi = 200
        self.aug = aug
        self.input_format = input_format
        self.mfd_pre_transform = mfd_pre_transform
        self.return_original_image = return_original_image
        self.det_pre_transform = det_pre_transform
        self.timer = timer
    
    def __len__(self):
        return len(self.pdf_id_and_page_id_pair)
    
    
    def get_pdf_by_pdf_id(self,pdf_id):
        pdf_path  = self.metadata[pdf_id]['path']
        return self.get_pdf_buffer(pdf_path)
    
    def retreive_resource(self,index):
        current_pdf_index, current_page_index = self.pdf_id_and_page_id_pair[index]
        with self.timer("load_page"):
            page  = self.get_pdf_by_pdf_id(current_pdf_index).load_page(current_page_index)
        with self.timer("from_page_to_pimage"):
            oimage = process_pdf_page_to_image(page, self.dpi)
        original_image = oimage[:, :, ::-1] if self.input_format == "RGB" else oimage
        height, width = original_image.shape[:2]
        with self.timer("get_layout_image"):
            layout_image = self.aug.get_transform(original_image).apply_image(original_image)
            layout_image = torch.as_tensor(layout_image.astype("float32").transpose(2, 0, 1))[:,:1042,:800] ## it will be 1043x800 --> 1042:800
        ## lets make sure the image has correct size 
        # if layout_image.size(1) < 1042:
        #     layout_image = torch.nn.functional.pad(layout_image, (0, 0, 0, 1042-layout_image.size(1)))
        with self.timer("get_mfd_image"):
            mfd_image=self.prepare_for_mfd_model(oimage)
        with self.timer("get_det_image"):
            det_images = torch.from_numpy(self.det_pre_transform(original_image)[0])
        
        output= {"pdf_index":current_pdf_index, "page_index":current_page_index, "mfd_image":mfd_image, "layout_image":layout_image, "det_images":det_images, "height":height, "width":width}
        if self.return_original_image:
            output['oimage'] = original_image
        return output

    def __getitem__(self, index):
        assert self.error_count < 10
        try:
            out = self.retreive_resource(index)
            self.error_count = 0
        except:
            random_index = np.random.randint(0,len(self.pdf_id_and_page_id_pair))
            self.error_count +=1
            out = self[random_index]
        return out 



class AddonDataset(Dataset,DatasetUtils,ImageTransformersUtils):
    error_count = 0
    dpi = 200
    def __init__(self, metadata_filepath,pdfid_pageid_list, aug, input_format, mfd_pre_transform, det_pre_transform=None,
                 return_original_image=False,timer=Timers(False),
                 partion_num = 1,
                 partion_idx = 0):
        super().__init__()
        self.metadata_filepath = metadata_filepath
        self.metadata= self.smart_read_json(metadata_filepath)
        self.pdfid_pageid_list = pdfid_pageid_list
        self.aug = aug
        self.input_format = input_format
        self.mfd_pre_transform = mfd_pre_transform
        self.return_original_image = return_original_image
        self.det_pre_transform = det_pre_transform
        self.timer = timer

    def __len__(self):
        return len(self.pdfid_pageid_list)

    def get_pdf_by_pdf_id(self,pdf_id):
        pdf_path  = self.metadata[pdf_id]['path']
        return self.get_pdf_buffer(pdf_path)
    
    def retreive_resource(self,index):
        current_pdf_index, current_page_index = self.pdfid_pageid_list[index]
        with self.timer("load_page"):
            page  = self.get_pdf_by_pdf_id(current_pdf_index).load_page(current_page_index)
        with self.timer("from_page_to_pimage"):
            oimage = process_pdf_page_to_image(page, self.dpi)
        original_image = oimage[:, :, ::-1] if self.input_format == "RGB" else oimage
        height, width = original_image.shape[:2]
        with self.timer("get_layout_image"):
            layout_image = self.aug.get_transform(original_image).apply_image(original_image)
            layout_image = torch.as_tensor(layout_image.astype("float32").transpose(2, 0, 1))[:,:1042,:800] ## it will be 1043x800 --> 1042:800
        ## lets make sure the image has correct size 
        # if layout_image.size(1) < 1042:
        #     layout_image = torch.nn.functional.pad(layout_image, (0, 0, 0, 1042-layout_image.size(1)))
        with self.timer("get_mfd_image"):
            mfd_image=self.prepare_for_mfd_model(oimage)
        with self.timer("get_det_image"):
            det_images = torch.from_numpy(self.det_pre_transform(original_image)[0])
        
        output= {"pdf_index":current_pdf_index, "page_index":current_page_index, "mfd_image":mfd_image, "layout_image":layout_image, "det_images":det_images, "height":height, "width":width}
        if self.return_original_image:
            output['oimage'] = original_image
        return output

    def __getitem__(self, index):
        assert self.error_count < 10
        try:
            out = self.retreive_resource(index)
            self.error_count = 0
        except:
            current_pdf_index, current_page_index = self.pdfid_pageid_list[index]
            pdf_path = self.metadata[current_pdf_index]['path']
            print(f"fail for pdf={pdf_path} and page={current_page_index}")
            random_index = np.random.randint(0,len(self.pdfid_pageid_list))
            self.error_count +=1
            out = self[random_index]
        return out 
import traceback
def deal_with_one_pdf(pdf_metadata,client):

    images_pool = {}
    pdf_path = pdf_metadata['path']
    if pdf_path.startswith('s3'):
        pdf_path = "opendata:"+pdf_path
    try:
        with read_pdf_from_path(pdf_path, client) as pdf:
            for pdf_page_metadata in pdf_metadata['doc_layout_result']:
                page_id = pdf_page_metadata['page_id']
                page    = pdf.load_page(page_id)
                ori_im  = process_pdf_page_to_image(page, 200)     
                bbox_id = 0
                for bbox_metadata in pdf_page_metadata['layout_dets']:
                    if bbox_metadata['category_id']!=15:continue
                    location= (clean_pdf_path(pdf_path),page_id,bbox_id)
                    tmp_box  = np.array(bbox_metadata['poly']).reshape(-1, 2)
                    tmp_box  = sorted_boxes(tmp_box[None])[0].astype('float32')
                    img_crop = get_rotate_crop_image(ori_im, tmp_box, padding=10)
                    bbox_id+=1
                    images_pool[location] = img_crop

        return (pdf_path,images_pool)
    except KeyboardInterrupt:
        raise
    except:
        traceback.print_exc()
        tqdm.write(f"[Error]: {pdf_path}")
        return (pdf_path,{})

import cv2
from tqdm import tqdm
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
      

def custom_collate_fn(batches):
    
    return_batch = {}
    for batch in batches:
        for key,val in batch.items():
            if key not in return_batch:
                return_batch[key] = []
            return_batch[key].append(val)
    
    keys = list(return_batch.keys())
    for key in keys:
        if key in ["pdf_index", "page_index","height", "width"]:
            return_batch[key] = torch.tensor(return_batch[key])
        elif key in ["mfd_image", "layout_image", "det_images"]:
            return_batch[key] = torch.stack(return_batch[key])
        elif key in ['oimage']:
            return_batch[key] = return_batch[key]
    return return_batch
    
def rec_collate_fn(batches):
    
    location_abs = []
    images_list  = []
    for images_pool in batches:
        for location, image in images_pool.items():
            location_abs.append(location)
            images_list.append(image)
    return location_abs,images_list

def none_collate_fn(batches):
    

    return batches
    