import numpy as np
import copy
import torch
import cv2
import torch
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

def formula_in_text(mf_bbox, text_bbox):
    x1, y1, x2, y2 = mf_bbox
    x3, y3 = text_bbox[0]
    x4, y4 = text_bbox[2]
    left_box, right_box = None, None
    same_line = abs((y1+y2)/2 - (y3+y4)/2) / abs(y4-y3) < 0.2
    if not same_line:
        return False, left_box, right_box
    else:
        drop_origin = False
        left_x = x1 - 1
        right_x = x2 + 1
        if x3 < x1 and x2 < x4:
            drop_origin = True
            left_box = np.array([text_bbox[0], [left_x, text_bbox[1][1]], [left_x, text_bbox[2][1]], text_bbox[3]]).astype('float32')
            right_box = np.array([[right_x, text_bbox[0][1]], text_bbox[1], text_bbox[2], [right_x, text_bbox[3][1]]]).astype('float32')
        if x3 < x1 and x1 <= x4 <= x2:
            drop_origin = True
            left_box = np.array([text_bbox[0], [left_x, text_bbox[1][1]], [left_x, text_bbox[2][1]], text_bbox[3]]).astype('float32')
        if x1 <= x3 <= x2 and x2 < x4:
            drop_origin = True
            right_box = np.array([[right_x, text_bbox[0][1]], text_bbox[1], text_bbox[2], [right_x, text_bbox[3][1]]]).astype('float32')
        if x1 <= x3 < x4 <= x2:
            drop_origin = True
        return drop_origin, left_box, right_box


def update_det_boxes(dt_boxes, mfdetrec_res):
    new_dt_boxes = dt_boxes
    for mf_box in mfdetrec_res:
        flag, left_box, right_box = False, None, None
        for idx, text_box in enumerate(new_dt_boxes):
            ret, left_box, right_box = formula_in_text(mf_box['bbox'], text_box)
            if ret:
                new_dt_boxes.pop(idx)
                if left_box is not None:
                    new_dt_boxes.append(left_box)
                if right_box is not None:
                    new_dt_boxes.append(right_box)
                break
            
    return new_dt_boxes

def get_gpu_memory():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # Convert bytes to MB
        return int(total_memory)
    return 0


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

def collect_text_image_and_its_coordinate(single_page_mfdetrec_res_this_batch, partition_per_batch, oimages, dt_boxes_list):
    text_image_batch = []
    text_image_position=[]
    text_line_bbox = []
    for partition_id, single_page_mfdetrec_res in enumerate(single_page_mfdetrec_res_this_batch):
        partition_start = partition_per_batch[partition_id]
        partition_end   = partition_per_batch[partition_id+1]
        #print(partition_start,partition_end)
        dt_boxes_per_page = dt_boxes_list[partition_start:partition_end]
        for text_box_id, dt_boxes in enumerate(dt_boxes_per_page):
            ori_im   = oimages[partition_id]
            height, width, _ = ori_im.shape
            dt_boxes = sorted_boxes(dt_boxes)
            dt_boxes = update_det_boxes(dt_boxes, single_page_mfdetrec_res)
            for bno in range(len(dt_boxes)):
                tmp_box = copy.deepcopy(dt_boxes[bno])
                text_line_bbox.append(tmp_box)
                img_crop = get_rotate_crop_image(ori_im, copy.deepcopy(tmp_box), padding=10)
                text_image_batch.append(img_crop)
                text_image_position.append((partition_id,text_box_id,bno))
                
    return text_image_batch, text_image_position,text_line_bbox


def collect_mfdetrec_res_per_page(single_page_res):
    single_page_mfdetrec_res = []
    for res in single_page_res:
        if int(res['category_id']) in [13, 14]:
            xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
            xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
            single_page_mfdetrec_res.append({"bbox": [xmin, ymin, xmax, ymax]})
    return single_page_mfdetrec_res

def collect_image_tensor_cropped(oimage:np.ndarray, single_page_res, scale=1):
    image_np    = oimage
    canvas_list = []
    canvas_idxes= [] 
    for bbox_id, res in enumerate(single_page_res):
        if int(res['category_id']) in [0, 1, 2, 4, 6, 7]:  #需要进行ocr的类别
            xmin, ymin = int(res['poly'][0]/scale), int(res['poly'][1]/scale)
            xmax, ymax = int(res['poly'][4]/scale), int(res['poly'][5]/scale)
            if isinstance(image_np, np.ndarray):
                canvas = image_np.ones_like(image_np) * 255
            else:
                canvas = torch.ones_like(image_np) * image_np[0,0,0]
            if canvas.shape[0]==3:
                canvas[:,ymin:ymax, xmin:xmax] = image_np[:,ymin:ymax, xmin:xmax]
            elif canvas.shape[2]==3:
                canvas[ymin:ymax, xmin:xmax,:] = image_np[ymin:ymax, xmin:xmax,:]
            else:
                raise ValueError("image shape is not 3 or 4")
            canvas_list.append(canvas)
            canvas_idxes.append(bbox_id)
    return canvas_list, canvas_idxes 
    
def collect_paragraph_image_and_its_coordinate(oimages, rough_layout_this_batch,scale=1):
    canvas_tensor_this_batch = []
    canvas_idxes_this_batch  = []
    single_page_mfdetrec_res_this_batch = []
    partition_per_batch = [0]
    for oimage, single_page_res in zip(oimages, rough_layout_this_batch):
        single_page_mfdetrec_res = collect_mfdetrec_res_per_page(single_page_res)
        canvas_tensor, canvas_idxes = collect_image_tensor_cropped(oimage, single_page_res,scale=scale)
        canvas_tensor_this_batch.extend(canvas_tensor)
        canvas_idxes_this_batch.append(canvas_idxes)
        single_page_mfdetrec_res_this_batch.append(single_page_mfdetrec_res)
        partition_per_batch.append(len(canvas_tensor_this_batch))
    return canvas_tensor_this_batch, partition_per_batch,canvas_idxes_this_batch,single_page_mfdetrec_res_this_batch

def collect_paragraph_image_and_its_coordinate_from_detection_batch(detection_images, rough_layout_this_batch):
    canvas_tensor_this_batch = []
    canvas_idxes_this_batch  = []
    single_page_mfdetrec_res_this_batch = []
    partition_per_batch = [0]
    for oimage, single_page_res in zip(detection_images, rough_layout_this_batch):
        single_page_mfdetrec_res = collect_mfdetrec_res_per_page(single_page_res)
        canvas_tensor, canvas_idxes = collect_image_tensor_cropped(oimage, single_page_res)
        canvas_tensor_this_batch.extend(canvas_tensor)
        canvas_idxes_this_batch.append(canvas_idxes)
        single_page_mfdetrec_res_this_batch.append(single_page_mfdetrec_res)
        partition_per_batch.append(len(canvas_tensor_this_batch))
    return canvas_tensor_this_batch, partition_per_batch,canvas_idxes_this_batch,single_page_mfdetrec_res_this_batch

def convert_boxes(boxes, original_width, original_height, target_width, target_height):
    """Convert bounding boxes to a new resolution."""
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height
    
    converted_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        new_x_min = x_min * width_ratio
        new_y_min = y_min * height_ratio
        new_x_max = x_max * width_ratio
        new_y_max = y_max * height_ratio
        converted_boxes.append((new_x_min, new_y_min, new_x_max, new_y_max))
    
    return converted_boxes

import time, math
class _DummyTimer:
    """A dummy timer that does nothing."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name = name
        self.count = 0
        self.mean = 0.0
        self.sum_squares = 0.0
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        elapsed_time = time.time() - self.start_time
        self.update(elapsed_time)
        self.start_time = None

    def update(self, elapsed_time):
        self.count += 1
        delta = elapsed_time - self.mean
        self.mean += delta / self.count
        delta2 = elapsed_time - self.mean
        self.sum_squares += delta * delta2

    def mean_elapsed(self):
        return self.mean

    def std_elapsed(self):
        if self.count > 1:
            variance = self.sum_squares / (self.count - 1)
            return math.sqrt(variance)
        else:
            return 0.0

class Timers:
    """Group of timers."""

    def __init__(self, activate=False,warmup=0):
        self.timers = {}
        self.activate = activate
        self.warmup = warmup
        self.count_per_name={}
    def __call__(self, name):
        if not self.activate:return _DummyTimer()
        if name not in self.timers:
            self.timers[name] = _Timer(name)
            self.count_per_name[name] = -1
        self.count_per_name[name]+=1
        if self.count_per_name[name] < self.warmup:
            return _DummyTimer()

        return self.timers[name]

    def log(self, names=None, normalizer=1.0):
        if not self.activate:return
        """Log a group of timers."""
        assert normalizer > 0.0
        if names is None:
            names = self.timers.keys()
        print("Timer Results:")
        for name in names:
            mean_elapsed = self.timers[name].mean_elapsed() * 1000.0 / normalizer
            std_elapsed = self.timers[name].std_elapsed() * 1000.0 / normalizer
            space_num = " "*name.count('/')
            print(f"{space_num}{name}: {mean_elapsed:.2f}±{std_elapsed:.2f} ms")
