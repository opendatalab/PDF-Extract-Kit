import re

def layout_rm_equation(layout_res):
    rm_idxs = []
    for idx, ele in enumerate(layout_res['layout_dets']):
        if ele['category_id'] == 10:
            rm_idxs.append(idx)
    
    for idx in rm_idxs[::-1]:
        del layout_res['layout_dets'][idx]
    return layout_res

def sorted_layout_boxes(page_layout_res, w):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        page_layout_res(list)
    return: None
    """
    boxes_type = 'single'
    # res = [layout_boxes[i].bbox.bbox for i in range(len(layout_boxes))]
    num_boxes = len(page_layout_res['layout_dets'])
    if num_boxes <= 1:
        # res[0]["layout"] = "single"
        return boxes_type, page_layout_res
    # sort by ymin then xmin
    sorted_boxes = sorted(page_layout_res['layout_dets'], key=lambda x: (x["poly"][1], x["poly"][0]))
    _boxes = list(sorted_boxes)

    new_res = []
    res_left = []
    res_right = []
    i = 0

    while i < num_boxes:
        box_len = max(0, _boxes[i]['poly'][2] - _boxes[i]['poly'][0])
        if box_len == 0:
            new_res += res_left
            new_res += res_right
            new_res.append(_boxes[i])
            res_left = []
            res_right = []
            i += 1
            continue
        if i >= num_boxes:
            break
        if i == num_boxes - 1:
            if (
                    _boxes[i]['poly'][1] > _boxes[i - 1]['poly'][5]
                    and _boxes[i]['poly'][0] < w / 2
                    and _boxes[i]['poly'][2] > w / 2
            ):
                new_res += res_left
                new_res += res_right
                new_res.append(_boxes[i])
            else:
                if _boxes[i]['poly'][2] > w / 2:
                    boxes_type = 'double'
                    res_right.append(_boxes[i])
                    new_res += res_left
                    new_res += res_right
                elif _boxes[i]['poly'][0] < w / 2:
                    res_left.append(_boxes[i])
                    boxes_type = 'double'
                    new_res += res_left
                    new_res += res_right
            # res_left = []
            # res_right = []
            break
        #   box两边距离中线偏移不大，则认为是居中的布局
        # If the distance between the box and the centerline is small, it is considered to be centered.
        elif _boxes[i]['poly'][0] < w / 2 and _boxes[i]['poly'][2] > w / 2 and (
                _boxes[i]['poly'][2] - w / 2) / box_len < 0.65 and (w / 2 - _boxes[i]['poly'][0]) / box_len < 0.65:
            new_res += res_left
            new_res += res_right
            new_res.append(_boxes[i])
            res_left = []
            res_right = []
            i += 1
        elif _boxes[i]['poly'][0] < w / 4 and _boxes[i]['poly'][2] < 3 * w / 4:
            res_left.append(_boxes[i])
            boxes_type = 'double'
            i += 1
        elif _boxes[i]['poly'][0] > w / 4 and _boxes[i]['poly'][2] > w / 2:
            res_right.append(_boxes[i])
            boxes_type = 'double'
            i += 1
        else:
            new_res += res_left
            new_res += res_right
            new_res.append(_boxes[i])
            res_left = []
            res_right = []
            i += 1
    page_layout_res['layout_dets'] = new_res
    return boxes_type


def is_contained(box1, box2, threshold=0.1):
    """
    box1,box2: (xmin,ymin,xmax,ymax)
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    # 不相交直接退出检测
    if b1_x2 < b2_x1 or b1_x1 > b2_x2 or b1_y2 < b2_y1 or b1_y1 > b2_y2:
        return False
    # 计算box2的总面积
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)

    # 计算box1和box2的交集
    intersect_x1 = max(b1_x1, b2_x1)
    intersect_y1 = max(b1_y1, b2_y1)
    intersect_x2 = min(b1_x2, b2_x2)
    intersect_y2 = min(b1_y2, b2_y2)

    # 计算交集的面积
    intersect_area = max(0, intersect_x2 - intersect_x1) * max(0, intersect_y2 - intersect_y1)

    # 计算外面的面积
    b1_outside_area = b1_area - intersect_area
    b2_outside_area = b2_area - intersect_area



    # 计算外面的面积占box2总面积的比例
    ratio_b1 = b1_outside_area / b1_area if b1_area > 0 else 0
    ratio_b2 = b2_outside_area / b2_area if b2_area > 0 else 0

    if ratio_b1 < threshold:
        return 1
    if ratio_b2 < threshold:
        return 2
    # 判断比例是否大于阈值
    return None


def calculate_iou(box1, box2):
    """
    box1,box2: (xmin,ymin,xmax,ymax)
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    # 不相交直接退出检测
    if b1_x2 < b2_x1 or b1_x1 > b2_x2 or b1_y2 < b2_y1 or b1_y1 > b2_y2:
        return 0.0
    # 计算交集
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # 计算并集
    area_box1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_box2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area_box1 + area_box2 - inter_area

    # 避免除零错误，如果区域小到乘积为0,认为是错误识别，直接去掉
    if union_area == 0:
        return 1
        # 检查完全包含
    iou = inter_area / union_area
    return iou


# 将包含关系和重叠关系的box进行过滤，只保留一个
def filter_consecutive_boxes(page_layout_res, iou_threshold=0.92):
    """
    检测布局框列表中包含关系和重叠关系，只保留一个
    LayoutBox.bbox: (xmin,ymin,xmax,ymax)
    """
    idx = set()
    if len(page_layout_res["layout_dets"]) <= 1:
        return
    for i, layout_box in enumerate(page_layout_res["layout_dets"]):
        if i in idx:
            continue
        box1 = (layout_box['poly'][0],layout_box['poly'][1],layout_box['poly'][2],layout_box['poly'][5])
        for j, layout_box2 in enumerate(page_layout_res["layout_dets"]):
            if i == j or j in idx:
                continue
            box2 = (layout_box2['poly'][0],layout_box2['poly'][1],layout_box2['poly'][2],layout_box2['poly'][5])
            # 重叠过多时，选择置信度高的一个
            if calculate_iou(box1, box2) > iou_threshold:
                if layout_box["score"] > layout_box2["score"]:
                    idx.add(i)
                else:
                    idx.add(j)
                continue
            # 包含关系只保留大的识别框
            contained_box_idx = is_contained(box1, box2)
            # 只有置信度够高或者同类型的时候，保留大的识别框
            if contained_box_idx == 1 and (layout_box["score"] >= layout_box2["score"] or layout_box['category_id'] == layout_box2['category_id']):
                idx.add(i)
                break
            elif contained_box_idx == 2 and (layout_box["score"] <= layout_box2["score"] or layout_box['category_id'] == layout_box2['category_id']):
                idx.add(j)
    page_layout_res["layout_dets"] = [layout_box for i, layout_box in enumerate(page_layout_res["layout_dets"]) if i not in idx]

def get_croped_image(image_pil, bbox):
    x_min, y_min, x_max, y_max = bbox
    croped_img = image_pil.crop((x_min, y_min, x_max, y_max))
    return croped_img


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
