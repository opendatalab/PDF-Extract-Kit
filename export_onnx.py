import argparse
import os
import time

import cv2
import torch
import onnxruntime as ort
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.export import TracingAdapter
from detectron2.modeling import GeneralizedRCNN, build_model
from detectron2.utils.logger import setup_logger
import detectron2.data.transforms as T
import numpy as np

from modules.layoutlmv3.model_init import setup
from modules.post_process import filter_consecutive_boxes, sorted_layout_boxes
# pip install onnxruntime
def resize_boxes(boxes, old_size, new_size):
    old_width,old_height = old_size
    new_width,new_height = new_size
    scale_x, scale_y = (
        old_width / new_width,
        old_height / new_height,
    )
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    boxes[:, 0::2] *= scale_x
    boxes[:, 1::2] *= scale_y
    x1 = np.clip(boxes[:, 0], a_min=0, a_max=old_width)
    y1 = np.clip(boxes[:, 1], a_min=0, a_max=old_height)
    x2 = np.clip(boxes[:, 2], a_min=0, a_max=old_width)
    y2 = np.clip(boxes[:, 3], a_min=0, a_max=old_height)

    # 使用 numpy 的 stack 函数
    return np.stack((x1, y1, x2, y2), axis=-1)
def resize_image(image, min_size=800, max_size=1333):
    """
    调整图片大小，确保宽度和高度都满足最小值和最大值的要求。

    :param image: 输入的 BGR 图片 numpy 数组
    :param min_size: 最小尺寸
    :param max_size: 最大尺寸
    :return: 调整大小后的图片 numpy 数组
    """
    # 获取原始图片的宽高
    height, width = image.shape[:2]
    if height >= min_size and width >= min_size and width <= max_size and height <= max_size:
        return image
    else:
        # 计算缩放比例
        scale_w = min_size / width
        scale_h = min_size / height
        scale = min(scale_w, scale_h)

        # 检查是否超过最大尺寸
        if width * scale > max_size or height * scale > max_size:
            scale = max_size / max(width, height)

        # 缩放图片
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image
def layout_box_order_render_with_label(layout_boxes, page_img_file, output_path=None):
    id2names = ["title", "plain_text", "abandon", "figure", "figure_caption", "table", "table_caption", "table_footnote",
                "isolate_formula", "formula_caption", " ", " ", " ", "inline_formula", "isolated_formula", "ocr_text"]
    img = cv2.imread(page_img_file)
    # 检查图像是否成功读取
    if img is None:
        raise IOError(f"Failed to load image file: {page_img_file}")

    for idx, box in enumerate(layout_boxes):
        x0, y0, x1, y1 = box["poly"][0], box["poly"][1], box["poly"][4], box["poly"][5]
        x0 = round(x0)
        y0 = round(y0)
        x1 = round(x1)
        y1 = round(y1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)

        # 增大字体大小和线宽
        font_scale = 1.0  # 原先是0.5
        thickness = 2  # 原先是1

        cv2.putText(
            img,
            str(idx) + '.' + id2names[box['category_id']],
            (x1, y1),
            cv2.FONT_HERSHEY_PLAIN,
            font_scale,
            (0, 0, 255),
            thickness,
            )

        # 修改文件名，添加_output后缀
    base_name = os.path.splitext(os.path.basename(page_img_file))[0]
    ext = os.path.splitext(page_img_file)[1]
    new_file_name = f"{base_name}_output{ext}"
    if output_path is not None:
        save_path = os.path.join(output_path, new_file_name)
    else:
        save_path = os.path.join(os.path.dirname(page_img_file), new_file_name)

    cv2.imwrite(save_path, img)
def export_tracing(torch_model, inputs, output, logger):
    image = inputs[0]["image"]
    inputs = [{"image": image}]  # remove other unused keys
    if isinstance(torch_model, GeneralizedRCNN):
        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly
    traceable_model = TracingAdapter(torch_model, inputs, inference)
    device = torch.device('cpu')  # 使用 CPU 进行校准
    # 保存优化后的模型
    with open(output, "wb") as f:
        torch.onnx.export(traceable_model, (image,), f,
                          opset_version=16,
                          export_params=True,
                          do_constant_folding=True,
                          input_names=["image"],
                          output_names=['pred_boxes', 'labels', 'pred_masks', 'scores', 'img_size'],  # 输出名称
                          dynamic_axes={
                              'image': {0: 'channel', 1: 'height', 2: 'width'},  # 动态轴设置
                              'pred_boxes': {0: 'num_boxes'},  # 第一个输出的第一个维度为动态
                              'labels': {0: 'num_boxes'},  # 第二个输出为动态
                              'pred_masks': {0: 'num_boxes', 2: 'num_boxes', 3: 'num_boxes'},  # 第三个输出的多个维度为动态
                              'scores': {0: 'num_boxes'},  # 第四个输出为动态
                              'img_size': {0: 'fixed_height', 1: 'fixed_width'}  # 第五个输出为固定形状
                          }
                          )
def get_img_inputs(sample_image, min_size, max_size):
    # get a sample data
    original_image = cv2.imread(sample_image)
    # Do same preprocessing as DefaultPredictor
    aug = T.ResizeShortestEdge([min_size, min_size], max_size)
    height, width = original_image.shape[:2]
    image = resize_image(original_image, min_size, max_size)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = {"image": image, "height": height, "width": width}
    # Sample ready
    sample_inputs = [inputs]
    return sample_inputs

def test_onnx(image_path, min_size, max_size, model_path, output_path=None):
    # 读取图像
    inputs = get_img_inputs(image_path, min_size, max_size)
    ori_height, ori_width = inputs[0]["height"], inputs[0]["width"]
    image = inputs[0]['image'].numpy()
    image = image.astype("float32")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    # 运行模型
    height, width = image.shape[1:3]
    # # 改变维度顺序
    start = time.time()
    outputs = session.run(['pred_boxes', 'labels', 'scores'], {
        "image": image
    })
    end = time.time()
    print(f"inference time: {end - start}")
    #  转换为本项目统一的layout输出格式
    page_layout_result = outpust_adapter(height, ori_height, ori_width, outputs, width)
    # 过滤重叠和覆盖的检测框
    filter_consecutive_boxes(page_layout_result)
    # 检测框排序
    sorted_layout_boxes(page_layout_result, ori_width)
    # 去除所有页眉页脚
    label_boxes = [box for _, box in enumerate(page_layout_result['layout_dets']) if box['category_id'] != 2]
    # 按顺序渲染
    layout_box_order_render_with_label(label_boxes, image_path, output_path)


def outpust_adapter(height, ori_height, ori_width, outputs, width):
    boxes = resize_boxes(outputs[0], (ori_width, ori_height), (width, height))
    labels = outputs[1]
    scores = outputs[2]
    page_layout_result = {
        "layout_dets": []
    }
    for bbox_idx in range(len(boxes)):
        if scores[bbox_idx] < 0.5:
            continue
        page_layout_result["layout_dets"].append({
            "category_id": labels[bbox_idx],
            "poly": [
                boxes[bbox_idx][0], boxes[bbox_idx][1],
                boxes[bbox_idx][2], boxes[bbox_idx][1],
                boxes[bbox_idx][2], boxes[bbox_idx][3],
                boxes[bbox_idx][0], boxes[bbox_idx][3],
            ],
            "score": scores[bbox_idx]
        })
    return page_layout_result

def export_and_quantize(args):
    args.opts = ["MODEL.WEIGHTS", args.MODEL_WEIGHTS]
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cfg = setup(args)
    # create a torch model
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
    torch_model.eval()
    inputs = get_img_inputs(args.sample_image, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
    export_tracing(torch_model, inputs, args.output, logger)
    if args.quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
        quant_pre_process(args.output, args.output, auto_merge=True)
        quantize_dynamic(args.output, args.output, weight_type=QuantType.QUInt8,
                         op_types_to_quantize=["MatMul", "Conv", "Gather", "Add"], )



if __name__ == "__main__":
    """
    for export onnx and quantize
    python .\export_onnx.py --sample_image ./demo/Layout/demo.jpg --quantize 
    for test onnx model
    python .\export_onnx.py --test_img ./demo/Layout/demo.jpg --model_path ./output/model.onnx
    """
    parser = argparse.ArgumentParser(description='Export and Test ONNX Model')
    parser.add_argument('--sample_image', type=str, help='Path to the sample image')
    parser.add_argument('--output', type=str, default='output/model.onnx', help='Output ONNX model path')
    parser.add_argument('--quantize', action='store_true', help='Enable quantize')
    parser.add_argument('--test_img', type=str, default=None, help='Path to the test image')
    parser.add_argument('--model_path', type=str, default="./output/model.onnx", help='Path to the test image')
    parser.add_argument('--config_file', type=str, default='modules/layoutlmv3/layoutlmv3_base_inference.yaml', help='Path to the config file')
    parser.add_argument('--MODEL_WEIGHTS', type=str, default='models/Layout/model_final.pth', help='Path to the model weights')
    args = parser.parse_args()
    # 根据命令行参数调用相应的函数
    if args.test_img:
        test_onnx(args.test_img, 800, 1333, args.model_path)
    else:
        export_and_quantize(args)
