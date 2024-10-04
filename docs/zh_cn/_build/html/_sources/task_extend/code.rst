==================================
代码实现
==================================

PDF-Extract-Kit项目的核心代码实现在pdf_extract_kit目录下，该路径下包含下述几个模块：

- configs: 特定模块的配置文件，如 ``pdf_extract_kit/configs/unimernet.yaml`` ，如果本身配置简单，建议放在 ``repo_root/configs`` 的 ``yaml`` 文件中的 ``model_config`` 里进行定义，方便用户修改。

- dataset: 自定义的 ``ImageDataset`` 类，用于加载和预处理图像数据。它支持多种输入类型，并且可以对图像进行统一的预处理操作（如调整大小、转换为张量等），以便于后续的模型推理加速。

- evaluation: 模型结果评测模块，支持多种任务类型评测，如 ``布局检测`` 、 ``公式检测`` 、 ``公式识别`` 等等，方便用户对不同任务、不同模型进行公平对比。

- registry: ``Registry`` 类是一个通用的注册表类，提供了注册、获取和列出注册项的功能。用户可以使用该类创建不同类型的注册表，例如任务注册表、模型注册表等。

- tasks: 最核心的任务模块，包含了许多不同类型的任务，如 ``布局检测`` 、 ``公式检测`` 、 ``公式识别`` 等等，用户添加新任务和新模型一般仅需要在这里进行代码添加。


.. note::
    基于上述的模块化设计，用户拓展新模块一般只需要在tasks里实现自己的新任务类及对应模型（更多情况下仅需要实现对应模型，任务已经定义好），然后在registry里注册即可。


下面我们以添加基于 ``YOLO``的 ``布局检测`` 模型为例，介绍如何添加新任务和新模型.

任务定义及注册
==============

首先我们在 ``tasks`` 下添加一个 ``layout_detection`` 目录，然后在该目录下添加一个 ``task.py`` 文件用于定义布局检测任务类，具体如下：

.. code-block:: python

    from pdf_extract_kit.registry.registry import TASK_REGISTRY
    from pdf_extract_kit.tasks.base_task import BaseTask


    @TASK_REGISTRY.register("layout_detection")
    class LayoutDetectionTask(BaseTask):
        def __init__(self, model):
            super().__init__(model)

        def predict_images(self, input_data, result_path):
            """
            Predict layouts in images.

            Args:
                input_data (str): Path to a single image file or a directory containing image files.
                result_path (str): Path to save the prediction results.

            Returns:
                list: List of prediction results.
            """
            images = self.load_images(input_data)
            # Perform detection
            return self.model.predict(images, result_path)

        def predict_pdfs(self, input_data, result_path):
            """
            Predict layouts in PDF files.

            Args:
                input_data (str): Path to a single PDF file or a directory containing PDF files.
                result_path (str): Path to save the prediction results.

            Returns:
                list: List of prediction results.
            """
            pdf_images = self.load_pdf_images(input_data)
            # Perform detection
            return self.model.predict(list(pdf_images.values()), result_path, list(pdf_images.keys()))

可以看到，任务定义包含下面几个要点：

* 使用 ``@TASK_REGISTRY.register("layout_detection")`` 语法直接将布局任务类注册到 ``TASK_REGISTRY`` 下 ；
* ``__init__`` 初始化函数传入 ``model`` , 具体参考 ``BaseTask`` 类
* 实现推理函数，这里考虑到布局检测通常会处理图像类及PDF文件，所以提供了两个函数 ``predict_images`` 和 ``predict_pdfs`` ，方便用户灵活选择。

模型定义及注册
==============

接下来我们实现具体模型，在task下面新建models目录，并添加yolo.py用于YOLO模型定义，具体定义如下：

.. code-block:: python

    import os
    import cv2
    import torch
    from torch.utils.data import DataLoader, Dataset
    from ultralytics import YOLO
    from pdf_extract_kit.registry import MODEL_REGISTRY
    from pdf_extract_kit.utils.visualization import  visualize_bbox
    from pdf_extract_kit.dataset.dataset import ImageDataset
    import torchvision.transforms as transforms


    @MODEL_REGISTRY.register('layout_detection_yolo')
    class LayoutDetectionYOLO:
        def __init__(self, config):
            """
            Initialize the LayoutDetectionYOLO class.

            Args:
                config (dict): Configuration dictionary containing model parameters.
            """
            # Mapping from class IDs to class names
            self.id_to_names = {
                0: 'title', 
                1: 'plain text',
                2: 'abandon', 
                3: 'figure', 
                4: 'figure_caption', 
                5: 'table', 
                6: 'table_caption', 
                7: 'table_footnote', 
                8: 'isolate_formula', 
                9: 'formula_caption'
            }

            # Load the YOLO model from the specified path
            self.model = YOLO(config['model_path'])

            # Set model parameters
            self.img_size = config.get('img_size', 1280)
            self.pdf_dpi = config.get('pdf_dpi', 200)
            self.conf_thres = config.get('conf_thres', 0.25)
            self.iou_thres = config.get('iou_thres', 0.45)
            self.visualize = config.get('visualize', False)
            self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            self.batch_size = config.get('batch_size', 1)

        def predict(self, images, result_path, image_ids=None):
            """
            Predict layouts in images.

            Args:
                images (list): List of images to be predicted.
                result_path (str): Path to save the prediction results.
                image_ids (list, optional): List of image IDs corresponding to the images.

            Returns:
                list: List of prediction results.
            """
            results = []
            for idx, image in enumerate(images):
                result = self.model.predict(image, imgsz=self.img_size, conf=self.conf_thres, iou=self.iou_thres, verbose=False)[0]
                if self.visualize:
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)
                    boxes = result.__dict__['boxes'].xyxy
                    classes = result.__dict__['boxes'].cls
                    vis_result = visualize_bbox(image, boxes, classes, self.id_to_names)

                    # Determine the base name of the image
                    if image_ids:
                        base_name = image_ids[idx]
                    else:
                        base_name = os.path.basename(image)
                    
                    result_name = f"{base_name}_MFD.png"
                    
                    # Save the visualized result                
                    cv2.imwrite(os.path.join(result_path, result_name), vis_result)
                results.append(result)
            return results


可以看到，模型定义包含下面几个要点：

* 使用 ``@MODEL_REGISTRY.register('layout_detection_yolo')`` 语法直接将yolo布局模型注册到 ``MODEL_REGISTRY`` 下；
* 初始化函数需要实现：
    + id_to_names的类别映射，用于可视化展示
    + 模型参数配置
    + 模型初始化
* 模型推理函数需要实现多种类型的模型推理：这里支持图像列表和PIL.Image类，可以方便用户直接基于图像路径或者图像流进行推理。

实现上述类定义后，将 ``LayoutDetectionYOLO`` 添加到 ``layout_detection`` 任务下 ``__init__.py`` 的 ``__all__`` 中即可。

.. code-block:: python

    from pdf_extract_kit.tasks.layout_detection.models.yolo import LayoutDetectionYOLO
    from pdf_extract_kit.registry.registry import MODEL_REGISTRY


    __all__ = [
        "LayoutDetectionYOLO",
    ]


.. note:: 
    对于同一个任务，我们支持多种模型，用户具体选择哪个可以根据评测结果进行选择，结合模型 ``精度`` 、 ``速度`` 和 ``场景适配程度`` 进行选择。


实现了任务和模型后，可以在 repo_root/scripts下添加脚本程序 ``layout_detection.py``

示例脚本
==============

.. code-block:: python

    import os
    import sys
    import os.path as osp
    import argparse

    sys.path.append(osp.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from pdf_extract_kit.utils.config_loader import load_config, initialize_tasks_and_models
    import pdf_extract_kit.tasks  # 确保所有任务模块被导入

    TASK_NAME = 'layout_detection'


    def parse_args():
        parser = argparse.ArgumentParser(description="Run a task with a given configuration file.")
        parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
        return parser.parse_args()

    def main(config_path):
        config = load_config(config_path)
        task_instances = initialize_tasks_and_models(config)

        # get input and output path from config
        input_data = config.get('inputs', None)
        result_path = config.get('outputs', 'outputs'+'/'+TASK_NAME)

        # layout_detection_task
        model_layout_detection = task_instances[TASK_NAME]

        # for image detection
        detection_results = model_layout_detection.predict_images(input_data, result_path)

        # for pdf detection
        # detection_results = model_layout_detection.predict_pdfs(input_data, result_path)

        # print(detection_results)
        print(f'The predicted results can be found at {result_path}')


    if __name__ == "__main__":
        args = parse_args()
        main(args.config)

支持类型拓展
==============


批处理拓展
==============
