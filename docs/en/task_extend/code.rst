==================================
Code Implementation
==================================

The core code of the PDF-Extract-Kit project is implemented in the `pdf_extract_kit` directory, which contains the following modules:

- configs: Configuration files for specific modules, such as `pdf_extract_kit/configs/unimernet.yaml`. If the configuration is simple, it is recommended to define it in the `yaml` file's `model_config` in `repo_root/configs` for easier user modification.

- dataset: A custom `ImageDataset` class used for loading and preprocessing image data. It supports various input types and can perform unified preprocessing operations on images (such as resizing, converting to tensors, etc.) to accelerate subsequent model inference.

- evaluation: A module for evaluating model results, supporting evaluations for various task types such as `layout detection`, `formula detection`, `formula recognition`, etc., allowing users to fairly compare different tasks and models.

- registry: The `Registry` class is a generic registry class that provides functions for registering, retrieving, and listing registered items. Users can use this class to create different types of registries, such as task registries, model registries, etc.

- tasks: The core task module contains many different types of tasks, such as `layout detection`, `formula detection`, `formula recognition`, etc. Users typically only need to add code here to add new tasks and models.

.. note::
    Based on the above modular design, users generally only need to implement their new task class and corresponding model in `tasks` to extend new modules (in most cases, only the corresponding model needs to be implemented, as the task is already defined), and then register it in `registry`.

Below we take adding a YOLO-based `layout detection` model as an example to introduce how to add new tasks and models.

Task Definition and Registration
==============

First, we add a `layout_detection` directory under `tasks`, and then add a `task.py` file in that directory to define the layout detection task class, as follows:

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

As you can see, the task definition includes the following key points:

* Use the `@TASK_REGISTRY.register("layout_detection")` syntax to directly register the layout task class under `TASK_REGISTRY`.
* The `__init__` initialization function takes `model` as an argument, specifically referring to the `BaseTask` class.
* Implement inference functions. Considering that layout detection usually processes images and PDF files, two functions `predict_images` and `predict_pdfs` are provided for users to choose flexibly.

Model Definition and Registration
==============

Next, we implement the specific model by creating a `models` directory under `task` and adding `yolo.py` for YOLO model definition, as follows:

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

As you can see, the model definition includes the following key points:

* Use the `@MODEL_REGISTRY.register('layout_detection_yolo')` syntax to directly register the YOLO layout model under `MODEL_REGISTRY`.
* The initialization function needs to implement:
    + The `id_to_names` category mapping for visualization.
    + Model parameter configuration.
    + Model initialization.
* The model inference function needs to implement various types of model inference: it supports image lists and `PIL.Image` class, allowing users to perform inference directly based on image paths or image streams.

After implementing the above class definition, add `LayoutDetectionYOLO` to the `__all__` in `__init__.py` under the `layout_detection` task.

.. code-block:: python

    from pdf_extract_kit.tasks.layout_detection.models.yolo import LayoutDetectionYOLO
    from pdf_extract_kit.registry.registry import MODEL_REGISTRY

    __all__ = [
        "LayoutDetectionYOLO",
    ]

.. note:: 
    For the same task, we support multiple models. Users can choose which one to use based on evaluation results, considering model `accuracy`, `speed`, and `scenario adaptability`.

After implementing the tasks and models, you can add a script program `layout_detection.py` under `repo_root/scripts`.

Example Script
==============

.. code-block:: python

    import os
    import sys
    import os.path as osp
    import argparse

    sys.path.append(osp.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from pdf_extract_kit.utils.config_loader import load_config, initialize_tasks_and_models
    import pdf_extract_kit.tasks  # Ensure all task modules are imported

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

Support Type Extension
==============

Batch Processing Extension
==============