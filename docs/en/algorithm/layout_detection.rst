.. _algorithm_layout_detection:

=================
Layout Detection Algorithm
=================

Introduction
=================

Layout detection is a fundamental task in document content extraction, aiming to locate different types of regions on a page, such as images, tables, text, and headings, to facilitate high-quality content extraction. For text and heading regions, OCR models can be used for text recognition, while table regions can be converted using table recognition models.

Model Usage
=================

Layout detection supports following models：

.. raw:: html

    <style type="text/css">
    .tg  {border-collapse:collapse;border-color:#9ABAD9;border-spacing:0;}
    .tg td{background-color:#EBF5FF;border-color:#9ABAD9;border-style:solid;border-width:1px;color:#444;
      font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg th{background-color:#409cff;border-color:#9ABAD9;border-style:solid;border-width:1px;color:#fff;
      font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg .tg-f8tz{background-color:#409cff;border-color:inherit;text-align:left;vertical-align:top}
    .tg .tg-0lax{text-align:left;vertical-align:top}
    .tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
    </style>
    <table class="tg"><thead>
      <tr>
        <th class="tg-0lax">Model</th>
        <th class="tg-f8tz">Description</th>
        <th class="tg-f8tz">Characteristics</th>
        <th class="tg-f8tz">Model weight</th>
        <th class="tg-f8tz">Config file</th>
      </tr></thead>
    <tbody>
      <tr>
        <td class="tg-0lax">DocLayout-YOLO</td>
        <td class="tg-0pky">Improved based on YOLO-v10：<br>1. Generate diverse pre-training data，enhance generalization ability across multiple document types<br>2. Model architecture improvement, improve perception ability on scale-varing instances<br>Details in <a href="https://github.com/opendatalab/DocLayout-YOLO" target="_blank" rel="noopener noreferrer">DocLayout-YOLO</a></td>
        <td class="tg-0pky">Speed:Fast, Accuracy:High</td>
        <td class="tg-0pky"><a href="https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/blob/main/models/Layout/YOLO/doclayout_yolo_ft.pt" target="_blank" rel="noopener noreferrer">doclayout_yolo_ft.pt</a></td>
        <td class="tg-0pky">layout_detection.yaml</td>
      </tr>
      <tr>
        <td class="tg-0lax">YOLO-v10</td>
        <td class="tg-0pky">Base YOLO-v10 model</td>
        <td class="tg-0pky">Speed:Fast, Accuracy:Moderate</td>
        <td class="tg-0pky"><a href="https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/blob/main/models/Layout/YOLO/yolov10l_ft.pt" target="_blank" rel="noopener noreferrer">yolov10l_ft.pt</a></td>
        <td class="tg-0pky">layout_detection_yolo.yaml</td>
      </tr>
      <tr>
        <td class="tg-0lax">LayoutLMv3</td>
        <td class="tg-0pky">Base LayoutLMv3 model</td>
        <td class="tg-0pky">Speed:Slow, Accuracy:High</td>
        <td class="tg-0pky"><a href="https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/tree/main/models/Layout/LayoutLMv3" target="_blank" rel="noopener noreferrer">layoutlmv3_ft</a></td>
        <td class="tg-0pky">layout_detection_layoutlmv3.yaml</td>
      </tr>
    </tbody></table>

Once enciroment is setup, you can perform layout detection by executing ``scripts/layout_detection.py`` directly.

**Run demo**

.. code:: shell

   $ python scripts/layout_detection.py --config configs/layout_detection.yaml

Model Configuration
-----------------

**1. DocLayout-YOLO / YOLO-v10**

.. code:: yaml

    inputs: assets/demo/layout_detection
    outputs: outputs/layout_detection
    tasks:
      layout_detection:
        model: layout_detection_yolo
        model_config:
          img_size: 1024
          conf_thres: 0.25
          iou_thres: 0.45
          model_path: path/to/doclayout_yolo_model
          visualize: True

- inputs/outputs: Define the input file path and the directory for visualization output.
- tasks: Define the task type, currently only a layout detection task is included.
- model: Specify the specific model type, e.g., layout_detection_yolo.
- model_config: Define the model configuration.
- img_size: Define the image long edge size; the short edge will be scaled proportionally based on the long edge, with the default long edge being 1024.
- conf_thres: Define the confidence threshold, detecting only targets above this threshold.
- iou_thres: Define the IoU threshold, removing targets with an overlap greater than this threshold.
- model_path: Path to the model weights.
- visualize: Whether to visualize the model results; visualized results will be saved in the outputs directory.


**2. layoutlmv3**

.. note::
   
   LayoutLMv3 cannot run directly by default. Please follow the steps below to modify the configuration:

   1. **Detectron2 Environment Setup**

   .. code-block:: bash

      # For Linux
      pip install https://wheels-1251341229.cos.ap-shanghai.myqcloud.com/assets/whl/detectron2/detectron2-0.6-cp310-cp310-linux_x86_64.whl

      # For macOS
      pip install https://wheels-1251341229.cos.ap-shanghai.myqcloud.com/assets/whl/detectron2/detectron2-0.6-cp310-cp310-macosx_10_9_universal2.whl

      # For Windows
      pip install https://wheels-1251341229.cos.ap-shanghai.myqcloud.com/assets/whl/detectron2/detectron2-0.6-cp310-cp310-win_amd64.whl

   2. **Enable LayoutLMv3 Registration Code**

   Uncomment the lines at the following links:
   
   - `line 2 <https://github.com/opendatalab/PDF-Extract-Kit/blob/main/pdf_extract_kit/tasks/layout_detection/__init__.py#L2>`_
   - `line 8 <https://github.com/opendatalab/PDF-Extract-Kit/blob/main/pdf_extract_kit/tasks/layout_detection/__init__.py#L8>`_

   .. code-block:: python

      from pdf_extract_kit.tasks.layout_detection.models.yolo import LayoutDetectionYOLO
      from pdf_extract_kit.tasks.layout_detection.models.layoutlmv3 import LayoutDetectionLayoutlmv3
      from pdf_extract_kit.registry.registry import MODEL_REGISTRY

      __all__ = [
         "LayoutDetectionYOLO",
         "LayoutDetectionLayoutlmv3",
      ]


.. code:: yaml

    inputs: assets/demo/layout_detection
    outputs: outputs/layout_detection
    tasks:
      layout_detection:
        model: layout_detection_layoutlmv3
        model_config:
          model_path: path/to/layoutlmv3_model

- inputs/outputs: Define the input file path and the directory for visualization output.
- tasks: Define the task type, currently only a layout detection task is included.
- model: Specify the specific model type, e.g., layout_detection_layoutlmv3.
- model_config: Define the model configuration.
- model_path: Path to the model weights.



Diverse Input Support
-----------------

The layout detection script in PDF-Extract-Kit supports input formats such as a ``single image``, a ``directory containing only image files``, a ``single PDF file``, and a ``directory containing only PDF files``.

.. note::

   Modify the path to inputs in configs/layout_detection.yaml according to your actual data format:
   - Single image: path/to/image  
   - Image directory: path/to/images  
   - Single PDF file: path/to/pdf  
   - PDF directory: path/to/pdfs  

.. note::
   When using PDF as input, you need to change ``predict_images`` to ``predict_pdfs`` in ``layout_detection.py``.

   .. code:: python

      # for image detection
      detection_results = model_layout_detection.predict_images(input_data, result_path)

   Change to:

   .. code:: python

      # for pdf detection
      detection_results = model_layout_detection.predict_pdfs(input_data, result_path)

Viewing Visualization Results
-----------------

When ``visualize`` is set to ``True`` in the config file, the visualization results will be saved in the ``outputs`` directory.

.. note::

   Visualization is helpful for analyzing model results, but for large-scale tasks, it is recommended to turn off visualization (set ``visualize`` to ``False`` ) to reduce memory and disk usage.