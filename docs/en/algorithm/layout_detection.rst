.. _algorithm_layout_detection:

=================
Layout Detection Algorithm
=================

Introduction
=================

Layout detection is a fundamental task in document content extraction, aiming to locate different types of regions on a page, such as images, tables, text, and headings, to facilitate high-quality content extraction. For text and heading regions, OCR models can be used for text recognition, while table regions can be converted using table recognition models.

Model Usage
=================

The layout detection model supports ``YOLOv10``, ``DocLayout-YOLO`` and ``LayoutLMv3``. Once the environment is set up, you can run the layout detection algorithm script by executing ``scripts/layout_detection.py``.

**Run demo**

.. code:: shell

   $ python scripts/layout_detection.py --config configs/layout_detection.yaml

Model Configuration
-----------------

**1. yolov10**

Compared to LayoutLMv3, YOLOv10 has faster inference speed and supports batch mode inference.

.. code:: yaml

    inputs: assets/demo/layout_detection
    outputs: outputs/layout_detection
    tasks:
      layout_detection:
        model: layout_detection_yolo
        model_config:
          img_size: 1280
          conf_thres: 0.25
          iou_thres: 0.45
          batch_size: 2
          model_path: path/to/yolov10_model
          visualize: True
          rect: True
          device: "0"

- inputs/outputs: Define the input file path and the directory for visualization output.
- tasks: Define the task type, currently only a layout detection task is included.
- model: Specify the specific model type, e.g., layout_detection_yolo.
- model_config: Define the model configuration.
- img_size: Define the image long edge size; the short edge will be scaled proportionally based on the long edge, with the default long edge being 1280.
- conf_thres: Define the confidence threshold, detecting only targets above this threshold.
- iou_thres: Define the IoU threshold, removing targets with an overlap greater than this threshold.
- batch_size: Define the batch size, the number of images inferred simultaneously during inference. Generally, the larger the batch size, the faster the inference speed; a better GPU allows for a larger batch size.
- model_path: Path to the model weights.
- visualize: Whether to visualize the model results; visualized results will be saved in the outputs directory.
- rect: Whether to enable rectangular inference, default is True. If set to True, images in the same batch will be scaled while maintaining aspect ratio and padded to the same size; if False, all images in the same batch will be resized to (img_size, img_size) for inference.


**2. layoutlmv3**

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
   When using PDF as input, you need to change ``predict_images`` to ``predict_pdfs`` in ``formula_detection.py``.

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