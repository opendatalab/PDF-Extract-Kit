..  _algorithm_formula_detection:

====================
Formula Detection Algorithm
====================

Introduction
====================

Formula detection involves identifying the positions of all formulas (including inline and block formulas) in a given input image.

.. note::

   Formula detection is technically a subtask of layout detection. However, due to its complexity, we recommend using a dedicated formula detection model to decouple it. This approach typically makes data annotation easier and improves detection performance.

Model Usage
====================

With the environment properly set up, simply run the layout detection algorithm script by executing ``scripts/formula_detection.py``.

.. code:: shell

   $ python scripts/formula_detection.py --config configs/formula_detection.yaml

Model Configuration
--------------------

.. code:: yaml

   inputs: assets/demo/formula_detection
   outputs: outputs/formula_detection
   tasks:
      formula_detection:
         model: formula_detection_yolo
         model_config:
            img_size: 1280
            conf_thres: 0.25
            iou_thres: 0.45
            batch_size: 1
            model_path: models/MFD/yolov8/weights.pt
            visualize: True

- inputs/outputs: Define the input file path and the visualization output directory, respectively.
- tasks: Define the task type, currently only a formula detection task is included.
- model: Define the specific model type: currently, only the YOLO formula detection model is available.
- model_config: Define the model configuration.
- img_size: Define the image's longer side size; the shorter side will be scaled proportionally.
- conf_thres: Define the confidence threshold; only targets above this threshold will be detected.
- iou_thres: Define the IoU threshold to remove targets with an overlap greater than this value.
- batch_size: Define the batch size; the number of images inferred simultaneously. Generally, the larger the batch size, the faster the inference speed. A better GPU allows for a larger batch size.
- model_path: Path to the model weights.
- visualize: Whether to visualize the model results. Visualized results will be saved in the outputs directory.

Diverse Input Support
--------------------

The formula detection script in PDF-Extract-Kit supports various input formats such as ``a single image``, ``a directory of image files``, ``a single PDF file``, and ``a directory of PDF files``.

.. note:: 

   Modify the ``inputs`` path in ``configs/formula_detection.yaml`` according to your actual data format:
   - Single image: path/to/image  
   - Image directory: path/to/images  
   - Single PDF file: path/to/pdf  
   - PDF directory: path/to/pdfs  

.. note::

   When using a PDF as input, you need to change ``predict_images`` to ``predict_pdfs`` in ``formula_detection.py``.

   .. code:: python

      # for image detection
      detection_results = model_formula_detection.predict_images(input_data, result_path)
   
   Change to:

   .. code:: python

      # for pdf detection
      detection_results = model_formula_detection.predict_pdfs(input_data, result_path)


Viewing Visualization Results
--------------------

When the ``visualize`` option in the config file is set to ``True``, visualization results will be saved in the ``outputs/formula_detection`` directory.

.. note::

   Visualization facilitates the analysis of model results. However, for large-scale tasks, it is recommended to disable visualization (set ``visualize`` to ``False`` ) to reduce memory and disk usage.