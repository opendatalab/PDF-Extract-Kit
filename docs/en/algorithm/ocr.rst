..  _algorithm_ocr:
==========================
OCR (Optical Character Recognition) Algorithm
==========================

Introduction
====================

OCR(Optical Character Recognition) involves identifying the positions ajnd contents of all text blocks in pictures.


Model Usage
====================

With the environment properly set up, simply run the ocr algorithm script by executing ``scripts/ocr.py`` .

.. code:: shell

   $ python scripts/ocr.py --config configs/ocr.yaml


Model Configuration
--------------------

.. code:: yaml

   inputs: assets/demo/ocr
   outputs: outputs/ocr
   visualize: True
   tasks:
      ocr:
         model: ocr_ppocr
         model_config:
            lang: ch
            show_log: True
            det_model_dir: models/OCR/PaddleOCR/det/ch_PP-OCRv4_det
            rec_model_dir: models/OCR/PaddleOCR/rec/ch_PP-OCRv4_rec
            det_db_box_thresh: 0.3

- inputs/outputs: Define the input path and the output path, respectively.
- visualize: Whether to visualize the model results. Visualized results will be saved in the outputs directory.
- tasks: Define the task type, currently only a OCR task is included.
- model: Define the specific model type, currently, only the PaddleOCR model is available.
- model_config: Define the model configuration.
- lang: Define the language, default language ch supports both english and chinese.
- show_log: Whether to print running logs.
- det_model_dir: Define the path of PaddleOCR' detection model, If the specified path does not exist, the model weight will be automatically downloaded to the path.
- rec_model_dir: Define the path of PaddleOCR' recognize model, If the specified path does not exist, the model weight will be automatically downloaded to the path.
- det_db_box_thresh: Confidence filter threshold, bounding boxes whose confidence is lower than the threshold are discarded.


Diverse Input Support
--------------------

The OCR script in PDF-Extract-Kit supports various input formats such as ``a single image/PDF``, ``a directory of image/PDF files``.


Viewing Visualization Results
--------------------

When the ``visualize`` option in the config file is set to ``True``, visualization results will be saved in the ``outputs`` directory.

.. note::

   Visualization facilitates the analysis of model results. However, for large-scale tasks, it is recommended to disable visualization (set ``visualize`` to ``False`` ) to reduce memory and disk usage.