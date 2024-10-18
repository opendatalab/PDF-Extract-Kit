=================
Document Content Extraction Project
=================

Introduction
====================

Document content extraction aiming to extract all information of document file and convert it to computer readable result(such as markdown file). It's subtasks including layout detection, formula detection, formula recognition, OCR and other tasks.


Project Usage
====================

With the environment properly set up, simply run the project by executing ``project/pdf2markdown/scripts/run_project.py`` .

.. code:: shell

   $ python project/pdf2markdown/scripts/run_project.py --config project/pdf2markdown/configs/pdf2markdown.yaml


Project Configuration
--------------------

.. code:: yaml

    inputs: assets/demo/formula_detection
    outputs: outputs/pdf2markdown
    visualize: True
    merge2markdown: True
    tasks:
        layout_detection:
            model: layout_detection_yolo
            model_config:
                img_size: 1024
                conf_thres: 0.25
                iou_thres: 0.45
                model_path: models/Layout/YOLO/doclayout_yolo_ft.pt
        formula_detection:
            model: formula_detection_yolo
            model_config:
                img_size: 1280
                conf_thres: 0.25
                iou_thres: 0.45
                batch_size: 1
                model_path: models/MFD/YOLO/yolo_v8_ft.pt
        formula_recognition:
            model: formula_recognition_unimernet
            model_config:
                batch_size: 128
                cfg_path: pdf_extract_kit/configs/unimernet.yaml
                model_path: models/MFR/unimernet_tiny
        ocr:
            model: ocr_ppocr
            model_config:
                lang: ch
                show_log: True
                det_model_dir: models/OCR/PaddleOCR/det/ch_PP-OCRv4_det
                rec_model_dir: models/OCR/PaddleOCR/rec/ch_PP-OCRv4_rec
                det_db_box_thresh: 0.3

- inputs/outputs: Define the input path and the output path, respectively.
- visualize: Whether to visualize the project results. Visualized results will be saved in the outputs directory.
- merge2markdown: Whether to merge the results into markdown documents. Only simple single-column text is supported. For markdown conversion of more complex layout documents, please refer to `MinerU <https://github.com/opendatalab/MinerU>`_ .
- tasks: Define the task types, PDF document extraction includes layout detection, formula detection, formula recognition, and OCR tasks.
- For details about the parameter meanings of each task and model, see the tutorial documentation of each task.


Diverse Input Support
--------------------

The Document content extraction script in PDF-Extract-Kit supports various input formats such as ``a single image/PDF``, ``a directory of image/PDF files``.


Output result
--------------------

The extracted results of PDF documents are stored in the outputs path in the form of json. The format of json is as follows:

.. code:: json

    [
        {
            "layout_dets": [
                {
                    "category_type": "text",
                    "poly": [
                        380.6792698635707,
                        159.85058512958923,
                        765.1419999999998,
                        159.85058512958923,
                        765.1419999999998,
                        192.51073013642917,
                        380.6792698635707,
                        192.51073013642917
                    ],
                    "text": "this is an example text",
                    "score": 0.97
                },
                ...
            ], 
            "page_info": {
                "page_no": 0,
                "height": 2339,
                "width": 1654,
            }
        },
        ...
    ]

- layout_dets: Single page of PDF or image content extraction results
- category_type: The attribution of a single piece of content, such as headings, images, inline formulas, and so on
- poly: The location coordinates of a single content block
- text: Text content of a single content block
- score: Confidence score
- page_info: Page information, including page number and page size
- page_no: Page number, counting from 0
- height: Page size: height
- width: Page size: width

If the ``merge2markdown`` parameter is True, an additional markdown file will be saved.