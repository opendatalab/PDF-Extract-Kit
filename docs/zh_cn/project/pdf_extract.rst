=================
文档内容提取项目
=================

简介
====================

文档内容提取是利用布局检测，公式检测，公式识别，OCR等模型，提取文档中的信息，并转换为markdown文本。


项目使用
====================

在配置好环境的情况下，直接执行 ``project/pdf2markdown/scripts/run_project.py`` 即可运行文档内容提取项目。

.. code:: shell

   $ python project/pdf2markdown/scripts/run_project.py --config project/pdf2markdown/configs/pdf2markdown.yaml


项目配置
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

- inputs/outputs: 分别定义输入文件路径和输出路径
- visualize: 是否对模型结果进行可视化，可视化结果会保存在outputs目录下。
- merge2markdown: 是否将结果合并为markdown文档，这里只支持简单的单栏文本从上往下进行拼接，更复杂布局文档的markdown转换请参考 `MinerU <https://github.com/opendatalab/MinerU>`_
- tasks: 定义任务类型，PDF文档提取包含了布局检测、公式检测、公式识别、OCR等任务
- 具体每个任务和模型的参数含义请参考各任务的教程文档


多样化输入支持
--------------------

PDF文档内容提取支持 ``单个图像/PDF文件`` 、 ``包含图像/PDF文件的目录`` 等输入形式。


输出结果
--------------------

PDF文档提取的结果以json形式保存在 ``outputs`` 路径下，json的格式如下所示：

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

- layout_dets: 单页PDF或图片的内容提取结果
- category_type: 单个内容块的所属内别，比如标题、图片、行内公式等等
- poly: 单个内容块的位置坐标
- text: 该文本块的文本内容
- score: 检测的置信度
- page_info: 页面信息，包含页码和页面尺寸
- page_no: 页码，从0开始计数
- height: 页面尺寸: 高
- width: 页面尺寸: 宽

如果 ``merge2markdown`` 参数为True的话，则会额外保存一个markdown文件。