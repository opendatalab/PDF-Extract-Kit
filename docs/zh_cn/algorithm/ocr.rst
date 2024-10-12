..  _algorithm_ocr:
==========================
光学字符识别(OCR)算法
==========================

简介
====================

光学字符识别(OCR)是指对图片中的文字块进行检测和识别。


模型使用
====================

在配置好环境的情况下，直接执行 ``scripts/ocr.py`` 即可运行OCR算法脚本。

.. code:: shell

   $ python scripts/ocr.py --config configs/ocr.yaml


模型配置
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

- inputs/outputs: 分别定义输入文件路径和输出路径
- visualize: 是否对模型结果进行可视化，可视化结果会保存在outputs目录下。
- tasks: 定义任务类型，当前只包含一个OCR任务
- model: 定义具体模型类型, 当前仅提供PaddleOCR模型
- model_config: 定义模型配置
- lang: 定义语种，默认语种ch支持中英文文字的检测和识别
- show_log: 是否打印检测识别过程的日志
- det_model_dir: 定义PaddleOCR检测模型的路径，指定路径不存在时，会自动下载模型权重到该路径
- rec_model_dir: 定义PaddleOCR识别模型的路径，指定路径不存在时，会自动下载模型权重到该路径
- det_db_box_thresh: 检测框筛选阈值，置信度低于该阈值的框会被舍弃


多样化输入支持
--------------------

PDF-Extract-Kit中的OCR脚本支持 ``单个图像/PDF文件`` 、 ``包含图像/PDF文件的目录`` 等输入形式。


可视化结果查看
--------------------

当config文件中 ``visualize`` 设置为 ``True`` 时，可视化结果会保存在 ``outputs`` 参数指定的目录下。

.. note::

   可视化可以方便对模型结果进行分析，但当进行大批量任务时，建议关掉可视化(设置 ``visualize`` 为 ``False`` )，减少内存和磁盘占用。