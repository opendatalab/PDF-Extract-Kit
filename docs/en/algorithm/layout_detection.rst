.. _algorithm_layout_detection:

=================
布局检测算法
=================

简介
=================

布局检测是文档内容提取的基础任务，目标对页面中不同类型的区域进行定位：如图像、表格、文本、标题等，方便后续高质量内容提取。对于文本、标题等区域，可以基于OCR模型进行文字识别，对于表格区域可以基于表格识别模型进行转换。

模型使用
=================

在配置好环境的情况下，直接执行``scripts/layout_detection.py``即可运行布局检测算法脚本。

.. code:: shell

   $ python scripts/layout_detection.py --config configs/layout_detection.yaml

模型配置
-----------------

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
               batch_size: 1
               model_path: models/Layout/yolov8/yolov8_mixed_1600.pt
               visualize: True

- inputs/outputs: 分别定义输入文件路径和可视化输出目录
- tasks: 定义任务类型，当前只包含一个布局检测任务
- model: 定义具体模型类型: 如layout_detection_yolo 或者 layout_detection_layoutlmv3
- model_config: 定义模型配置
- img_size: 定义图像长边大小，短边会根据长边等比例缩放
- conf_thres: 定义置信度阈值，仅检测大于该阈值的目标
- iou_thres: 定义IoU阈值，去除重叠度大于该阈值的目标
- batch_size: 定义批量大小，推理时每次同时推理的图像数，一般情况下越大推理速度越快，显卡越好该数值可以设置的越大
- model_path: 模型权重路径
- visualize: 是否对模型结果进行可视化，可视化结果会保存在outputs目录下。

多样化输入支持
-----------------

PDF-Extract-Kit中的布局检测脚本支持 ``单个图像``、 ``只包含图像文件的目录``、 ``单个PDF文件``、 ``只包含PDF文件的目录``等输入形式。

.. note::

   根据自己实际数据形式，修改configs/layout_detection.yaml中inputs的路径即可
   - 单个图像: path/to/image  
   - 图像文件夹: path/to/images  
   - 单个PDF文件: path/to/pdf  
   - PDF文件夹: path/to/pdfs  

.. note::
   当使用PDF作为输入时，需要将 ``formula_detection.py``

   .. code:: python

      # for image detection
      detection_results = model_layout_detection.predict_images(input_data, result_path)

   中的 ``predict_images``修改为 ``predict_pdfs``。

   .. code:: python

      # for pdf detection
      detection_results = model_layout_detection.predict_pdfs(input_data, result_path)

可视化结果查看
-----------------

当config文件中 ``visualize`` 设置为 ``True`` 时，可视化结果会保存在 ``outputs`` 目录下。

.. note::

   可视化可以方便对模型结果进行分析，但当进行大批量任务时，建议关掉可视化(设置 ``visualize``为 ``False``)，减少内存和磁盘占用。