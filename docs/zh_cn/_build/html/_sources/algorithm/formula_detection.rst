..  _algorithm_formula_detection:

====================
公式检测算法
====================

简介
====================

公式检测是针对给定的输入图像，检测出图像中所有包含公式的位置（包含行内公式和行间公式）

.. note::

   公式检测实际上属于布局检测子任务，但由于公式检查的复杂性，我们建议使用单独的公式检测模型解耦。
   这样通常使得数据标注更加方便，且公式检测效果也更好。

模型使用
====================

在配置好环境的情况下，直接执行 ``scripts/formula_detection.py`` 即可运行布局检测算法脚本。

.. code:: shell

   $ python scripts/formula_detection.py --config configs/formula_detection.yaml

模型配置
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

- inputs/outputs: 分别定义输入文件路径和可视化输出目录
- tasks: 定义任务类型，当前只包含一个公式检测任务
- model: 定义具体模型类型: 当前仅提供YOLO公式检测模型
- model_config: 定义模型配置
- img_size: 定义图像长边大小，短边会根据长边等比例缩放
- conf_thres: 定义置信度阈值，仅检测大于该阈值的目标
- iou_thres: 定义IoU阈值，去除重叠度大于该阈值的目标
- batch_size: 定义批量大小，推理时每次同时推理的图像数，一般情况下越大推理速度越快，显卡越好该数值可以设置的越大
- model_path: 模型权重路径
- visualize: 是否对模型结果进行可视化，可视化结果会保存在outputs目录下。

多样化输入支持
--------------------

PDF-Extract-Kit中的公式检测脚本支持 ``单个图像`` 、 ``只包含图像文件的目录`` 、 ``单个PDF文件`` 、 ``只包含PDF文件的目录`` 等输入形式。

.. note:: 

   根据自己实际数据形式，修改 ``configs/formula_detection.yaml`` 中 ``inputs`` 的路径即可
   - 单个图像: path/to/image  
   - 图像文件夹: path/to/images  
   - 单个PDF文件: path/to/pdf  
   - PDF文件夹: path/to/pdfs  

.. note::

   当使用PDF作为输入时，需要将 ``formula_detection.py`` 中的 ``predict_images`` 修改为 ``predict_pdfs`` 。


   .. code:: python

      # for image detection
      detection_results = model_formula_detection.predict_images(input_data, result_path)
   

   .. code:: python

      # for pdf detection
      detection_results = model_formula_detection.predict_pdfs(input_data, result_path)


可视化结果查看
--------------------

当config文件中 ``visualize`` 设置为 ``True`` 时，可视化结果会保存在 ``outputs/formula_detection`` 目录下。

.. note::

   可视化可以方便对模型结果进行分析，但当进行大批量任务时，建议关掉可视化(设置 ``visualize`` 为 ``False`` )，减少内存和磁盘占用。