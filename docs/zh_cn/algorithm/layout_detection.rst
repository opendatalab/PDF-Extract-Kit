.. _algorithm_layout_detection:

=================
布局检测算法
=================

简介
=================

``布局检测`` 是文档内容提取的基础任务，目标对页面中不同类型的区域进行定位：如 ``图像`` 、 ``表格`` 、 ``文本`` 、 ``标题`` 等，方便后续高质量内容提取。对于 ``文本`` 、 ``标题`` 等区域，可以基于 ``OCR模型`` 进行文字识别，对于表格区域可以基于表格识别模型进行转换。

模型使用
=================

布局检测模型支持以下模型：

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
        <th class="tg-0lax">模型</th>
        <th class="tg-f8tz">简述</th>
        <th class="tg-f8tz">特点</th>
        <th class="tg-f8tz">模型权重</th>
        <th class="tg-f8tz">配置文件</th>
      </tr></thead>
    <tbody>
      <tr>
        <td class="tg-0lax">DocLayout-YOLO</td>
        <td class="tg-0pky">基于YOLO-v10模型改进：<br>1. 生成多样性预训练数据，提升对多种类型文档泛化性<br>2. 模型结构改进，提升对多尺度目标感知能力<br>详见<a href="https://github.com/opendatalab/DocLayout-YOLO" target="_blank" rel="noopener noreferrer">DocLayout-YOLO</a></td>
        <td class="tg-0pky">速度快、精度高</td>
        <td class="tg-0pky"><a href="https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/blob/main/models/Layout/YOLO/doclayout_yolo_ft.pt" target="_blank" rel="noopener noreferrer">doclayout_yolo_ft.pt</a></td>
        <td class="tg-0pky">layout_detection.yaml</td>
      </tr>
      <tr>
        <td class="tg-0lax">YOLO-v10</td>
        <td class="tg-0pky">基础YOLO-v10模型</td>
        <td class="tg-0pky">速度快，精度一般</td>
        <td class="tg-0pky"><a href="https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/blob/main/models/Layout/YOLO/yolov10l_ft.pt" target="_blank" rel="noopener noreferrer">yolov10l_ft.pt</a></td>
        <td class="tg-0pky">layout_detection_yolo.yaml</td>
      </tr>
      <tr>
        <td class="tg-0lax">LayoutLMv3</td>
        <td class="tg-0pky">基础LayoutLMv3模型</td>
        <td class="tg-0pky">速度慢，精度较好</td>
        <td class="tg-0pky"><a href="https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/tree/main/models/Layout/LayoutLMv3" target="_blank" rel="noopener noreferrer">layoutlmv3_ft</a></td>
        <td class="tg-0pky">layout_detection_layoutlmv3.yaml</td>
      </tr>
    </tbody></table>


在配置好环境的情况下，直接执行 ``scripts/layout_detection.py`` 即可运行布局检测算法脚本。


**执行布局检测程序**

.. code:: shell

   $ python scripts/layout_detection.py --config configs/layout_detection.yaml

模型配置
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

- inputs/outputs: 分别定义输入文件路径和可视化输出目录
- tasks: 定义任务类型，当前只包含一个布局检测任务
- model: 定义具体模型类型，例如 ``layout_detection_yolo``
- model_config: 定义模型配置
- img_size: 定义图像长边大小，短边会根据长边等比例缩放，默认长边保持1024
- conf_thres: 定义置信度阈值，仅检测大于该阈值的目标
- iou_thres: 定义IoU阈值，去除重叠度大于该阈值的目标
- model_path: 模型权重路径
- visualize: 是否对模型结果进行可视化，可视化结果会保存在outputs目录下


**2. LayoutLMv3**

.. note::

   LayoutLMv3 默认情况下不能直接运行。运行时请将配置文件修改为configs/layout_detection_layoutlmv3.yaml，并且请按照以下步骤进行配置修改：

   1. **Detectron2 环境配置**

   .. code-block:: bash

      # 对于 Linux
      pip install https://wheels-1251341229.cos.ap-shanghai.myqcloud.com/assets/whl/detectron2/detectron2-0.6-cp310-cp310-linux_x86_64.whl

      # 对于 macOS
      pip install https://wheels-1251341229.cos.ap-shanghai.myqcloud.com/assets/whl/detectron2/detectron2-0.6-cp310-cp310-macosx_10_9_universal2.whl

      # 对于 Windows
      pip install https://wheels-1251341229.cos.ap-shanghai.myqcloud.com/assets/whl/detectron2/detectron2-0.6-cp310-cp310-win_amd64.whl

   2. **启用 LayoutLMv3 注册代码**

   请取消注释以下链接中的代码行：
   
   - `第2行 <https://github.com/opendatalab/PDF-Extract-Kit/blob/main/pdf_extract_kit/tasks/layout_detection/__init__.py#L2>`_
   - `第8行 <https://github.com/opendatalab/PDF-Extract-Kit/blob/main/pdf_extract_kit/tasks/layout_detection/__init__.py#L8>`_

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

- inputs/outputs: 分别定义输入文件路径和可视化输出目录
- tasks: 定义任务类型，当前只包含一个布局检测任务
- model: 定义具体模型类型，例如layout_detection_layoutlmv3
- model_config: 定义模型配置
- model_path: 模型权重路径


多样化输入支持
-----------------

PDF-Extract-Kit中的布局检测脚本支持 ``单个图像`` 、 ``只包含图像文件的目录`` 、 ``单个PDF文件`` 、 ``只包含PDF文件的目录`` 等输入形式。

.. note::

   根据自己实际数据形式，修改configs/layout_detection.yaml中inputs的路径即可
   - 单个图像: path/to/image  
   - 图像文件夹: path/to/images  
   - 单个PDF文件: path/to/pdf  
   - PDF文件夹: path/to/pdfs  

.. note::
   当使用PDF作为输入时，需要将 ``layout_detection.py``

   .. code:: python

      # for image detection
      detection_results = model_layout_detection.predict_images(input_data, result_path)

   中的 ``predict_images`` 修改为 ``predict_pdfs`` 。

   .. code:: python

      # for pdf detection
      detection_results = model_layout_detection.predict_pdfs(input_data, result_path)

可视化结果查看
-----------------

当config文件中 ``visualize`` 设置为 ``True`` 时，可视化结果会保存在 ``outputs`` 目录下。

.. note::

   可视化可以方便对模型结果进行分析，但当进行大批量任务时，建议关掉可视化(设置 ``visualize`` 为 ``False`` )，减少内存和磁盘占用。