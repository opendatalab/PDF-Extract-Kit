==================================
安装
==================================

本节中，我们将演示如何安装 PDF-Extract-Kit。

最佳实践
========

我们推荐用户参照我们的最佳实践安装 PDF-Extract-Kit。
推荐使用 Python-3.10 的 conda 虚拟环境安装 PDF-Extract-Kit。

**步骤 1.** 使用 conda 先构建一个 Python-3.10 的虚拟环境

.. code-block:: console

    $ conda create -n pdf-extract-kit-1.0 python=3.10 -y
    $ conda activate pdf-extract-kit-1.0

**步骤 2.** 安装 PDF-Extract-Kit 的依赖项

.. code-block:: console

    $ # 对于GPU设备
    $ pip install -r requirements.txt
    $ # 对于CPU设备
    $ pip install -r requirements-cpu.txt

.. note::

    考虑到用户环境配置的便捷性，我们在requirements.txt只包含当前最好模型需要的环境，目前包含  

    - 布局检测：YOLO系列（YOLOv10, DocLayout-YOLO）  
    - 公式检测：YOLO系列 (YOLOv8)  
    - 公式识别：UniMERNet  
    - OCR： PaddleOCR  

    对于其他模型请，如LayoutLMv3需要单独安装环境，具体见\ :ref:`布局检测算法 <algorithm_layout_detection>`