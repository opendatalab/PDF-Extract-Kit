..  _algorithm_formula_recognition:

============
公式识别算法
============

简介
=================

公式检测是指给定输入公式图像，识别公式图像内容并转为 ``LaTeX`` 格式。

模型使用
=================

在配置好环境的情况下，直接执行 ``scripts/formula_recognition.py`` 即可运行布局检测算法脚本。

.. code:: shell

   $ python scripts/formula_recognition.py --config configs/formula_recognition.yaml

模型配置
-----------------

.. code:: yaml

   inputs: assets/demo/formula_recognition
   outputs: outputs/formula_recognition
   tasks:
      formula_recognition:
         model: formula_recognition_unimernet
         model_config:
            cfg_path: pdf_extract_kit/configs/unimernet.yaml
            model_path: models/MFR/unimernet_tiny
            visualize: False

- inputs/outputs: 分别定义输入文件路径和LaTeX预测结果目录
- tasks: 定义任务类型，当前只包含一个公式识别任务
- model: 定义具体模型类型: 当前仅提供 `UniMERNet <https://github.com/opendatalab/UniMERNet>`_ 公式识别模型
- model_config: 定义模型配置
- cfg_path: UniMERNet配置文件路径
- model_path: 模型权重路径
- visualize: 是否对模型结果进行可视化，可视化结果会保存在outputs目录下。

多样化输入支持
-----------------

PDF-Extract-Kit中的公式检测脚本支持 ``单个公式图像`` 、 ``文档图像及对应公式区域``

可视化结果查看
-----------------

当config文件中visualize设置为True时， ``LaTeX`` 预测结果会保存在outputs目录下。