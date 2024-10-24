..  _algorithm_table_recognition:

============
表格识别算法
============

简介
=================

表格识别是指输入表格图像，识别表格结构和内容，并将其转换为 ``LaTeX`` 或 ``HTML`` 等格式。

模型使用
=================

在配置好环境的情况下，直接执行 ``scripts/table_parsing.py`` 即可运行表格识别算法脚本。

.. code:: shell

   $ python scripts/table_parsing.py --config configs/table_parsing.yaml

模型配置
-----------------

.. code:: yaml

    inputs: assets/demo/table_parsing
    outputs: outputs/table_parsing
    tasks:
      table_parsing:
        model: table_parsing_struct_eqtable
        model_config:
          model_path: models/TabRec/StructEqTable
          max_new_tokens: 1024
          max_time: 30
          output_format: latex
          lmdeploy: False
          flash_attn: True

- inputs/outputs: 分别定义输入文件路径和表格识别结果目录
- tasks: 定义任务类型，当前只包含一个表格识别任务
- model: 定义具体模型类型: 当前使用 `StructEqTable  <https://github.com/UniModal4Reasoning/StructEqTable-Deploy>`_ 表格识别模型
- model_config: 定义模型配置
- model_path: 模型权重路径
- max_new_tokens: 生成的最大token数量, 默认为1024, 最大支持4096
- max_time: 模型运行的最大时间（秒）
- output_format: 输出格式，默认设置为 ``latex``, 可选有 ``html`` 和 ``markdown``
- lmdeploy: 是否使用 LMDeploy 进行部署，当前设置为 False
- flash_attn: 是否使用flash attention，仅适用于Ampere GPU


多样化输入支持
-----------------

PDF-Extract-Kit中的表格识别脚本支持 ``单个表格图像`` 和 ``多个表格图像`` 作为输入。

.. note::

   StructEqTable表格模型仅支持GPU设备下运行

.. note::
    
    根据表格内容调整 ``max_new_tokens`` 和 ``max_time``, 默认分别为1024和30。

.. note::
    
    lmdeploy为加速推理的选项，如果设置为True，将使用LMDeploy进行加速推理部署。
    使用LMDeploy部署需要安装LMDeploy，安装方法参考 `LMDeploy <https://github.com/InternLM/lmdeploy>`_ 。

