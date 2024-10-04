==================================
快速开始
==================================

配置好PDF-Extract-Kit环境，并下载好模型后，我们可以开始使用PDF-Extract-Kit了。



布局检测示例
==============

.. code-block:: console

    $ python scripts/layout_detection.py --config configs/layout_detection.yaml

执行完之后，我们可以在`outpus/layout_detection`目录下查看检测结果。

.. note::   

    ``layout_detection.yaml``设置输入、输出及模型配置，布局检测更详细教程见\ :ref:`布局检测算法 <algorithm_layout_detection>` \ 。


公式检测示例
==============


.. code-block:: console

    $ python scripts/formula_detection.py --config configs/formula_detection.yaml

执行完之后，我们可以在`outpus/formula_detection`目录下查看检测结果。

.. note::   

    ``formula_detection.yaml``设置输入、输出及模型配置，公式检测更详细教程见 \ :ref:`公式检测算法 <algorithm_formula_detection>` \ 。
