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

    $ pip install -r requirements.txt

.. note::

    如果你的设备不支持 GPU，请使用 ``requirements-cpu.txt`` 安装 CPU 版本的依赖。