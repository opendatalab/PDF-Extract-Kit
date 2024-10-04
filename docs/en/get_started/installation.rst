==================================
安装
==================================

本节中，我们将演示如何安装 PDF-Extract-Kit。

最佳实践
========

我们推荐用户参照我们的最佳实践安装 PDF-Extract-Kit。
推荐使用 Python-3.10 的 conda 虚拟环境安装 PDF-Extract-Kit。

**步骤 0.** 使用 conda 先构建一个 Python-3.10 的虚拟环境

.. code-block:: console

    $ conda create --name pdf-extract python=3.10 -y
    $ conda activate pdf-extract

**步骤 1.** 安装 PDF-Extract-Kit

方案a: 通过 pip 直接安装

.. code-block:: console

    $ pip install -U 'pdf-extract-kit'

方案b: 从源码安装

.. code-block:: console

   $ git clone https://github.com/opendatalab/PDF-Extract-Kit.git
   $ cd PDF-Extract-Kit
   $ pip install -e '.'

.. note::

   "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，仅推荐有源码修改需求的用户使用。

验证
========

为了验证 PDF-Extract-Kit 是否安装正确，我们将使用命令打印配置文件。

**打印支持任务类型及模型信息：** 在命令行中使用 ``pdf-extract-kit list`` 验证是否能打印支持模型列表。

.. code-block:: console

   $ pdf-extract-kit list