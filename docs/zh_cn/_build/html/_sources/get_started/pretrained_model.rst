==================================
模型权重下载
==================================

在使用PDF-Extract-Kit前，我们需要下载所需要的模型权重。可以根据自己需求下载全部模型或者特定的模型文件（如公式检测MFD）

[推荐] 方法 1：``snapshot_download``
========================================

HuggingFace
------------

``huggingface_hub.snapshot_download`` 支持下载特定的 HuggingFace Hub
模型权重，并且允许多线程。您可以利用下列代码并行下载模型权重：

.. code:: python

   from huggingface_hub import snapshot_download

   snapshot_download(repo_id='opendatalab/pdf-extract-kit-1.0', local_dir='./', max_workers=20)

如果想仅下载单个算法模型（如公式检测任务的YOLO模型），可以使用如下代码：

.. code:: python

   from huggingface_hub import snapshot_download

   snapshot_download(repo_id='opendatalab/pdf-extract-kit-1.0', local_dir='./', allow_patterns='models/MFD/YOLO/*') 

.. note::

   其中，\ ``repo_id`` 表示模型在 HuggingFace Hub 的名字、\ ``local_dir`` 表示期望存储到的本地路径、\ ``max_workers`` 表示下载的最大并行数，\ ``allow_patterns`` 表示想要现在的文件。

.. tip::

   如果未指定 ``local_dir``\ ，则将下载至 HuggingFace 的默认 cache 路径中（\ ``~/.cache/huggingface/hub``\ ）。若要修改默认 cache 路径，需要修改相关环境变量：

   .. code:: console

      $ # 默认为 ~/.cache/huggingface/
      $ export HF_HOME=Comming soon!

.. tip::
   
   如果觉得下载较慢（例如无法达到最大带宽等情况），可以尝试设置\ ``export HF_HUB_ENABLE_HF_TRANSFER=1`` 以获得更高的下载速度。

ModelScope
-----------

``modelscope.snapshot_download``
支持下载指定的模型权重，您可以利用下列命令下载模型：

.. code:: python

   from modelscope import snapshot_download

   snapshot_download(model_id='opendatalab/pdf-extract-kit-1.0', cache_dir='./')

如果想仅下载单个算法模型（如公式检测任务的YOLO模型），可以使用如下代码：

.. code:: python

   from modelscope import snapshot_download

   snapshot_download(repo_id='opendatalab/pdf-extract-kit-1.0', local_dir='./', allow_patterns='models/MFD/YOLO/*') 


.. note::
   其中，\ ``model_id`` 表示模型在 ModelScope 模型库的名字，\ ``cache_dir`` 表示期望存储到的本地路径， \ ``allow_patterns`` 表示想要现在的文件。


.. note::
   ``modelscope.snapshot_download`` 不支持多线程并行下载。

.. tip::

   如果未指定 ``cache_dir``\ ，则将下载至 ModelScope 的默认 cache 路径中（\ ``~/.cache/huggingface/hub``\ ）。

   若要修改默认 cache 路径，需要修改相关环境变量：

   .. code:: console

      $ # 默认为 ~/.cache/modelscope/hub/
      $ export MODELSCOPE_CACHE=XXXX



方法 2： Git LFS
===================

HuggingFace 和 ModelScope 的远程模型仓库就是一个由 Git LFS 管理的 Git
仓库。因此，我们可以利用 ``git clone`` 完成权重的下载：

.. code:: console

   $ git lfs install
   $ # From HuggingFace
   $ git lfs clone https://huggingface.co/opendatalab/pdf-extract-kit-1.0
   $ # From ModelScope
   $ git clone https://www.modelscope.cn/opendatalab/pdf-extract-kit-1.0.git
