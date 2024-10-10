==================================
Model Weights Download
==================================

Before using the PDF-Extract-Kit, we need to download the required model weights. You can download all models or specific model files (e.g., formula detection MFD) according to your needs.

[Recommended] Method 1: ``snapshot_download``
========================================

HuggingFace
------------

``huggingface_hub.snapshot_download`` supports downloading specific model weights from the HuggingFace Hub and allows multithreading. You can use the following code to download model weights in parallel:

.. code:: python

   from huggingface_hub import snapshot_download

   snapshot_download(repo_id='opendatalab/pdf-extract-kit-1.0', local_dir='./', max_workers=20)

If you want to download a single algorithm model (e.g., the YOLO model for the formula detection task), use the following code:

.. code:: python

   from huggingface_hub import snapshot_download

   snapshot_download(repo_id='opendatalab/pdf-extract-kit-1.0', local_dir='./', allow_patterns='models/MFD/YOLO/*') 

.. note::

   Here, ``repo_id`` represents the name of the model on HuggingFace Hub, ``local_dir`` indicates the desired local storage path, ``max_workers`` specifies the maximum number of parallel downloads, and ``allow_patterns`` specifies the files you want to download.

.. tip::

   If ``local_dir`` is not specified, it will be downloaded to the default cache path of HuggingFace (``~/.cache/huggingface/hub``). To change the default cache path, modify the relevant environment variables:

   .. code:: console

      $ # Default is `~/.cache/huggingface/`
      $ export HF_HOME=Comming soon!

.. tip::
   
   If the download speed is slow (e.g., unable to reach maximum bandwidth), try setting ``export HF_HUB_ENABLE_HF_TRANSFER=1`` for higher download speeds.

ModelScope
-----------

``modelscope.snapshot_download`` supports downloading specified model weights. You can use the following command to download the model:

.. code:: python

   from modelscope import snapshot_download

   snapshot_download(model_id='opendatalab/pdf-extract-kit-1.0', cache_dir='./')

If you want to download a single algorithm model (e.g., the YOLO model for the formula detection task), use the following code:

.. code:: python

   from modelscope import snapshot_download

   snapshot_download(repo_id='opendatalab/pdf-extract-kit-1.0', local_dir='./', allow_patterns='models/MFD/YOLO/*') 


.. note::
   Here, ``model_id`` represents the name of the model in the ModelScope library, ``cache_dir`` indicates the desired local storage path, and ``allow_patterns`` specifies the files you want to download.

.. note::
   ``modelscope.snapshot_download`` does not support multithreaded parallel downloads.

.. tip::

   If ``cache_dir`` is not specified, it will be downloaded to the default cache path of ModelScope (``~/.cache/huggingface/hub``).

   To change the default cache path, modify the relevant environment variables:

   .. code:: console

      $ # Default is ~/.cache/modelscope/hub/
      $ export MODELSCOPE_CACHE=XXXX



Method 2: Git LFS
===================

The remote model repositories of HuggingFace and ModelScope are Git repositories managed by Git LFS. Therefore, we can use ``git clone`` to download the weights:

.. code:: console

   $ git lfs install
   $ # From HuggingFace
   $ git lfs clone https://huggingface.co/opendatalab/pdf-extract-kit-1.0
   $ # From ModelScope
   $ git clone https://www.modelscope.cn/opendatalab/pdf-extract-kit-1.0.git