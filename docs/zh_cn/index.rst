.. xtuner documentation master file, created by
   sphinx-quickstart on Tue Jan  9 16:33:06 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎来到 PDF-Extract-Kit 的中文文档
==============================================

.. figure:: ./_static/image/logo.png
  :align: center
  :alt: pdf-extract-kit
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>高质量文档解析工具箱
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/opendatalab/PDF-Extract-Kit" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/opendatalab/PDF-Extract-Kit/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/opendatalab/PDF-Extract-Kit/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>


文档
-------------
.. toctree::
   :maxdepth: 2
   :caption: 快速上手

   get_started/installation.rst
   get_started/pretrained_model.rst
   get_started/quickstart.rst

.. toctree::
   :maxdepth: 2
   :caption: 基础算法模块

   algorithm/layout_detection.rst
   algorithm/formula_detection.rst
   algorithm/formula_recognition.rst
   algorithm/ocr.rst
   algorithm/table_recognition.rst
   algorithm/reading_order.rst

.. toctree::
   :maxdepth: 2
   :caption: 新任务拓展

   task_extend/code.rst
   task_extend/doc.rst
   task_extend/evaluation.rst

.. toctree::
   :maxdepth: 2
   :caption: 支持的模型列表

   models/supported.md


.. toctree::
   :maxdepth: 2
   :caption: 模型性能评测

   evaluation/layout_detection.rst
   evaluation/formula_detection.rst
   evaluation/formula_recognition.rst
   evaluation/ocr.rst
   evaluation/table_recognition.rst
   evaluation/reading_order.rst
   evaluation/pdf_extract.rst

.. toctree::
   :maxdepth: 2
   :caption: PDF项目

   project/pdf_extract.md
   project/doc_translate.md
   project/speed_up.md