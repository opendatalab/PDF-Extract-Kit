==================================
Installation
==================================

In this section, we will demonstrate how to install PDF-Extract-Kit.

Best Practices
==============

We recommend users follow our best practices for installing PDF-Extract-Kit. It is recommended to use a Python 3.10 conda virtual environment for the installation.

**Step 1.** Create a Python 3.10 virtual environment using conda.

.. code-block:: console

    $ conda create -n pdf-extract-kit-1.0 python=3.10 -y
    $ conda activate pdf-extract-kit-1.0

**Step 2.** Install the dependencies for PDF-Extract-Kit.

.. code-block:: console

    $ # For GPU devices
    $ pip install -r requirements.txt
    $ # For CPU-only devices
    $ pip install -r requirements-cpu.txt

.. note::

    For the convenience of user environment configuration, requirements.txt only includes the environment needed for the current best models, which currently include:
   
    - Layout Detection: YOLO series (YOLOv10, DocLayout-YOLO)  
    - Formula Detection: YOLO series (YOLOv8)  
    - Formula Recognition: UniMERNet  
    - OCR: PaddleOCR  

    For other models, such as LayoutLMv3, additional environment setup is required. For details, see \ :ref:`Layout Detection Algorithms <algorithm_layout_detection>`.