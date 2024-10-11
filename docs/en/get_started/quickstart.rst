==================================
Quick Start
==================================

Once the PDF-Extract-Kit environment is set up and the models are downloaded, we can start using PDF-Extract-Kit.

Layout Detection Example
==============

Layout detection offers several models: ``LayoutLMv3``, ``YOLOv10``, and ``DocLayout-YOLO``. Compared to ``LayoutLMv3``, ``YOLOv10`` is faster. ``DocLayout-YOLO`` is based on YOLOv10 and includes diverse document pre-training and model optimization, offering both speed and high accuracy.

**1. Using Layout Detection Models**

.. code-block:: console

    $ python scripts/layout_detection.py --config configs/layout_detection.yaml

After execution, we can view the detection results in the `outputs/layout_detection` directory.

.. note::   

    The ``layout_detection.yaml`` file sets the input, output, and model configuration. For a more detailed tutorial on layout detection, see :ref:`Layout Detection Algorithm <algorithm_layout_detection>`.

Formula Detection Example
==============

.. code-block:: console

    $ python scripts/formula_detection.py --config configs/formula_detection.yaml

After execution, we can view the detection results in the `outputs/formula_detection` directory.

.. note::   

    The ``formula_detection.yaml`` file sets the input, output, and model configuration. For a more detailed tutorial on formula detection, see :ref:`Formula Detection Algorithm <algorithm_formula_detection>`.