==================================
Installation
==================================

In this section, we will demonstrate how to install PDF-Extract-Kit.

Best Practices
==============

We recommend users follow our best practices to install PDF-Extract-Kit.
It is recommended to use a Python 3.10 conda virtual environment to install PDF-Extract-Kit.

**Step 1.** Use conda to create a Python 3.10 virtual environment

.. code-block:: console

    $ conda create -n pdf-extract-kit-1.0 python=3.10 -y
    $ conda activate pdf-extract-kit-1.0

**Step 2.** Install the dependencies for PDF-Extract-Kit

.. code-block:: console

    $ pip install -r requirements.txt

.. note::

    If your device does not support GPU, please install the CPU version dependencies using ``requirements-cpu.txt`` instead of requirements.txt