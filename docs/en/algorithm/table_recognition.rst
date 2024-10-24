..  _algorithm_table_recognition:

========================
Table Recognition Algorithm
========================

Introduction
=================

Table recognition refers to the process of inputting a table image, identifying the table structure and content, and converting it into formats such as ``LaTeX`` or ``HTML``.

Model Usage
=================

With the environment properly configured, you can run the table recognition algorithm script by directly executing ``scripts/table_parsing.py``.

.. code:: shell

   $ python scripts/table_parsing.py --config configs/table_parsing.yaml

Model Configuration
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

- inputs/outputs: Define the input file path and table recognition result directory respectively
- tasks: Define the task type, currently only including one table recognition task
- model: Define the specific model type: currently using the `StructEqTable <https://github.com/UniModal4Reasoning/StructEqTable-Deploy>`_ table recognition model
- model_config: Define the model configuration
- model_path: Path to the model weights
- max_new_tokens: Maximum number of tokens to generate, default is 1024, maximum supported is 4096
- max_time: Maximum runtime for the model (in seconds)
- output_format: Output format, default is set to ``latex``, options include ``html`` and ``markdown``
- lmdeploy: Whether to use LMDeploy for deployment, currently set to False
- flash_attn: Whether to use flash attention, only available for Ampere GPUs

Diverse Input Support
-----------------

The table recognition script in PDF-Extract-Kit supports ``single table images`` and ``multiple table images`` as input.

.. note::

   The StructEqTable model only supports running on GPU devices

.. note::
    
    Adjust ``max_new_tokens`` and ``max_time`` according to the table content, defaults are 1024 and 30 respectively.

.. note::
    
    lmdeploy is an option for accelerated inference. If set to True, it will use LMDeploy for accelerated inference deployment.
    To use LMDeploy deployment, you need to install LMDeploy. For installation methods, refer to `LMDeploy <https://github.com/InternLM/lmdeploy>`_.