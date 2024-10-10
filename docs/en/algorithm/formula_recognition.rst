..  _algorithm_formula_recognition:

============
Formula Recognition Algorithm
============

Introduction
=================

Formula detection involves recognizing the content of a given input formula image and converting it to ``LaTeX`` format.

Model Usage
=================

With the environment properly configured, you can run the layout detection algorithm script by executing ``scripts/formula_recognition.py``.

.. code:: shell

   $ python scripts/formula_recognition.py --config configs/formula_recognition.yaml

Model Configuration
-----------------

.. code:: yaml

   inputs: assets/demo/formula_recognition
   outputs: outputs/formula_recognition
   tasks:
      formula_recognition:
         model: formula_recognition_unimernet
         model_config:
            cfg_path: pdf_extract_kit/configs/unimernet.yaml
            model_path: models/MFR/unimernet_tiny
            visualize: False

- inputs/outputs: Define the input file path and the directory for LaTeX prediction results, respectively.
- tasks: Define the task type, currently only containing a formula recognition task.
- model: Define the specific model type: Currently, only the `UniMERNet <https://github.com/opendatalab/UniMERNet>`_ formula recognition model is provided.
- model_config: Define the model configuration.
- cfg_path: Path to the UniMERNet configuration file.
- model_path: Path to the model weights.
- visualize: Whether to visualize the model results. Visualized results will be saved in the outputs directory.

Support for Diverse Inputs
-----------------

The formula detection script in PDF-Extract-Kit supports ``single formula images`` and ``document images with corresponding formula regions``.

Viewing Visualization Results
-----------------

When the visualize setting in the config file is set to True, ``LaTeX`` prediction results will be saved in the outputs directory.