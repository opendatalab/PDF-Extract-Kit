# Using PDF-Extract-Kit on macOS

## Overview

The project was initially developed with a default environment of Linux servers, so running it directly on a macOS machine can be challenging.
After encountering some issues, we have compiled a list of problems that might arise on macOS and documented them in this guide. Not all solutions provided here may apply to your specific setup. If you have any questions, please raise them in an issue.


## Preprocessing

To run the project smoothly on macOS, perform the following preparations:
- Install ImageMagick:
  - https://docs.wand-py.org/en/latest/guide/install.html#install-imagemagick-on-mac
- Modify configurations:
  - PDF-Extract-Kit/pdf_extract.py:148 
    ```python
    dataloader = DataLoader(dataset, batch_size=128, num_workers=0)
    ```
    
## Installation Process

### 1.Create a Virtual Environment

Use either venv or conda, with Python version recommended as 3.10.

### 2.Install Dependencies

```bash
pip install unimernet==0.1.0
pip install -r requirements-macos.txt

# For detectron2, compile it yourself as per https://github.com/facebookresearch/detectron2/issues/5114
# Or use our precompiled wheel
pip install https://github.com/opendatalab/PDF-Extract-Kit/raw/main/assets/whl/detectron2-0.6-cp310-cp310-macosx_10_9_universal2.whl
```

### 3.Modify Configuration to Adapt to Device Type

- #### For Intel CPU Machines, Use CPU for Inference

PDF-Extract-Kit/configs/model_configs.yaml:2
```yaml
device: cpu
```
PDF-Extract-Kit/modules/layoutlmv3/layoutlmv3_base_inference.yaml:72
```yaml
DEVICE: cpu
```

- #### Acceleration Using M Series Chips

PDF-Extract-Kit/configs/model_configs.yaml:2
```yaml
device: mps
```
PDF-Extract-Kit/modules/layoutlmv3/layoutlmv3_base_inference.yaml:72
```yaml
DEVICE: mps
```

### 4.Run the Application

```bash
python pdf_extract.py --pdf demo/demo1.pdf
```
