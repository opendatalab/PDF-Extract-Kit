# Using PDF-Extract-Kit on macOS

## Overview

The project was initially developed with a default environment of Linux servers, so running it directly on a macOS machine can be challenging.
After encountering some issues, we have compiled a list of problems that might arise on macOS and documented them in this guide. Not all solutions provided here may apply to your specific setup. If you have any questions, please raise them in an issue.

- [Intel CPU](#using-on-intel-cpu-machine) Click here for Intel CPU machines
- [M-series CPU](#using-on-m-series-chip-machine) Click here for M-series chip machines


## Preprocessing

To run the project smoothly on macOS, perform the following preparations:
- Install ImageMagick:
  - https://docs.wand-py.org/en/latest/guide/install.html#install-imagemagick-on-mac
- Modify configurations:
  - PDF-Extract-Kit/pdf_extract.py:148 
    ```python
    dataloader = DataLoader(dataset, batch_size=128, num_workers=0)
    ```
   
 
## Using on Intel CPU machine

### 1.Create a Virtual Environment

Use either venv or conda, with Python version recommended as 3.10.

### 2.Install Dependencies

```bash
pip install unimernet==0.1.0
pip install -r requirements-without-unimernet+cpu.txt

# For detectron2, compile it yourself as per https://github.com/facebookresearch/detectron2/issues/5114
# Or use our precompiled wheel
pip install https://github.com/opendatalab/PDF-Extract-Kit/raw/main/assets/whl/detectron2-0.6-cp310-cp310-macosx_10_9_universal2.whl
```

### 3.Modify config, use CPU for inference

PDF-Extract-Kit/configs/model_configs.yaml:2
```yaml
device: cpu
```
PDF-Extract-Kit/modules/layoutlmv3/layoutlmv3_base_inference.yaml:72
```yaml
DEVICE: cpu
```

### 4.Run the Application

```bash
python pdf_extract.py --pdf demo/demo1.pdf
```


## Using on M-series chip machine

### 1.Create a Virtual Environment

Use either venv or conda, with Python version recommended as 3.10.

### 2.Install Dependencies

```bash
pip install -r requirements+cpu.txt

# For detectron2, compile it yourself as per https://github.com/facebookresearch/detectron2/issues/5114
# Or use our precompiled wheel
pip install https://github.com/opendatalab/PDF-Extract-Kit/raw/main/assets/whl/detectron2-0.6-cp310-cp310-macosx_11_0_arm64.whl
```

### 3. Modify config, use MPS for accelerated inference

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

### 5.FAQ

- On some newer M chip devices, MPS acceleration fails to activate.
  - Uninstall torch and torchvision, then reinstall the nightly build versions of torch and torchvision.
  - ```bash
    pip uninstall torch torchvision 
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
    ```
  - Reference source: https://github.com/opendatalab/PDF-Extract-Kit/issues/23

