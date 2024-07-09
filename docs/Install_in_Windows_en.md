# Using PDF-Extract-Kit on Windows

## Overview

The project was initially developed with a default environment of Linux servers, so running it directly on a Windows machine can be challenging.
After encountering some issues, we have compiled a list of problems that might arise on Windows and documented them in this guide. Since the Windows environment is highly fragmented, not all solutions provided here may apply to your specific setup. If you have any questions, please raise them in an issue.
- [CPU Environment](#Using-in-CPU-Environment)  Click here for CPU usage
- [GPU Environment](#Using-in-GPU-Environment)  Click here if you need CUDA acceleration

## Preprocessing

To run the project smoothly on Windows, perform the following preparations:
- Install ImageMagick:
  - https://docs.wand-py.org/en/latest/guide/install.html#install-imagemagick-on-windows
- Modify configurations:
  - PDF-Extract-Kit/pdf_extract.py:148 
    ```python
    dataloader = DataLoader(dataset, batch_size=128, num_workers=0)
    ```
    
## Using in CPU Environment

### 1.Create a Virtual Environment

Use either venv or conda, with Python version recommended as 3.10.

### 2.Install Dependencies

```bash
pip install -r requirements+cpu.txt

# For detectron2, compile it yourself as per https://github.com/facebookresearch/detectron2/issues/5114
# Or use our precompiled wheel
pip install https://github.com/opendatalab/PDF-Extract-Kit/raw/main/assets/whl/detectron2-0.6-cp310-cp310-win_amd64.whl
```

### 3.Modify Configurations for CPU Inference

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

## Using in GPU Environment

### 1.Verify CUDA and GPU Memory

- Recommended: CUDA 11.8 and cuDNN 8.7.0 (test other versions if needed)
  - CUDA 11.8
  https://developer.nvidia.com/cuda-11-8-0-download-archive
  - cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x
  https://developer.nvidia.com/rdp/cudnn-archive
- Ensure your GPU has adequate memory, with a minimum of 6GB recommended; ideally, 16GB or more is preferred.
  - If the GPU memory is less than 16GB, adjust the `batch_size` in the [Preprocessing](#Preprocessing) section as needed, lowering it to "64" or "32" appropriately.



### 2.Create a Virtual Environment

Use either venv or conda, with Python version recommended as 3.10.

### 3.Install Dependencies

```bash
pip install -r requirements+cpu.txt

# For detectron2, compile it yourself as per https://github.com/facebookresearch/detectron2/issues/5114
# Or use our precompiled wheel
pip install https://github.com/opendatalab/PDF-Extract-Kit/blob/main/assets/whl/detectron2-0.6-cp310-cp310-win_amd64.whl

# For GPU usage, ensure PyTorch is installed with CUDA support.
pip install --force-reinstall torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
```

### 3.Modify Configurations for CUDA Inference

PDF-Extract-Kit/configs/model_configs.yaml:2
```yaml
device: cuda
```
PDF-Extract-Kit/modules/layoutlmv3/layoutlmv3_base_inference.yaml:72
```yaml
DEVICE: cuda
```

### 4.Run the Application

```bash
python pdf_extract.py --pdf demo/demo1.pdf
```
