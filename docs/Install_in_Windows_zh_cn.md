# 在Windows环境下使用PDF-Extract-Kit

## 概述

项目开发之初默认使用环境是Linux服务器环境，因此在windows单机直接运行本项目存在一些困难，经过一段时间的踩坑后，我们总结了一些windows上可能遇到的问题，
并写下本文档。由于windows环境碎片化严重，本文档中的解决方案可能不适用于您，如有疑问，请在issue中向我们提问。

- [cpu环境](#在cpu环境使用)  在cpu环境使用点这里
- [gpu环境](#在gpu环境使用)  需要CUDA加速点这里

## 预处理

在windows正常运行本项目需要提前进行的处理
- 安装ImageMagick
  - https://docs.wand-py.org/en/latest/guide/install.html#install-imagemagick-on-windows
- 需要修改的配置
  - PDF-Extract-Kit/pdf_extract.py:148 
    ```python
    dataloader = DataLoader(dataset, batch_size=128, num_workers=0)
    ```
    
## 在cpu环境使用

### 1.创建一个虚拟环境

使用venv或conda均可, python版本建议3.10

### 2.安装依赖

```bash
pip install -r requirements+cpu.txt

# detectron2需要编译安装,自行编译安装可以参考https://github.com/facebookresearch/detectron2/issues/5114
# 或直接使用我们编译好的的whl包
pip install https://github.com/opendatalab/PDF-Extract-Kit/raw/main/assets/whl/detectron2-0.6-cp310-cp310-win_amd64.whl
```

### 3.修改config, 使用cpu推理

PDF-Extract-Kit/configs/model_configs.yaml:2
```yaml
device: cpu
```
PDF-Extract-Kit/modules/layoutlmv3/layoutlmv3_base_inference.yaml:72
```yaml
DEVICE: cpu
```

### 4.运行

```bash
python pdf_extract.py --pdf demo/demo1.pdf
```

## 在gpu环境使用

### 1.确认CUDA环境和显卡显存

- 推荐安装CUDA 11.8 + cuDNN 8.7.0(其他版本可以自行测试)
  - CUDA 11.8
  https://developer.nvidia.com/cuda-11-8-0-download-archive
  - cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x
  https://developer.nvidia.com/rdp/cudnn-archive
- 确认显卡显存是否够用，最低6GB，推荐16GB及以上 
  - 如果显存小于16GB，请将[预处理](#预处理)中需要修改的配置中batch_size酌情调低至"64"或"32"


### 2.创建一个虚拟环境

使用venv或conda均可, python版本建议3.10

### 3.安装依赖

```bash
pip install -r requirements+cpu.txt

# detectron2需要编译安装,自行编译安装可以参考https://github.com/facebookresearch/detectron2/issues/5114
# 或直接使用我们编译好的的whl包
pip install https://github.com/opendatalab/PDF-Extract-Kit/blob/main/assets/whl/detectron2-0.6-cp310-cp310-win_amd64.whl

# 使用gpu方案时，需要重新安装cuda版本的pytorch
pip install --force-reinstall torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
```

### 3.修改config, 使用cuda推理

PDF-Extract-Kit/configs/model_configs.yaml:2
```yaml
device: cuda
```
PDF-Extract-Kit/modules/layoutlmv3/layoutlmv3_base_inference.yaml:72
```yaml
DEVICE: cuda
```

### 4.运行

```bash
python pdf_extract.py --pdf demo/demo1.pdf
```
