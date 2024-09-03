# 在macOS系统使用PDF-Extract-Kit

## 概述

项目开发之初默认使用环境是Linux服务器环境，因此在macOS单机直接运行本项目存在一些困难，经过一段时间的踩坑后，我们总结了一些macOS上可能遇到的问题，
并写下本文档。本文档中的解决方案可能不适用于您，如有疑问，请在issue中向我们提问。

- [Intel cpu](#在Intel-cpu机器上使用)  使用Intel cpu机器点这里
- [M系列cpu](#在M系列芯片机器上使用)  使用M系列芯片机器点这里

## 预处理

在macOS正常运行本项目需要提前进行的处理
- 安装ImageMagick
  - https://docs.wand-py.org/en/latest/guide/install.html#install-imagemagick-on-mac
- 需要修改的配置
  - PDF-Extract-Kit/pdf_extract.py:148 
    ```python
    dataloader = DataLoader(dataset, batch_size=128, num_workers=0)
    ```


## 在Intel cpu机器上使用

### 1.创建一个虚拟环境

使用venv或conda均可, python版本建议3.10

### 2.安装依赖

```bash
pip install unimernet==0.1.0
pip install -r requirements-without-unimernet+cpu.txt

# detectron2需要编译安装,自行编译安装可以参考https://github.com/facebookresearch/detectron2/issues/5114
# 或直接使用我们编译好的的whl包
pip install https://github.com/opendatalab/PDF-Extract-Kit/raw/main/assets/whl/detectron2-0.6-cp310-cp310-macosx_10_9_universal2.whl
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


## 在M系列芯片机器上使用

### 1.创建一个虚拟环境

使用venv或conda均可, python版本建议3.10

### 2.安装依赖

```bash
pip install -r requirements+cpu.txt

# detectron2需要编译安装,自行编译安装可以参考https://github.com/facebookresearch/detectron2/issues/5114
# 或直接使用我们编译好的的whl包
pip install https://github.com/opendatalab/PDF-Extract-Kit/raw/main/assets/whl/detectron2-0.6-cp310-cp310-macosx_11_0_arm64.whl
```

### 3.修改config，使用mps加速推理

PDF-Extract-Kit/configs/model_configs.yaml:2
```yaml
device: mps
```
PDF-Extract-Kit/modules/layoutlmv3/layoutlmv3_base_inference.yaml:72
```yaml
DEVICE: mps
```

### 4.运行

```bash
python pdf_extract.py --pdf demo/demo1.pdf
```

### 5.FAQ
- 在部分较新的M芯片设备上，MPS加速开启失败
  - 卸载torch和torchvision，重新安装nightly构建版torch和torchvision
  - ```bash
    pip uninstall torch torchvision
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
    ```
  - 参考来源 https://github.com/opendatalab/PDF-Extract-Kit/issues/23