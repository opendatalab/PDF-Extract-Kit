
<p align="center">
  <img src="assets/readme/pdf-extract-kit_logo.png" width="220px" style="vertical-align:middle;">
</p>

<div align="center">

[English](./README.md) | 简体中文

[PDF-Extract-Kit-1.0中文教程](https://pdf-extract-kit.readthedocs.io/zh-cn/latest/get_started/pretrained_model.html)

[[Models (🤗Hugging Face)]](https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0) | [[Models(<img src="./assets/readme/modelscope_logo.png" width="20px">ModelScope)]](https://www.modelscope.cn/models/OpenDataLab/PDF-Extract-Kit-1.0) 
 
🔥🔥🔥 [MinerU：基于PDF-Extract-Kit的高效文档内容提取工具](https://github.com/opendatalab/MinerU)
</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/JYsXDXXN" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=MinerU" target="_blank">WeChat</a>
</p>


## 整体介绍

`PDF-Extract-Kit` 是一款功能强大的开源工具箱，旨在从复杂多样的 PDF 文档中高效提取高质量内容。以下是其主要功能和优势：

- **集成文档解析主流模型**：汇聚布局检测、公式检测、公式识别、OCR等文档解析核心任务的众多SOTA模型；
- **多样性文档下高质量解析结果**：结合多样性文档标注数据在进行模型微调，在复杂多样的文档下提供高质量解析结果；
- **模块化设计**：模块化设计使用户可以通过修改配置文件及少量代码即可自由组合构建各种应用，让应用构建像搭积木一样简便；  
- **全面评测基准**：提供多样性全面的PDF评测基准，用户可根据评测结果选择最适合自己的模型。  

**立即体验 PDF-Extract-Kit，解锁 PDF 文档的无限潜力！** 

> **注意：** PDF-Extract-Kit 专注于高质量文档处理，适合作为模型工具箱使用。
> 如果你想提取高质量文档内容(PDF转Markdown)，请直接使用[MinerU](https://github.com/opendatalab/MinerU)，MinerU结合PDF-Extract-Kit的高质量预测结果，进行了专门的工程优化，使得PDF文档内容提取更加便捷高效；  
> 如果你是一位开发者，希望搭建更多有意思的应用（如文档翻译，文档问答，文档助手等），基于PDF-Extract-Kit自行进行DIY将会十分便捷。特别地，我们会在`PDF-Extract-Kit/project`下面不定期更新一些有趣的应用，敬请期待！  

**我们欢迎社区研究员和工程师贡献优秀模型和创新应用，通过提交 PR 成为 PDF-Extract-Kit 的贡献者。**


## 模型概览

| **任务类型** | **任务描述**                                                                    | **模型**                     |
|--------------|---------------------------------------------------------------------------------|------------------------------|
| **布局检测** | 定位文档中不同元素位置：包含图像、表格、文本、标题、公式等 | `DocLayout-YOLO_ft`, `YOLO-v10_ft`, `LayoutLMv3_ft` |
| **公式检测** | 定位文档中公式位置：包含行内公式和行间公式                                      | `YOLOv8_ft`                       |
| **公式识别** | 识别公式图像为latex源码                                                         | `UniMERNet`                  |
|    **OCR**   | 提取图像中的文本内容（包括定位和识别）                                          | `PaddleOCR`                  |
| **表格识别** | 识别表格图像为对应源码（Latex/HTML/Markdown）                                   | `PaddleOCR+TableMaster`,`StructEqTable`  |
| **阅读顺序** | 将离散的文本段落进行排序拼接                                                    |  Coming Soon !                            |



## 新闻和更新
- `2024.10.22` 🎉🎉🎉 支持LaTex和HTML等多种输出格式的表格模型[StructTable-InternVL2-1B](https://huggingface.co/U4R/StructTable-InternVL2-1B)正式接入`PDF-Extract-Kit 1.0`，请参考[表格识别算法文档](https://pdf-extract-kit.readthedocs.io/zh-cn/latest/algorithm/table_recognition.html)进行使用！
- `2024.10.17` 🎉🎉🎉 检测结果更准确，速度更快的布局检测模型`DocLayout-YOLO`正式接入`PDF-Extract-Kit 1.0`，请参考[布局检测算法文档](https://pdf-extract-kit.readthedocs.io/zh-cn/latest/algorithm/layout_detection.html)进行使用！
- `2024.10.10` 🎉🎉🎉 基于模块化重构的`PDF-Extract-Kit 1.0`正式版本正式发布，模型使用更加便捷灵活！老版本请切换至[release/0.1.1](https://github.com/opendatalab/PDF-Extract-Kit/tree/release/0.1.1)分支进行使用。
- `2024.08.01` 🎉🎉🎉 新增了[StructEqTable](demo/TabRec/StructEqTable/README_TABLE.md)表格识别模块用于表格内容提取，欢迎使用！
- `2024.07.01` 🎉🎉🎉 我们发布了`PDF-Extract-Kit`，一个用于高质量PDF内容提取的综合工具包，包括`布局检测`、`公式检测`、`公式识别`和`OCR`。



## 效果展示

当前的一些开源SOTA模型多基于学术数据集进行训练评测，仅能在单一的文档类型上获取高质量结果。为了使得模型能够在多样性文档上也能获得稳定鲁棒的高质量结果，我们构建多样性的微调数据集，并在一些SOTA模型上微调已得到可实用解析模型。下边是一些模型的可视化结果。

### 布局检测

结合多样性PDF文档标注，我们训练了鲁棒的`布局检测`模型。在论文、教材、研报、财报等多样性的PDF文档上，我们微调后的模型都能得到准确的提取结果，对于扫描模糊、水印等情况也有较高鲁棒性。下面可视化示例是经过微调后的LayoutLMv3模型的推理结果。

![](assets/readme/layout_example.png)


### 公式检测

同样的，我们收集了包含公式的中英文文档进行标注，基于先进的公式检测模型进行微调，下面可视化结果是微调后的YOLO公式检测模型的推理结果：

![](assets/readme/mfd_example.png)


### 公式识别

[UniMERNet](https://github.com/opendatalab/UniMERNet)是针对真实场景下多样性公式识别的算法，通过构建大规模训练数据及精心设计的结果，使得其可以对复杂长公式、手写公式、含噪声的截图公式均有不错的识别效果。

### 表格识别

[StructEqTable](https://github.com/UniModal4Reasoning/StructEqTable-Deploy)是一个高效表格内容提取工具，能够将表格图像转换为LaTeX/HTML/Markdown格式，最新版本使用InternVL2-1B基础模型，提高了中文识别准确度并增加了多格式输出能力。

#### 更多模型的可视化结果及推理结果可以参考[PDF-Extract-Kit教程文档](xxx)


## 评测指标

Coming Soon! 

## 使用教程

### 环境安装

```bash
conda create -n pdf-extract-kit-1.0 python=3.10
conda activate pdf-extract-kit-1.0
pip install -r requirements.txt
```
> **注意：** 如果你的设备不支持 GPU，请使用 `requirements-cpu.txt` 安装 CPU 版本的依赖。

> **注意：** 目前doclayout-yolo仅支持从pypi源安装，如果出现doclayout-yolo无法安装，请通过 `pip3 install doclayout-yolo==0.0.2b1 --extra-index-url=https://pypi.org/simple` 安装。

### 模型下载

参考[模型权重下载教程](https://pdf-extract-kit.readthedocs.io/zh-cn/latest/get_started/pretrained_model.html)下载所需模型权重。注：可以选择全部下载，也可以选择部分下载，具体操作参考教程。


### Demo运行

#### 布局检测模型

```bash 
python scripts/layout_detection.py --config=configs/layout_detection.yaml
```
布局检测模型支持**DocLayout-YOLO**（默认模型），YOLO-v10，以及LayoutLMv3。对于YOLO-v10和LayoutLMv3的布局检测，请参考[Layout Detection Algorithm](https://pdf-extract-kit.readthedocs.io/zh-cn/latest/algorithm/layout_detection.html)。你可以在 `outputs/layout_detection` 文件夹下查看布局检测结果。

#### 公式检测模型

```bash 
python scripts/formula_detection.py --config=configs/formula_detection.yaml
```
你可以在 `outputs/formula_detection` 文件夹下查看公式检测结果。


#### 文本识别（OCR）模型

```bash 
python scripts/ocr.py --config=configs/ocr.yaml
```
你可以在 `outputs/ocr` 文件夹下查看OCR结果。


#### 公式识别模型

```bash 
python scripts/formula_recognition.py --config=configs/formula_recognition.yaml
```
你可以在 `outputs/formula_recognition` 文件夹下查看公式识别结果。


#### 表格识别模型

```bash 
python scripts/table_parsing.py --config configs/table_parsing.yaml
```
你可以在 `outputs/table_parsing` 文件夹下查看表格内容识别结果。


> **注意：** 更多模型使用细节请查看[PDF-Extract-Kit-1.0 中文教程](https://pdf-extract-kit.readthedocs.io/zh-cn/latest/get_started/pretrained_model.html).

> 本项目专注使用模型对`多样性`文档进行`高质量`内容提取，不涉及提取后内容拼接成新文档，如PDF转Markdown。如果有此类需求，请参考我们另一个Github项目: [MinerU](https://github.com/opendatalab/MinerU)


## 待办事项

- [x] **表格解析**：开发能够将表格图像转换成对应的LaTeX/Markdown格式源码的功能。  
- [ ] **化学方程式检测**：实现对化学方程式的自动检测。  
- [ ] **化学方程式/图解识别**：开发识别并解析化学方程式的模型。  
- [ ] **阅读顺序排序模型**：构建模型以确定文档中文本的正确阅读顺序。  

**PDF-Extract-Kit** 旨在提供高质量PDF文件的提取能力。我们鼓励社区提出具体且有价值的需求，并欢迎大家共同参与，以不断改进PDF-Extract-Kit工具，推动科研及产业发展。


## 协议

本项目采用 [AGPL-3.0](LICENSE) 协议开源。

由于本项目中使用了 YOLO 代码和 PyMuPDF 进行文件处理，这些组件都需要遵循 AGPL-3.0 协议。因此，为了确保遵守这些依赖项的许可证要求，本仓库整体采用 AGPL-3.0 协议。


## 致谢

   - [LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3): 布局检测模型
   - [UniMERNet](https://github.com/opendatalab/UniMERNet): 公式识别模型
   - [StructEqTable](https://github.com/UniModal4Reasoning/StructEqTable-Deploy): 表格识别模型
   - [YOLO](https://github.com/ultralytics/ultralytics): 公式检测模型
   - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): OCR模型
   - [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO): 布局检测模型


## Citation

如果你觉得我们模型/代码/技术报告对你有帮助，请给我们⭐和引用📝,谢谢 :)  
```bibtex
@article{wang2024mineru,
  title={MinerU: An Open-Source Solution for Precise Document Content Extraction},
  author={Wang, Bin and Xu, Chao and Zhao, Xiaomeng and Ouyang, Linke and Wu, Fan and Zhao, Zhiyuan and Xu, Rui and Liu, Kaiwen and Qu, Yuan and Shang, Fukai and others},
  journal={arXiv preprint arXiv:2409.18839},
  year={2024}
}

@misc{wang2024unimernet,
      title={UniMERNet: A Universal Network for Real-World Mathematical Expression Recognition}, 
      author={Bin Wang and Zhuangcheng Gu and Chao Xu and Bo Zhang and Botian Shi and Conghui He},
      year={2024},
      eprint={2404.15254},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{zhao2024doclayoutyoloenhancingdocumentlayout,
      title={DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception}, 
      author={Zhiyuan Zhao and Hengrui Kang and Bin Wang and Conghui He},
      year={2024},
      eprint={2410.12628},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.12628}, 
}

@article{he2024opendatalab,
  title={Opendatalab: Empowering general artificial intelligence with open datasets},
  author={He, Conghui and Li, Wei and Jin, Zhenjiang and Xu, Chao and Wang, Bin and Lin, Dahua},
  journal={arXiv preprint arXiv:2407.13773},
  year={2024}
}
```


## Star历史

<a>
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=opendatalab/PDF-Extract-Kit&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=opendatalab/PDF-Extract-Kit&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=opendatalab/PDF-Extract-Kit&type=Date" />
 </picture>
</a>

## 友情链接
- [UniMERNet（真实场景公式识别算法）](https://github.com/opendatalab/UniMERNet)
- [LabelU（轻量级多模态标注工具）](https://github.com/opendatalab/labelU)
- [LabelLLM（开源LLM对话标注平台）](https://github.com/opendatalab/LabelLLM)
- [MinerU（一站式高质量数据提取工具）](https://github.com/opendatalab/MinerU)