
<p align="center">
  <img src="assets/readme/pdf-extract-kit_logo.png" width="220px" style="vertical-align:middle;">
</p>

<div align="center">

English | [简体中文](./README_zh-CN.md)

[PDF-Extract-Kit-1.0 Tutorial](https://pdf-extract-kit.readthedocs.io/en/latest/get_started/pretrained_model.html)

[[Models (🤗Hugging Face)]](https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0) | [[Models(<img src="./assets/readme/modelscope_logo.png" width="20px">ModelScope)]](https://www.modelscope.cn/models/OpenDataLab/PDF-Extract-Kit-1.0) 
 
🔥🔥🔥 [MinerU: Efficient Document Content Extraction Tool Based on PDF-Extract-Kit](https://github.com/opendatalab/MinerU)

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/Tdedn9GTXq" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=MinerU" target="_blank">WeChat</a>
</p>


## Overview

`PDF-Extract-Kit` is a powerful open-source toolkit designed to efficiently extract high-quality content from complex and diverse PDF documents. Here are its main features and advantages:

- **Integration of Leading Document Parsing Models**: Incorporates state-of-the-art models for layout detection, formula detection, formula recognition, OCR, and other core document parsing tasks.
- **High-Quality Parsing Across Diverse Documents**: Fine-tuned with diverse document annotation data to deliver high-quality results across various complex document types.
- **Modular Design**: The flexible modular design allows users to easily combine and construct various applications by modifying configuration files and minimal code, making application building as straightforward as stacking blocks.
- **Comprehensive Evaluation Benchmarks**: Provides diverse and comprehensive PDF evaluation benchmarks, enabling users to choose the most suitable model based on evaluation results.

**Experience PDF-Extract-Kit now and unlock the limitless potential of PDF documents!**

> **Note:** PDF-Extract-Kit is designed for high-quality document processing and functions as a model toolbox.    
> If you are interested in extracting high-quality document content (e.g., converting PDFs to Markdown), please use [MinerU](https://github.com/opendatalab/MinerU), which combines the high-quality predictions from PDF-Extract-Kit with specialized engineering optimizations for more convenient and efficient content extraction.    
> If you're a developer looking to create engaging applications such as document translation, document Q&A, or document assistants, you'll find it very convenient to build your own projects using PDF-Extract-Kit. In particular, we will periodically update the PDF-Extract-Kit/project directory with interesting applications, so stay tuned!

**We welcome researchers and engineers from the community to contribute outstanding models and innovative applications by submitting PRs to become contributors to the PDF-Extract-Kit project.**

## Model Overview

| **Task Type**     | **Description**                                                                 | **Models**                    |
|-------------------|---------------------------------------------------------------------------------|-------------------------------|
| **Layout Detection** | Locate different elements in a document: including images, tables, text, titles, formulas | `DocLayout-YOLO_ft`, `YOLO-v10_ft`, `LayoutLMv3_ft` | 
| **Formula Detection** | Locate formulas in documents: including inline and block formulas            | `YOLOv8_ft`                   |  
| **Formula Recognition** | Recognize formula images into LaTeX source code                             | `UniMERNet`                   |  
| **OCR**           | Extract text content from images (including location and recognition)            | `PaddleOCR`                   | 
| **Table Recognition** | Recognize table images into corresponding source code (LaTeX/HTML/Markdown)   | `PaddleOCR+TableMaster`, `StructEqTable` |  
| **Reading Order** | Sort and concatenate discrete text paragraphs                                    | Coming Soon!                  | 

## News and Updates
- `2024.10.22` 🎉🎉🎉 We are excited to announce that table recognition model [StructTable-InternVL2-1B](https://huggingface.co/U4R/StructTable-InternVL2-1B), which supports output LaTeX, HTML and MarkdDown formats has been officially integrated into `PDF-Extract-Kit 1.0`. Please refer to the [table recognition algorithm documentation](https://pdf-extract-kit.readthedocs.io/en/latest/algorithm/table_recognition.html) for usage instructions!
- `2024.10.17` 🎉🎉🎉 We are excited to announce that the more accurate and faster layout detection model, [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO), has been officially integrated into `PDF-Extract-Kit 1.0`. Please refer to the [layout detection algorithm documentation](https://pdf-extract-kit.readthedocs.io/en/latest/algorithm/layout_detection.html) for usage instructions!
- `2024.10.10` 🎉🎉🎉 The official release of `PDF-Extract-Kit 1.0`, rebuilt with modularity for more convenient and flexible model usage! Please switch to the [release/0.1.1](https://github.com/opendatalab/PDF-Extract-Kit/tree/release/0.1.1) branch for the old version.
- `2024.08.01` 🎉🎉🎉 Added the [StructEqTable](demo/TabRec/StructEqTable/README_TABLE.md) module for table content extraction. Welcome to use it!
- `2024.07.01` 🎉🎉🎉 We released `PDF-Extract-Kit`, a comprehensive toolkit for high-quality PDF content extraction, including `Layout Detection`, `Formula Detection`, `Formula Recognition`, and `OCR`.

## Performance Demonstration

Many current open-source SOTA models are trained and evaluated on academic datasets, achieving high-quality results only on single document types. To enable models to achieve stable and robust high-quality results on diverse documents, we constructed diverse fine-tuning datasets and fine-tuned some SOTA models to obtain practical parsing models. Below are some visual results of the models.

### Layout Detection

We trained robust `Layout Detection` models using diverse PDF document annotations. Our fine-tuned models achieve accurate extraction results on diverse PDF documents such as papers, textbooks, research reports, and financial reports, and demonstrate high robustness to challenges like blurring and watermarks. The visualization example below shows the inference results of the fine-tuned LayoutLMv3 model.
 
![](assets/readme/layout_example.png)

### Formula Detection

Similarly, we collected and annotated documents containing formulas in both English and Chinese, and fine-tuned advanced formula detection models. The visualization result below shows the inference results of the fine-tuned YOLO formula detection model:

![](assets/readme/mfd_example.png)

### Formula Recognition

[UniMERNet](https://github.com/opendatalab/UniMERNet) is an algorithm designed for diverse formula recognition in real-world scenarios. By constructing large-scale training data and carefully designed results, it achieves excellent recognition performance for complex long formulas, handwritten formulas, and noisy screenshot formulas.

### Table Recognition

[StructEqTable](https://github.com/UniModal4Reasoning/StructEqTable-Deploy) is a high efficiency toolkit that can converts table images into LaTeX/HTML/MarkDown. The latest version, powered by the InternVL2-1B foundation model,  improves Chinese recognition accuracy and expands multi-format output options.

#### For more visual and inference results of the models, please refer to the [PDF-Extract-Kit tutorial documentation](xxx).

## Evaluation Metrics

Coming Soon!

## Usage Guide

### Environment Setup

```bash
conda create -n pdf-extract-kit-1.0 python=3.10
conda activate pdf-extract-kit-1.0
pip install -r requirements.txt
```
> **Note:** If your device does not support GPU, please install the CPU version dependencies using `requirements-cpu.txt` instead of `requirements.txt`.

> **Note：** Current Doclayout-YOLO only supports installation from pypi，if error raises during DocLayout-YOLO installation，please install through `pip3 install doclayout-yolo==0.0.2 --extra-index-url=https://pypi.org/simple` .

### Model Download

Please refer to the [Model Weights Download Tutorial](https://pdf-extract-kit.readthedocs.io/en/latest/get_started/pretrained_model.html) to download the required model weights. Note: You can choose to download all the weights or select specific ones. For detailed instructions, please refer to the tutorial.

### Running Demos

#### Layout Detection Model

```bash 
python scripts/layout_detection.py --config=configs/layout_detection.yaml
```
Layout detection models support **DocLayout-YOLO** (default model), YOLO-v10, and LayoutLMv3. For YOLO-v10 and LayoutLMv3, please refer to [Layout Detection Algorithm](https://pdf-extract-kit.readthedocs.io/en/latest/algorithm/layout_detection.html). You can view the layout detection results in the `outputs/layout_detection` folder.

#### Formula Detection Model

```bash 
python scripts/formula_detection.py --config=configs/formula_detection.yaml
```
You can view the formula detection results in the `outputs/formula_detection` folder.

#### OCR Model

```bash 
python scripts/ocr.py --config=configs/ocr.yaml
```
You can view the OCR results in the `outputs/ocr` folder.

#### Formula Recognition Model

```bash 
python scripts/formula_recognition.py --config=configs/formula_recognition.yaml
```
You can view the formula recognition results in the `outputs/formula_recognition` folder.

#### Table Recognition Model

```bash 
python scripts/table_parsing.py --config configs/table_parsing.yaml
```
You can view the table recognition results in the `outputs/table_parsing` folder.

> **Note:** For more details on using the model, please refer to the[PDF-Extract-Kit-1.0 Tutorial](https://pdf-extract-kit.readthedocs.io/en/latest/get_started/pretrained_model.html).

> This project focuses on using models for `high-quality` content extraction from `diverse` documents and does not involve reconstructing extracted content into new documents, such as PDF to Markdown. For such needs, please refer to our other GitHub project: [MinerU](https://github.com/opendatalab/MinerU).

## To-Do List

- [x] **Table Parsing**: Develop functionality to convert table images into corresponding LaTeX/Markdown format source code.
- [ ] **Chemical Equation Detection**: Implement automatic detection of chemical equations.
- [ ] **Chemical Equation/Diagram Recognition**: Develop models to recognize and parse chemical equations and diagrams.
- [ ] **Reading Order Sorting Model**: Build a model to determine the correct reading order of text in documents.

**PDF-Extract-Kit** aims to provide high-quality PDF content extraction capabilities. We encourage the community to propose specific and valuable needs and welcome everyone to participate in continuously improving the PDF-Extract-Kit tool to advance research and industry development.

## License

This project is open-sourced under the [AGPL-3.0](LICENSE) license.

Since this project uses YOLO code and PyMuPDF for file processing, these components require compliance with the AGPL-3.0 license. Therefore, to ensure adherence to the licensing requirements of these dependencies, this repository as a whole adopts the AGPL-3.0 license.

## Acknowledgement

   - [LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3): Layout detection model
   - [UniMERNet](https://github.com/opendatalab/UniMERNet): Formula recognition model
   - [StructEqTable](https://github.com/UniModal4Reasoning/StructEqTable-Deploy): Table recognition model
   - [YOLO](https://github.com/ultralytics/ultralytics): Formula detection model
   - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): OCR model
   - [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO): Layout detection model

## Citation
If you find our models / code / papers useful in your research, please consider giving ⭐ and citations 📝, thx :)  
```bibtex
@article{wang2024mineru,
  title={MinerU: An Open-Source Solution for Precise Document Content Extraction},
  author={Wang, Bin and Xu, Chao and Zhao, Xiaomeng and Ouyang, Linke and Wu, Fan and Zhao, Zhiyuan and Xu, Rui and Liu, Kaiwen and Qu, Yuan and Shang, Fukai and others},
  journal={arXiv preprint arXiv:2409.18839},
  year={2024}
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

@misc{wang2024unimernet,
      title={UniMERNet: A Universal Network for Real-World Mathematical Expression Recognition}, 
      author={Bin Wang and Zhuangcheng Gu and Chao Xu and Bo Zhang and Botian Shi and Conghui He},
      year={2024},
      eprint={2404.15254},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{he2024opendatalab,
  title={Opendatalab: Empowering general artificial intelligence with open datasets},
  author={He, Conghui and Li, Wei and Jin, Zhenjiang and Xu, Chao and Wang, Bin and Lin, Dahua},
  journal={arXiv preprint arXiv:2407.13773},
  year={2024}
}
```

## Star History

<a>
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=opendatalab/PDF-Extract-Kit&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=opendatalab/PDF-Extract-Kit&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=opendatalab/PDF-Extract-Kit&type=Date" />
 </picture>
</a>

## Related Links
- [UniMERNet (Real-World Formula Recognition Algorithm)](https://github.com/opendatalab/UniMERNet)
- [LabelU (Lightweight Multimodal Annotation Tool)](https://github.com/opendatalab/labelU)
- [LabelLLM (Open Source LLM Dialogue Annotation Platform)](https://github.com/opendatalab/LabelLLM)
- [MinerU (One-Stop High-Quality Data Extraction Tool)](https://github.com/opendatalab/MinerU)
