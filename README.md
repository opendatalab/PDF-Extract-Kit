
<p align="center">
  <img src="assets/images/pdf-extract-kit_logo.png" width="220px" style="vertical-align:middle;">
</p>

<div align="center">

English | [简体中文](./README-zh_CN.md)

[[Models (🤗Hugging Face)]](https://huggingface.co/wanderkid/PDF-Extract-Kit) | [[Models(<img src="./assets/images/modelscope_logo.png" width="20px">ModelScope)]](https://www.modelscope.cn/models/wanderkid/PDF-Extract-Kit) 
 
🔥🔥🔥 [MinerU: Efficient Document Content Extraction Tool Based on PDF-Extract-Kit](https://github.com/opendatalab/MinerU)

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/JYsXDXXN" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=MinerU" target="_blank">WeChat</a>
</p>



## Overview

PDF documents contain a wealth of knowledge, yet extracting high-quality content from PDFs is not an easy task. To address this, we have broken down the task of PDF content extraction into several components:
- **Layout Detection**: Using the [LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3) model for region detection, such as `images`, `tables`, `titles`, `text`, etc.;
- **Formula Detection**: Using [YOLOv8](https://github.com/ultralytics/ultralytics) for detecting formulas, including `inline formulas` and `isolated formulas`;
- **Formula Recognition**: Using [UniMERNet](https://github.com/opendatalab/UniMERNet) for formula recognition;
- **Table Recognition**: Using [StructEqTable](https://github.com/UniModal4Reasoning/StructEqTable-Deploy) for table recognition;
- **Optical Character Recognition**: Using [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for text recognition;

> **Note:** *Due to the diversity of document types, existing open-source layout and formula detection models struggle with diverse PDF documents. Therefore, we have collected diverse data for annotation and training to achieve precise detection effects on various types of documents. For details, refer to the sections on [Layout Detection](#layout-anchor) and [Formula Detection](#mfd-anchor). For formula recognition, the UniMERNet method rivals commercial software in quality across various types of formulas. For OCR, we use PaddleOCR, which performs well for both Chinese and English.*

The PDF content extraction framework is illustrated below:

![](assets/demo/pipeline_v2.png)


<details>
  <summary>PDF-Extract-Kit Output Format</summary>

```Bash
{
    "layout_dets": [    # Elements on the page
        {
            "category_id": 0, # Category ID, 0~9, 13~15
            "poly": [
                136.0, # Coordinates are in image format, need to convert back to PDF coordinates, order is top-left, top-right, bottom-right, bottom-left x,y coordinates
                781.0,
                340.0,
                781.0,
                340.0,
                806.0,
                136.0,
                806.0
            ],
            "score": 0.69,   # Confidence score
            "latex": ''      # Formula recognition result, only categories 13, 14 have content, others are empty, additionally 15 is the OCR result, this key will be replaced with text
        },
        ...
    ],
    "page_info": {         # Page information: resolution size when extracting bounding boxes, alignment can be based on this information if scaling is involved
        "page_no": 0,      # Page number
        "height": 1684,    # Page height
        "width": 1200      # Page width
    }
}
```

The types included in `category_id` are as follows:

```
{0: 'title',              # Title
 1: 'plain text',         # Text
 2: 'abandon',            # Includes headers, footers, page numbers, and page annotations
 3: 'figure',             # Image
 4: 'figure_caption',     # Image caption
 5: 'table',              # Table
 6: 'table_caption',      # Table caption
 7: 'table_footnote',     # Table footnote
 8: 'isolate_formula',    # Display formula (this is a layout display formula, lower priority than 14)
 9: 'formula_caption',    # Display formula label

 13: 'inline_formula',    # Inline formula
 14: 'isolated_formula',  # Display formula
 15: 'ocr_text'}          # OCR result
```
</details>

## News and Update
- `2024.08.01` 🎉🎉🎉 Added the [StructEqTable](demo/TabRec/StructEqTable/README_TABLE.md) module for table content extraction. Welcome to use it!
- `2024.07.01` 🎉🎉🎉 We released `PDF-Extract-Kit`, a comprehensive toolkit for high-quality PDF content extraction, including `layout detection`, `formula detection`, `formula recognition`, and `OCR`.


## Visualization of Results

By annotating a variety of PDF documents, we have trained robust models for `layout detection` and `formula detection`. Our pipeline achieves accurate extraction results on diverse types of PDF documents such as academic papers, textbooks, research reports, and financial statements, and is highly robust even in cases of scanned blurriness or watermarks.

![](assets/demo/example.png)


## Evaluation Metrics

Existing open-source models are often trained on data from Arxiv papers and fall short when facing diverse PDF documents. In contrast, our models, trained on diverse data, are capable of adapting to various document types for extraction.

The introduction of Validation process can be seen [here](./assets/validation/README.md).

<span id="layout-anchor"></span>
### Layout Detection

We have compared our model with existing open-source layout detection models, including [DocXchain](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/Applications/DocXChain), [Surya](https://github.com/VikParuchuri/surya), and two models from [360LayoutAnalysis](https://github.com/360AILAB-NLP/360LayoutAnalysis). The model present as LayoutLMv3-SFT in the table refers to the checkpoint we further trained with our SFT data on [LayoutLMv3-base-chinese pre-trained model](https://huggingface.co/microsoft/layoutlmv3-base-chinese). The validation set for academic papers consists of 402 pages, while the textbook validation set is composed of 587 pages from various sources of textbooks.

<table>
    <tr>
        <th align="center" rowspan="2">Model</th> 
        <th colspan="3" align="center">Academic papers val</th> 
        <th colspan="3" align="center">Textbook val</th> 
   </tr>
    <tr>
      	 <th>mAP</th>
         <th>AP50</th>
         <th>AR50</th>
         <th>mAP</th>
         <th>AP50</th>
         <th>AR50</th>    
    </tr>
    <tr>
        <td>DocXchain</td>
        <td>52.8</td>
        <td>69.5</td>
        <td>77.3</td> 
        <td>34.9</td>
        <td>50.1</td>
        <td>63.5</td>   
    </tr>
    <tr>
        <td>Surya</td>
        <td>24.2</td>
        <td>39.4</td>
        <td>66.1</td> 
        <td>13.9</td>
        <td>23.3</td>
        <td>49.9</td>   
    </tr>
    <tr>
        <td>360LayoutAnalysis-Paper</td>
        <td>37.7</td>
        <td>53.6</td>
        <td>59.8</td> 
        <td>20.7</td>
        <td>31.3</td>
        <td>43.6</td>   
    </tr>
    <tr>
        <td>360LayoutAnalysis-Report</td>
        <td>35.1</td>
        <td>46.9</td>
        <td>55.9</td> 
        <td>25.4</td>
        <td>33.7</td>
        <td>45.1</td>   
    </tr>
    <tr>
        <td bgcolor="#f0f0f0">LayoutLMv3-SFT</td>
        <th bgcolor="#f0f0f0">77.6</th>
        <th bgcolor="#f0f0f0">93.3</th>
        <th bgcolor="#f0f0f0">95.5</th> 
        <th bgcolor="#f0f0f0">67.9</th>
        <th bgcolor="#f0f0f0">82.7</th>
        <th bgcolor="#f0f0f0">87.9</th>   
    </tr>
</table>

<span id="mfd-anchor"></span>
### Formula Detection

We have compared our model with the open-source formula detection model [Pix2Text-MFD](https://github.com/breezedeus/pix2text). Additionally, the YOLOv8-Trained is the weight obtained after we performed training on the basis of the [YOLOv8l](https://github.com/ultralytics/) model. The paper's validation set is composed of 255 academic paper pages, and the multi-source validation set consists of 789 pages from various sources, including textbooks and books.

<table>
    <tr>
        <th align="center" rowspan="2">Model</th> 
        <th colspan="2" align="center">Academic papers val</th> 
        <th colspan="2" align="center">Multi-source val</th> 
   </tr>
    <tr>
         <th>AP50</th>
         <th>AR50</th>
         <th>AP50</th>
         <th>AR50</th>    
    </tr>
    <tr>
        <td>Pix2Text-MFD</td>
        <td align="center">60.1</td> 
        <td align="center">64.6</td>
        <td align="center">58.9</td>
        <td align="center">62.8</td>   
    </tr>
    <tr>
        <td bgcolor="#f0f0f0">YOLOv8-Trained</td>
        <th bgcolor="#f0f0f0">87.7</th> 
        <th bgcolor="#f0f0f0">89.9</th>
        <th bgcolor="#f0f0f0">82.4</th>
        <th bgcolor="#f0f0f0">87.3</th>   
    </tr>
</table>

### Formula Recognition
![BLEU](https://github.com/opendatalab/VIGC/assets/69186975/ec8eb3e2-4ccc-4152-b18c-e86b442e2dcc)

The formula recognition we used is based on the weights downloaded from [UniMERNet](https://github.com/opendatalab/UniMERNet), without any further SFT training, and the accuracy validation results can be obtained on its GitHub page.

### Table Recognition
![StructEqTable](assets/demo/table_expamle.png)

The table recognition we used is based on the weights downloaded from [StructEqTable](https://github.com/UniModal4Reasoning/StructEqTable-Deploy), a solution that converts images of Table into LaTeX. Compared to the table recognition capability of PP-StructureV2, StructEqTable demonstrates stronger recognition performance, delivering good results even with complex tables, which may currently be best suited for data within research papers. There is also significant room for improvement in terms of speed, and we are continuously iterating and optimizing. Within a week, we will update the table recognition capability to [MinerU](https://github.com/opendatalab/MinerU).


## Installation Guide

```bash
conda create -n pipeline python=3.10

pip install -r requirements.txt

pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==0.6+pt2.3.1cu121
```

After installation, you may encounter some version conflicts leading to version changes. If you encounter version-related errors, you can try the following commands to reinstall specific versions of the libraries.

```bash
pip install pillow==8.4.0
```

In addition to version conflicts, you may also encounter errors where torch cannot be invoked. First, uninstall the following library and then reinstall cuda12 and cudnn.

```bash
pip uninstall nvidia-cusparse-cu12
```

### Refer to [Model Download](models/README.md) to download the required model weights.


## Running on Windows

If you intend to run this project on Windows, please refer to [Using PDF-Extract-Kit on Windows](docs/Install_in_Windows_en.md).


## Running on macOS

If you intend to run this project on macOS, please refer to [Using PDF-Extract-Kit on macOS](docs/Install_in_macOS_en.md).

## Running in Google Colab

If you intend to experience this project on Google Colab, please <a href="https://colab.research.google.com/gist/zhchbin/f7ca974b3594befe59893241d6ad6374/pdf-extract-kit.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Run Extraction Script

```bash 
python pdf_extract.py --pdf data/pdfs/ocr_1.pdf
```

Parameter explanations:
- `--pdf`: PDF file to be processed; if a folder is passed, all PDF files in the folder will be processed.
- `--output`: Path where the results are saved, default is "output".
- `--vis`: Whether to visualize the results; if yes, detection results including bounding boxes and categories will be visualized.
- `--render`: Whether to render the recognized results, including LaTeX code for formulas and plain text, which will be rendered and placed in the detection boxes. Note: This process is very time-consuming, and also requires prior installation of `xelatex` and `imagemagic`.
- `--batch-size`: Batch size for dataloader. Larger batch sizes are recommended, but smaller sizes require less GPU memory. Default is 128.

> This project is dedicated to using models for high-quality content extraction from documents on diversity. It does not involve reassembling the extracted content into new documents, such as converting PDFs to Markdown. For those needs, please refer to our other GitHub project: [MinerU](https://github.com/opendatalab/MinerU)

## TODO List

- [x] **Table Parsing**: Develop a feature to convert table images into corresponding LaTeX/Markdown format source code.
- [ ] **Chemical Equation Detection**: Implement automatic detection of chemical equations.
- [ ] **Chemical Equation/Diagram Recognition**: Develop a model to recognize and parse chemical equations and diagrams.
- [ ] **Reading Order Sorting Model**: Build a model to determine the correct reading order of text in documents.

**PDF-Extract-Kit** aims to provide high-quality PDF extraction capabilities. We encourage the community to propose specific and valuable requirements and welcome everyone to participate in continuously improving the PDF-Extract-Kit tool to advance scientific research and industrial development.

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

Please follow the model licenses to use the corresponding model weights: [LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3) / [UniMERNet](https://github.com/opendatalab/UniMERNet) / [StructEqTable](https://github.com/UniModal4Reasoning/StructEqTable-Deploy) / [YOLOv8](https://github.com/ultralytics/ultralytics) / [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).


## Acknowledgement

   - [LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3): Layout detection model
   - [UniMERNet](https://github.com/opendatalab/UniMERNet): Formula recognition model
   - [StructEqTable](https://github.com/UniModal4Reasoning/StructEqTable-Deploy): Table recognition model
   - [YOLOv8](https://github.com/ultralytics/ultralytics): Formula detection model
   - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): OCR model

## Citation
If you find our models / code / papers useful in your research, please consider giving ⭐ and citations 📝, thx :)  
```bibtex
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

## Links
- [LabelU (A Lightweight Multi-modal Data Annotation Tool)](https://github.com/opendatalab/labelU)
- [LabelLLM (An Open-source LLM Dialogue Annotation Platform)](https://github.com/opendatalab/LabelLLM)
- [Miner U (A One-stop Open-source High-quality Data Extraction Tool)](https://github.com/opendatalab/MinerU)
