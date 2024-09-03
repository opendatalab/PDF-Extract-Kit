# Validation

During the model training and updating process, we follow the validation process provided on its GitHub for each model to test the ability of the trained models. If there is no validation code provided, we have developed it based on its code. For details, please refer to:

- **Layout Detection**: Using the [LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3);
- **Formula Detection**: Using [YOLOv8](https://github.com/ultralytics/ultralytics);

**Formula Recognition** and **Optical Character Recognition** using the existing weight provided on [UniMERNet](https://github.com/opendatalab/UniMERNet) and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), so no validation process involved.

In addition, if you wish to directly verify the results output by this pipeline, we have also provided a script for reference.

Due to copyright reasons, the validation datasets cannot be made public.

## Layout Detection

For Layout Detection, we use the validation process officiently provided in [LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3):

```
python train_net.py --config-file config.yaml --eval-only --num-gpus 8 \
        MODEL.WEIGHTS /path/to/your/model_final.pth \
        OUTPUT_DIR /path/to/save/dir
```

For validation of other open-source models, we directly used the inference code provided in each model's GitHub repository without any special settings. For details, please refer to:
    - [Surya](https://github.com/VikParuchuri/surya?tab=readme-ov-file#layout-analysis)
    - [360LayoutAnalysis](https://github.com/360AILAB-NLP/360LayoutAnalysis/blob/main/README_EN.md)

The evaluation matrix is provided by mmeval using [COCODetection](https://mmeval.readthedocs.io/zh-cn/latest/api/generated/mmeval.metrics.COCODetection.html).

Due to the fact that the category labels of each model are not completely consistent, we implement the category mapping for alignment before the validation process.

### Surya

```python
# Participated categories in the validation
label_classes = ["title", "plain text", "abandon", "figure", "caption", "table", "isolate_formula"] 

# Ground Truth Category Mapping (original categories aligned with the LayoutLmv3-SFT fine-tuned in this repository)
anno_class_change_dict = {
    'formula_caption': 'caption',
    'table_caption': 'caption',
    'table_footnote': 'plain text'
}

# Surya Category Mapping
class_dict = {
    'Caption': 'caption',
    'Section-header' : 'title',
    'Title': 'title',
    'Figure': 'figure',
    'Picture': 'figure',
    'Footnote': 'abandon',
    'Page-footer': 'abandon',
    'Page-header': 'abandon',
    'Table': 'table',
    'Text': 'plain text',
    'List-item': 'plain text',
    'Formula': 'isolate_formula',
}
```

### 360LayoutAnalysis-Paper

```python
# Participated categories in the validation
label_classes = ["title", "plain text", "abandon", "figure", "figure_caption", "table", "table_caption", "isolate_formula"]

# Ground Truth Category Mapping
anno_class_change_dict = {
    'formula_caption': 'plain text',
    'table_footnote': 'plain text'
}

# 360LayoutAnalysis Category Mapping
class_change_dict = {
    'Text': 'plain text',  
    'Title': 'title', 
    'Figure': 'figure', 
    'Figure caption': 'figure_caption',    
    'Table': 'table',      
    'Table caption': 'table_caption',  
    'Header': 'abandon', 
    'Footer': 'abandon',     
    'Reference': 'plain text',   
    'Equation': 'isolate_formula',
    'Toc': 'plain text'   
}
```

### 360LayoutAnalysis-Report

```python
# Participated categories in the validation
label_classes = ["title", "plain text", "abandon", "figure", "figure_caption", "table", "table_caption"]

# Ground Truth Category Mapping
anno_class_change_dict = {
    'formula_caption': 'plain text',
    'table_footnote': 'plain text',
    'isolate_formula': 'plain text',
}

# 360LayoutAnalysis Category Mapping
class_change_dict = {
    'Text': 'plain text',  
    'Title': 'title', 
    'Figure': 'figure', 
    'Figure caption': 'figure_caption',    
    'Table': 'table',      
    'Table caption': 'table_caption',  
    'Header': 'abandon', 
    'Footer': 'abandon',     
    'Reference': 'plain text',   
    'Equation': 'isolate_formula',
    'Toc': 'plain text'   
}
```

## Formula Detection

For Formula Detection, we have developed validation process based on [YOLOv8](https://github.com/ultralytics/ultralytics).

Firstly, put the python file we provided in `./modules/yolov8/mfd_val.py` to `~/ultralytics/models/yolo/detect`, which means to add a new class named MFDValidator.

Sencondly, place the required YAML file in the directory `~/ultralytics/cfg/mfd_dataset`. Here is an example provided: `./modules/yolov8/opendata.yaml`.

Lastly, place the validation code directly in the `~/ultralytics/` directory. The validation code is located at `./modules/yolov8/eval_mfd.py`.

The script for running can be referred to at `./modules/yolov8/eval_mfd_1888.sh`. The command to run is as follows:

```
bash eval_mfd_1888.sh /path/to/your/trained/yolov8/weights
```

Note that the default image size used here is 1888, which can be set through the `--imsize` parameter.

## Pipeline Output Verification

The format of the Pipeline output has been shown in the [README](../../README-zh_CN.md), please prepare the validation dataset according to this format.

We provide a code for directly verifying the Pipeline output and a demo data (not real data, does not represent the actual accuracy of this pipeline), please run the following command directly in this directory:

```
python pdf_validation.py
```
