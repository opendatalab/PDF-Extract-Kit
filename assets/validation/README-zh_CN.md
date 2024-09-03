# 验证

在模型迭代的过程中，我们遵循各个模型各自的GitHub上提供的验证代码来输出验证结果，如果没有合适的验证代码，我们在其代码基础上进行了开发，详情请参考：

- 布局检测：使用[LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3)；
- 公式检测：使用[YOLOv8](https://github.com/ultralytics/ultralytics)；

公式识别和光学字符识别我们使用的是[UniMERNet](https://github.com/opendatalab/UniMERNet)和[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)官方提供的权重，没有做进一步的训练和验证，因此不涉及验证代码。

除此之外，如果想要直接对本pipeline输出的结果进行验证，我们也提供了一个脚本供参考。

验证数据由于版权原因无法公开。

## 布局检测

布局检测使用的是[LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3)官方提供的验证代码：

```
python train_net.py --config-file config.yaml --eval-only --num-gpus 8 \
        MODEL.WEIGHTS /path/to/your/model_final.pth \
        OUTPUT_DIR /path/to/save/dir
```

与其他开源模型的对比部分，我们没有对其模型做特殊参数设置，评测时候用每个模型repo里提供的代码直接进行推理，详情请参考：
    - [Surya](https://github.com/VikParuchuri/surya?tab=readme-ov-file#layout-analysis)
    - [360LayoutAnalysis](https://github.com/360AILAB-NLP/360LayoutAnalysis/blob/main/README_EN.md)

评测结果输出依赖于mmeval的[COCODetection](https://mmeval.readthedocs.io/zh-cn/latest/api/generated/mmeval.metrics.COCODetection.html)。

由于每个模型的类别标签不完全一致，我们在验证的时候对标签进行了映射对齐，具体如下：

### Surya

```python
# 参与验证的类别
label_classes = ["title", "plain text", "abandon", "figure", "caption", "table", "isolate_formula"] 

# GT的类别映射 (原本的类别与本repo微调的LayoutLmv3-SFT对齐)
anno_class_change_dict = {
    'formula_caption': 'caption',
    'table_caption': 'caption',
    'table_footnote': 'plain text'
}

# Surya的类别映射
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
# 参与验证的类别
label_classes = ["title", "plain text", "abandon", "figure", "figure_caption", "table", "table_caption", "isolate_formula"]

# GT的类别映射表
anno_class_change_dict = {
    'formula_caption': 'plain text',
    'table_footnote': 'plain text'
}

# 360LayoutAnalysis的类别映射
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
# 参与验证的类别
label_classes = ["title", "plain text", "abandon", "figure", "figure_caption", "table", "table_caption"]

# GT的类别映射表
anno_class_change_dict = {
    'formula_caption': 'plain text',
    'table_footnote': 'plain text',
    'isolate_formula': 'plain text',
}

# 360LayoutAnalysis的类别映射
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

## 公式检测

公式检测的部分，我们在[YOLOv8](https://github.com/ultralytics/ultralytics)的基础上新增了验证代码。

首先，需要将`./modules/yolov8/mfd_val.py`放在`~/ultralytics/models/yolo/detect`路径下，作用是新增MFDValidator类别。

然后将需要用到的yaml文件放在`~/ultralytics/cfg/mfd_dataset`下，这里给了一个示例：`./modules/yolov8/opendata.yaml`。

最后将验证的代码直接放在`~/ultralytics/`路径下，验证代码在`./modules/yolov8/eval_mfd.py`。

运行的脚本可以参考`./modules/yolov8/eval_mfd_1888.sh`，具体运行的命令如下：

```
bash eval_mfd_1888.sh /path/to/your/trained/yolov8/weights
```

注意，这里用的图像大小默认是1888，可以通过--imsize参数设置。

## Pipeline输出验证

Pipeline输出结果的格式已经在[README](../../README-zh_CN.md)中展示，请参考这个格式准备验证数据。

我们提供了一个直接验证Pipeline输出结果的代码和示例数据（非真实数据，不代表本pipeline真实验证结果），请直接在本目录下运行以下命令：

```
python pdf_validation.py
```


