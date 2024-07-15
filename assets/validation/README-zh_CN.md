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


