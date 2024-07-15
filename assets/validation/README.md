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
