# Inference Accelerated PDF parsing
This fold include a series infra-accelerate modules for origin PDF parsing, including:
- intergrated preprocessing into dataloader
- fast_postprocessing
- torch.compile + bf16
- tensorRT
- Torch-TensorRT[https://pytorch.org/TensorRT/]

Those engine is tested on a 80,000,000 pdf dataset and get a 5-10x speedup compared with the origin pdf parsing engine. Basicly, it can reach 6-10 pages per second on a single A100 GPU.

This is not a pipline framework but seperated into three task-wise batch processing engine. But it can be easily integrated into your own pipline framework. 
- Detection (Bounding Boxing)
- Recognition (OCR)
- Math formula recognition (MFR)

## Detection (Bounding Boxing)
Check the unit case:1000pdf takes around 20-30min
```
    python batch_running_task/task_layout/rough_layout.py
```
### LayoutLM
    The layoutLM is based on the `detectron2`. The main Vision Engine(ViT) is implemented via huggingface, the postprocess is based on detectron.
    There is a tensorRT version of the detectron model https://github.com/NVIDIA/TensorRT/tree/main/samples/python/detectron2 , but it is only for Mask R-CNN backbone.
    The tensorRT author manuelly develop the CUDA NMS and ROIAlign such as `DET2GraphSurgeon` (see https://github.com/NVIDIA/TensorRT/blob/main/samples/python/detectron2/create_onnx.py) to convert the detectron2 model to tensorRT model.
    For layoutLM, there is no such tool to convert whole model into a tensorRT engine. 
    There are serveral ways to accelerate the layoutLM model:
    - accelerate part by part, such as the tensorRT ViT backbone with detectron ROIAlign, NMS.
    - use torch.compile
    - use bf16

    In this repo, I use the torch.compile(1.5x) and bf16(2x) to accelerate the layoutLM model. The tensorRT version is not implemented yet.

    Another way to accelerate the layoutLM is `avoid .numpy() large GPU tensor`. Origin code will use 
    ```
    boxes = outputs["instances"].to("cpu")._fields["pred_boxes"].tensor.numpy()
    labels = outputs["instances"].to("cpu")._fields["pred_classes"].numpy()
    scores = outputs["instances"].to("cpu")._fields["scores"].numpy()
    ```
    This will copy the large tensor from GPU to CPU. (Since we later only gather part of data via `mask`, full tensor copy is unnecessary). 
    The better way is to do slicing on GPU tensor and then copy the sliced tensor to CPU. (2x) (see batch_running_task/task_layout/get_batch_layout_model.py)
    
    
### MFD
    MFD(Math Formula Detection) is a simple YOLO model build through `ultralytics`. It has a good tensorRT convert tool chain. See https://docs.ultralytics.com/modes/export/ and convension/MDF/convert.py
    Download the engine via `huggingface-cli download --resume-download --local-dir-use-symlinks False LLM4SCIENCE/ultralytics-YOLO-MFD --local-dir models/MFD`. The `batchsize` and `tensorRT version==10.3.0` must match! if you want to use the `trt_engine` directly.
    
### PaddleOCR-Det
    PaddleOCR-Det is the best text detecter around the world. But original paddle det only support one image per batch. In our detection task, every image is normlized into same size, so the original paddle det does not fit our task. Refer to `https://github.com/WenmuZhou/PytorchOCR`, Zhou has convert the paddleOCR into pytorch. It allow us use batch detection in pytorch now.

    There is a big speed up possiblity for the postprocessing for the paddleOCR-Det module. Currently, we use the DB postprocessing. See `https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/postprocess/db_postprocess.py`. The DB postprocessing is the slow part compare to whole detection process. Currently, there is no any speedup solution for the DB postprocessing.

### Detection Async(experimental)
    See `batch_running_task/task_layout/rough_layout_with_aync.py`
    The async detection is a way to async postprocess and GPU inference. It works perfectly. But in slurm system, there is `exit` error when run the script, this will make your machine `CPU soft lock`. So, I do not recommend to use this script in slurm system.

## Recognition (OCR)
    Check the unit case:1000 pdf takes around 2-5 min
    ```
        python batch_running_task/task_rec/rough_rec.py
    ```
    PaddleOCR-Rec is the best text recognizer around the world. The original paddle rec support batch image processing. And the origin paddleOCR "is already very fast".
    However, you can see I still use `PytorchOCR` in this part. Just want to provide a non-paddle solution. 
    Download the engine via `huggingface-cli download --resume-download --local-dir-use-symlinks False LLM4SCIENCE/pytorch_paddle_weight --local-dir models/pytorch_paddle_weight `. The `batchsize` and `tensorRT version==10.3.0` must match! if you want to use the `trt_engine` directly.
    

## Math formula recognition (MFR)
    Check the unit case: 1000 pdf takes around 2-5min
    ```
        python batch_running_task/task_mfr/rough_mfr.py
    ```
    MFR model is `nougat` based model named `UniMERNet`. I tried to use Huggingface tensorRT convert tool chain to convert the model into tensorRT. But it failed. (The reshape module is not set properly). One way is using the `TensorRT-LLM`, see `https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal` and `convension/unimernet`. 
    - Notice `TensorRT-LLM` will default install `mpi4py=4.*.*` which will require `mpi.so40`. The conda `conda install -c conda-forge openmpi` can only support `openmpi==3.*.*'. So you need to install `openmpi` from source. Or, you can just `pip install mpi4py==3.*`.
    - Notice you should `srun --mpi=pmi2` when run script in slurm.

    Download the engine via `huggingface-cli download --resume-download --local-dir-use-symlinks False LLM4SCIENCE/unimernet --local-dir models/MFR/unimernet`. The `batchsize` and `tensorRT version==10.3.0` must match! if you want to use the `trt_engine` directly.

    The different between `LLM4SCIENCE/unimernet` and `wanderkid/unimernet` is we delete the `counting` module in weight file. (it only works in training). And it is a pure nougat model.


## Batch run the task
    Each task has a "batch_deal_with_xxx" module which will automatively schedule task. For example, your can prepare a `.jsonl` file named `test.filelist` with each line is 
    ``` 
    {"track_id":"e8824f5a-9fcb-4ee5-b2d4-6bf2c67019dc","path":"10.1017/cbo9780511770425.012.pdf","file_type":"pdf","content_type":"application/pdf","content_length":80078,"title":"German Idealism and the Concept of Punishment || Conclusion","remark":{"file_id":"cbo9780511770425.012","file_source_type":"paper","original_file_id":"10.1017/cbo9780511770425.012","file_name":"10.1017/cbo9780511770425.012.pdf","author":"Merle, Jean-Christophe"}}
{"track_id":"64d182ba-21bf-478f-bb65-6a276aab3f4d","path":"10.1111/j.1365-2559.2006.02442.x.pdf","file_type":"pdf","content_type":"application/pdf","content_length":493629,"title":"Sensitivity and specificity of immunohistochemical antibodies used to distinguish between benign and malignant pleural disease: a systematic review of published reports","remark":{"file_id":"j.1365-2559.2006.02442.x","file_source_type":"paper","original_file_id":"10.1111/j.1365-2559.2006.02442.x","file_name":"10.1111/j.1365-2559.2006.02442.x.pdf","author":"J King; N Thatcher; C Pickering; P Hasleton"}}
    ```
    and then run 
    ```
    python batch_running_task/task_layout/batch_deal_with_layout.py --root test.filelist
    python batch_running_task/task_layout/batch_deal_with_rec.py --root test.filelist
    python batch_running_task/task_layout/batch_deal_with_mfr.py --root test.filelist
    ```
## 

## Batch Postprocess (Pure CPU task)
    We provide series of postprocess module to deal with the missing page and merge the layout, rec, mfr result into a single json file, and so on.
    Below is a list of postprocess module:
    - check the pdf status for each page: `script\scan_and_get_page_num_map\scan_and_get_page_num_map.py`
    - get the page ocr status: `script\scan_and_judge_det_ocr_status\scan_and_judge_ready_for_ocr.py`
    - merge layout, rec, mfr result: `script\merge_layout_rec_mfr_result\script\merge_all_patch_back\merge_all_patch_back.py`
    - format the final result: `script\format_into_markdown\format_into_markdown.py`
    - .....

    You may find a script named `batch_job_dispatch.sh` at each fold. It looks like: (We default use Slurm cluster)
    ```
    PARTION_TOTAL=320 #32 #
    PARTION_INTERVAL=16 # 32
    PARTION_NUM=$(($PARTION_TOTAL / $PARTION_INTERVAL))
    #echo $PARTION_NUM
    for ((i=0; i<PARTION_NUM; i++));
    do  
        START=$((($i)*$PARTION_INTERVAL))
        sbatch  -p ai4earth -N1 -c20 --gres=gpu:0 script/scan_and_get_page_num_map/job_dispatch.sh $FILEPATH $START $PARTION_INTERVAL $PARTION_TOTAL $JOBSCRIPT
        sleep 1
    done
    ```
    It will automative partition the `$FILEPATH` into `$PARTION_TOTAL` part and do multi-thread processing at each compute node.

    The `$FILEPATH` is the filepath list file. It should ends with `.filelist`. Each line is the path to the information bulk file endswith `.jsonl`.(Same as above)

    A quick example to use the code is directly run the script in `batch_job_dispatch.sh` with correctly assigning `$JOBSCRIPT` and `$FILEPATH` (Example can see `script\scan_and_get_page_num_map\batch_job_dispatch.sh`)

## Deal with missing page
    After running `script\scan_and_judge_det_ocr_status\scan_and_judge_ready_for_ocr.py`, it will create the status list for each page of each pdf in each jsonl file.
    
    In `batch_running_task.task_layout.rough_layout.deal_with_page_info_dataset_for_missing_page`, it accept an extra keyword name `page_num_for_name` which means the process-waiting page for each pdf for current bulk jsonl. It can help you fast to add patch for the missing page and missing pdf.


