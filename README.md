## 安装教程

1. 创建环境并安装依赖

```
conda create -n pipeline python=3.10

pip install unimernet

pip install -r requirements.txt

pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==0.6+pt2.2.2cu121

pip uninstall PyMuPDF

pip install PyMuPDF==1.20.2

pip install pillow==8.4.0
```

最后三行是解决版本冲突的，如果代码没有报错也可以不执行。

2. 安装cuda12

首先需要卸载上一步安装过程中自动安装的一个依赖（不然调用torch会报错）：

```
pip uninstall nvidia-cusparse-cu12
```

安装cuda12，如果在集群上，可以在环境变量里加上下面两句话并重新source：
```
export PATH="/mnt/lustre/share/cuda-12.0/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/lustre/share/cuda-12.0/lib64:$LD_LIBRARY_PATH"
```

## 使用方法

1. 处理本地pdf文件

```srun
srun -p s2_bigdata --gres=gpu:1 --async python process_pdf.py --pdf data/pdfs/ocr_1.pdf
```

如果本地有多个pdf文件，可以把文件放在一个目录，提交命令的时候把`--pdf`参数指向这个目录的路径即可，如：`--pdf data/pdfs/`.

2. 批量处理enroll文件

```srun
bash run_enroll.sh

nohup bash run_enroll.sh &     # 挂在后台处理
```



一些参数定义在`global_args.yaml`文件里面，很多处理流程需要的参数会在这里读取, 在执行命令前可以根据实际情况修改，参数如下：

```
model_args:
    模型权重文件的路径，检测的参数等等，此处不一一展开；
s3_args:
    input_s3_dir: 需要处理的enroll文件所在的路径；
    output_s3_dir: 处理结果的保存路径；
run_args:
    input_list_file: 待处理列表，一般不需要修改
    output_list_file: 已处理进度，一般不需要修改
    run_num: 最大处理的jsonl文件数量，想全量处理的话，取一个较大的数即可
    gpu_quota: 批量运行的时候，最多占用多少gpu
```

处理enroll版本的时候需要从ceph上读取数据，需要生成一个配置文件`~/.aws/s3_bucket_cfg.json`来保存不同bucket的配置，内容大致如下：
```
{
    "bucket_name": {'endpoint': 'http://xxxx', 'ak': 'xxx', 'sk': 'xxx'},
    ...
}
```

3. 一些特殊的工具

- 查看当前已处理和待处理的文件列表个数：`python show_enroll_list.py`， 如果需要更新input_list_file和output_list_file，需要加上`--update`.(一般更新了`global_args.yaml`文件里的输入输出路径，这里也需要同步进行更新)

- 统计日志中的异常： `python analyze_log.py`

- 停止所有srun进程：`bash shut_all.sh`

- 删除已完成和异常退出的日志文件：`bash clean_srun_log.sh`

## 处理结果

pipelin处理结果的格式如下：

```Bash
{
    "layout_dets": [    # 页中的元素
        {
            "category_id": 0, # 类别编号， 0~9，13~15
            "poly": [
                136.0, # 坐标为图片坐标，需要转换回pdf坐标, 顺序是 左上-右上-右下-左下的x,y坐标
                781.0,
                340.0,
                781.0,
                340.0,
                806.0,
                136.0,
                806.0
            ],
            "score": 0.69,   # 置信度
            "latex": ''      # 公式识别的结果，只有13,14有内容，其他为空，另外15是ocr的结果，这个key会换成text
        },
        ...
    ],
    "page_info": {         # 页信息：提取bbox时的分辨率大小，如果有缩放可以基于该信息进行对齐
        "page_no": 0,      # 页数
        "height": 1684,    # 页高
        "width": 1200      # 页宽
    }
}
```

其中category_id包含的类型如下：

```
{0: 'title',              # 标题
 1: 'plain text',         # 文本
 2: 'abandon',            # 包括页眉页脚页码和页面注释
 3: 'figure',             # 图片
 4: 'figure_caption',     # 图片描述
 5: 'table',              # 表格
 6: 'table_caption',      # 表格描述
 7: 'table_footnote',     # 表格注释
 8: 'isolate_formula',    # 行间公式（这个是layout的行间公式，不太准，实际用14就好）
 9: 'formula_caption',    # 行间公式的标号

 13: 'embedding',         # 行内公式
 14: 'isolated',          # 行间公式
 15: 'text'}              # ocr识别结果
```