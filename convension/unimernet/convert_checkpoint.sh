#!/bin/bash
#SBATCH -J ParseSciHUB
#SBATCH -o log/%j-convert.out  
#SBATCH -e log/%j-convert.out  
mpirun -n 1 python convert_checkpoint.py --model_type bart \
    --model_dir /mnt/petrelfs/zhangtianning.di/projects/PDF-Extract-Kit/weights/unimernet_clean \
    --output_dir trt_models/unimernet/bfloat16     --tp_size 1     --pp_size 1     --dtype bfloat16     --nougat
## please use sbatch 