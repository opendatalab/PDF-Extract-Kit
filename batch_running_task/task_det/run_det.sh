#!/bin/bash
#SBATCH -J ParseSciHUB
#SBATCH -o .log/%j-ParseSciHUB.out  
#SBATCH -e .log/%j-ParseSciHUB.out  

python batch_running_task/task_det/batch_deal_with_det.py --root_path $1 --index_part $2 --num_parts $3 --shuffle --num_workers 8 # --accelerated_layout --accelerated_mfd 