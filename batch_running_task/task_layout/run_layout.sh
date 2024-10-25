#!/bin/bash
#SBATCH -J ParseSciHUB
#SBATCH -o .log/%j-ParseSciHUB.out  
#SBATCH -e .log/%j-ParseSciHUB.out  
if [[ $(hostname) == SH* ]]; then
    inner_batch_size=16
    batch_size=16
    export LD_LIBRARY_PATH=/mnt/cache/share/gcc/gcc-7.5.0/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export PATH=/mnt/cache/share/gcc/gcc-7.5.0/bin:$PATH
    GCC_VERSION=$(gcc -v 2>&1 | grep "gcc version" | awk '{print $3}')
    # Required version
    REQUIRED_VERSION="7.5.0"
    if [ "$GCC_VERSION" != "$REQUIRED_VERSION" ]; then
      echo "[`hostname`] GCC version is not $REQUIRED_VERSION. Exiting."
      exit 1
    else
      echo "[`hostname`] GCC version is $REQUIRED_VERSION."
    fi
else
    inner_batch_size=8
    batch_size=8

fi
EXTRA_ARGS=$4
python batch_running_task/task_layout/batch_deal_with_layout.py --root_path $1 --index_part $2 --num_parts $3 --inner_batch_size $inner_batch_size --batch_size $batch_size --num_workers 8 $EXTRA_ARGS  # --redo   # --result_save_path custom_collection/result --lock_server_path custom_collection # --accelerated_layout --accelerated_mfd 