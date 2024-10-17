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
    export OPENMPIPATH=/mnt/petrelfs/share/openmpi-3.1.2-cuda9.0
    export PATH=$OPENMPIPATH/bin:$PATH
    export LD_LIBRARY_PATH=$OPENMPIPATH/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/mnt/petrelfs/share/test-cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH; ##### <<---- this line will extremly slow the bash
    export LD_LIBRARY_PATH=/mnt/petrelfs/share/test-cuda/cuda-12.1/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

else
    inner_batch_size=8
    batch_size=8

fi
EXTRA_ARGS=$1
python batch_running_task/task_mfr/batch_deal_with_mfr.py --root_path $1 --index_part $2 --num_parts $3 --shuffle --num_workers 8 $EXTRA_ARGS # --replace --update_origin --check_lock False --accelerated_mfr True