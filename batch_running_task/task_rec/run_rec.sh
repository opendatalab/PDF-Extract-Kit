#!/bin/bash
#SBATCH -J ParseSciHUB
#SBATCH -o .log/%j-ParseSciHUB.out  
#SBATCH -e .log/%j-ParseSciHUB.out  




# Check if the version matches

if [[ $(hostname) == SH* ]]; then
    IMAGE_BATCH_SIZE=256
    PDF_BATCH_SIZE=32
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
    IMAGE_BATCH_SIZE=128
    PDF_BATCH_SIZE=16

fi


python batch_running_task/task_rec/batch_deal_with_rec.py --image_batch_size $IMAGE_BATCH_SIZE --pdf_batch_size $PDF_BATCH_SIZE --root_path $1 --index_part $2 --num_parts $3 --num_workers 8 --update_origin --replace --shuffle #--compile 
