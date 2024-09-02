#!/bin/bash
#SBATCH -J ParseSciHUB
#SBATCH -o .log/%j-ParseSciHUB.out  
#SBATCH -e .log/%j-ParseSciHUB.out  
export LD_LIBRARY_PATH=/mnt/cache/share/gcc/gcc-7.5.0/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/mnt/cache/share/gcc/gcc-7.5.0/bin:$PATH

GCC_VERSION=$(gcc -v 2>&1 | grep "gcc version" | awk '{print $3}')

# Required version
REQUIRED_VERSION="7.5.0"

# Check if the version matches
if [ "$GCC_VERSION" != "$REQUIRED_VERSION" ]; then
  echo "[`hostname`] GCC version is not $REQUIRED_VERSION. Exiting."
  exit 1
else
  echo "[`hostname`] GCC version is $REQUIRED_VERSION."
fi
export OPENMPIPATH=/mnt/petrelfs/share/openmpi-3.1.2-cuda9.0
export PATH=$OPENMPIPATH/bin:$PATH
export LD_LIBRARY_PATH=$OPENMPIPATH/lib:$LD_LIBRARY_PATH
python batch_deal_with_mfr.py --root_path $1 --index_part $2 --num_parts $3 --shuffle --num_workers 8 # --accelerated_layout --accelerated_mfd 