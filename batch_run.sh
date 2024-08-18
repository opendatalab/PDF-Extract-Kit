CPU_NUM=16 # Automatically get the number of CPUs
export LD_LIBRARY_PATH=/mnt/cache/share/gcc/gcc-7.5.0/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/mnt/cache/share/gcc/gcc-7.5.0/bin:$PATH
for ((CPU=0; CPU<CPU_NUM; CPU++));
do
sbatch  -p test_s2 -N1 -c8 --gres=gpu:1  run.sh sci_index_files.remain.filelist 0 10
#sbatch --quotatype=spot -p AI4Chem -N1 -c8 --gres=gpu:1  run.sh sci_index_files.remain.filelist 0 2000
done 
