CPU_NUM=20 # Automatically get the number of CPUs
export LD_LIBRARY_PATH=/mnt/cache/share/gcc/gcc-7.5.0/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/mnt/cache/share/gcc/gcc-7.5.0/bin:$PATH
for ((CPU=0; CPU<CPU_NUM; CPU++));
do
sbatch --quotatype=spot -p AI4Chem -N1 -c16 --gres=gpu:1 -x SH-IDC1-10-140-24-41 run.sh sci_index_files.filelist $CPU $CPU_NUM
done 
