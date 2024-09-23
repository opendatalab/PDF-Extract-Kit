
export LD_LIBRARY_PATH=/mnt/cache/share/gcc/gcc-7.5.0/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/mnt/cache/share/gcc/gcc-7.5.0/bin:$PATH

TOTALNUM=30
CPU_NUM=$1 # Automatically get the number of CPUs
if [ -z "$CPU_NUM" ]; then
    CPU_NUM=$TOTALNUM
fi
START=0
for ((CPU=0; CPU<CPU_NUM; CPU++));
do
    
    #sbatch --quotatype=spot -p AI4Chem -N1 -c8 --gres=gpu:1  run.sh sci_index_files.addon.filelist $(($CPU+$START)) $TOTALNUM
    #sbatch --quotatype=spot -p AI4Chem -N1 -c8 --gres=gpu:1  run_mfr.sh physics_collection/sci_index_files.remain.filelist 0 1
    sbatch --quotatype=spot -p AI4Chem -N1 -c8 --gres=gpu:1  batch_running_task/task_rec/run_rec.sh physics_collection/wait_for_ocr.filelist $(($CPU+$START)) $TOTALNUM
    #sbatch --quotatype=spot -p AI4Chem -N1 -c8 --gres=gpu:1  batch_running_task/task_mfr/run_mfr.sh physics_collection/sci_index_files.finished.filelist $(($CPU+$START)) $TOTALNUM
    #sbatch --quotatype=spot -p AI4Chem -N1 -c8 --gres=gpu:1  batch_running_task/task_layout/run_layout_for_missing_page.sh physics_collection/analysis/not_complete_pdf_page_id.pairlist.remain.filelist $(($CPU+$START)) $TOTALNUM
    ## lets sleep 20s every 10 job start
    if [ $(($CPU % 10)) -eq 9 ]; then
        sleep 20
    fi
done 
