
# FILEPATH=sci_index_files.addon.filelist #page_num_map.remain.filelist
# JOBSCRIPT=scan_and_get_page_num_map.py

# FILEPATH=sci_index_files.finished.filelist
# JOBSCRIPT=check_the_detected_row_is_part_of_one_category.py

FILEPATH=script/batch_job_dispatch.sh
JOBSCRIPT=scan_and_judge_ready_for_ocr.py

# JOBSCRIPT=physics_collection/collect_and_upload_to_ceph.py
# FILEPATH=physics_collection/physics_collection.metadata.filelist



PARTION_TOTAL=320 #32 #
PARTION_INTERVAL=16 # 32
PARTION_NUM=$(($PARTION_TOTAL / $PARTION_INTERVAL))
#echo $PARTION_NUM
for ((i=0; i<PARTION_NUM; i++));
do  
    START=$((($i)*$PARTION_INTERVAL))
    sbatch  -p ai4earth -N1 -c20 --gres=gpu:0 job_dispatch.sh $FILEPATH $START $PARTION_INTERVAL $PARTION_TOTAL $JOBSCRIPT
    sleep 1
done

# sbatch  -p ai4earth -N1 -c40 --gres=gpu:0 job_dispatch.sh $FILEPATH   0 32 256 $JOBSCRIPT
# sbatch  -p ai4earth -N1 -c40 --gres=gpu:0 job_dispatch.sh $FILEPATH  32 32 256 $JOBSCRIPT
# sbatch  -p ai4earth -N1 -c40 --gres=gpu:0 job_dispatch.sh $FILEPATH  64 32 256 $JOBSCRIPT
# sbatch  -p ai4earth -N1 -c40 --gres=gpu:0 job_dispatch.sh $FILEPATH  96 32 256 $JOBSCRIPT
# sbatch  -p ai4earth -N1 -c40 --gres=gpu:0 job_dispatch.sh $FILEPATH 128 32 256 $JOBSCRIPT
# sbatch  -p ai4earth -N1 -c40 --gres=gpu:0 job_dispatch.sh $FILEPATH 160 32 256 $JOBSCRIPT
# sbatch  -p ai4earth -N1 -c40 --gres=gpu:0 job_dispatch.sh $FILEPATH 192 32 256 $JOBSCRIPT
# sbatch  -p ai4earth -N1 -c40 --gres=gpu:0 job_dispatch.sh $FILEPATH 224 32 256 $JOBSCRIPT