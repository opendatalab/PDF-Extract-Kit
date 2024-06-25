#! /bin/bash

function get_yaml_key(){
    cat $1 | while read LINE
    do
        if [ "$(echo $LINE | grep "$2:")" != "" ];then
            if [ "$(echo $LINE | grep -E ' ')" != "" ];then
                if [ "$(echo $LINE | grep -E '#')" == "" ];then
                    echo "$LINE" | awk -F " " '{print $2}'
                    continue
                else
                    continue
                fi
            else
                continue
            fi
        fi
    done
}

function count_used_gpus(){
    all_jobs=`squeue --me -p $1`
    gpu_num=0
    for name in $all_jobs
    do
        if [ "$(echo $name | grep "gpu:")" != "" ];then
            num="${name//gpu:/}"
            gpu_num=$((($gpu_num+$num)))
        fi
    done
    echo $gpu_num
}

yaml_name="global_args.yaml"
echo -e "\033[32m=> parse global config... \033[0m"

input_s3_dir=($(get_yaml_key $yaml_name input_s3_dir))
output_s3_dir=($(get_yaml_key $yaml_name output_s3_dir))
input_list_file=($(get_yaml_key $yaml_name input_list_file))
output_list_file=($(get_yaml_key $yaml_name output_list_file))
partition=($(get_yaml_key $yaml_name partition))
run_num=($(get_yaml_key $yaml_name run_num))
gpu_quota=($(get_yaml_key $yaml_name gpu_quota))

echo -e "\033[32m  => input s3 path: $input_s3_dir \033[0m"
echo -e "\033[32m  => output s3 path: $output_s3_dir \033[0m"
echo -e "\033[32m  => max run num: $run_num \033[0m"


file_list=`tail -n 1000000 $input_list_file`
echo -e "\033[32m=> get all file names. \033[0m"

N=0
for name in $file_list
do
    result=$(echo $name | grep ".jsonl")
    if [ "$result" == "" ]
    then
        continue
    fi

    if grep -Fxq "$name" $output_list_file
    then
        echo -e "\033[32m  => found in run_list: $name  \033[0m"
        continue
    fi
    
    used_gpus=($(count_used_gpus $partition))
    while [ $used_gpus -ge $gpu_quota ]
    do
        echo -e "\033[33m    => gpu over than quota, waitting for former jobs done... \033[0m"
        sleep 300s
        used_gpus=($(count_used_gpus $partition))
    done

    srun -p $partition --gres=gpu:1 --async python process_enroll.py --input $input_s3_dir$name --output $output_s3_dir$name
    echo -e "\033[32m  => run $input_s3_dir$name \033[0m"
    echo $name >> $output_list_file
    sleep 0.5s
    rm batchscript*

    N=$((($N+1)))
    if [ $N -eq $run_num ]
    then
        break
    fi
done

echo -e "\033[32m=> all job submitted. \033[0m"


