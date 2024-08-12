#!/bin/bash

# Function to get the count of pending tasks
get_pending_count() {
    squeue -u zhangtianning.di | grep PD | wc -l
}

# Function to get the count of running tasks
get_running_count() {
    squeue -u zhangtianning.di | grep R | wc -l
}

# Function to get the JOBIDs of pending tasks
get_pending_jobids() {
    squeue -u zhangtianning.di | grep PD | awk '{print $1}'
}

# Function to submit a task
submit_task() {
    sbatch --quotatype=spot -p AI4Chem -N1 -c16 --gres=gpu:1 run.sh sci_index_files.filelist 0 100
}

# Function to cancel extra pending tasks
cancel_extra_pending_tasks() {
    pending_jobids=($(get_pending_jobids))
    for (( i=5; i<${#pending_jobids[@]}; i++ )); do
        echo "Cancelling extra pending task: ${pending_jobids[$i]}"
        scancel "${pending_jobids[$i]}"
    done
}

# Main loop to check and submit tasks every 2 seconds
while true; do
    pending_count=$(get_pending_count)
    running_count=$(get_running_count)

    echo "Pending tasks: $pending_count"
    echo "Running tasks: $running_count"

    # Cancel extra pending tasks if pending count > 5
    if [ "$pending_count" -gt 5 ]; then
        cancel_extra_pending_tasks
        sleep 30
    fi

    pending_count=$(get_pending_count)
    running_count=$(get_running_count)

    echo "Pending tasks: $pending_count"
    echo "Running tasks: $running_count"

    # Submit a task only when running tasks < 60 and pending tasks < 3
    if [ "$running_count" -lt 100 ] && [ "$pending_count" -lt 3 ]; then
        echo "Submitting a new task..."
        submit_task
    fi

    
    # for file in log/*; 
    # do 
    #     ## if head -n 1 file has string `is not` then delete this file
    #     if [ "$(tail -n 3 "$file"|head -n 1|grep -c 'is not')" -eq 1 ]; then
    #         echo "Deleting $file"
    #         rm -f "$file"
    #     fi
    # done
    

    # Sleep for 2 seconds before checking again
    sleep 30
done