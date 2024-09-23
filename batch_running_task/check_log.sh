
for file in .log/*; 
do 
    ### skip if it is not a file
    [ -f "$file" ] || continue
    ## if head -n 1 file has string `is not` then delete this file
    if [ "$(tail -n 3 "$file"|head -n 1|grep -c 'is not')" -eq 1 ]; then
        echo "Deleting $file"
        rm -f "$file"
    fi
done

user=`whoami`
jobname='ParseSciHUB'

runningPID=`squeue -u $user -n $jobname | awk '{print $1}'`
for log_file in .log/*; 
do 
    ### skip if it is not a file
    [ -f "$log_file" ] || continue
    ## get PID from log_file, the name rule is like $PID-ParseSciHUB.out
    PID=$(echo $log_file|awk -F'/' '{print $2}'|awk -F'-' '{print $1}')
    ## if the PID is not in runningPID, then delete this file
    if [ "$(echo $runningPID|grep -c $PID)" -eq 0 ]; then
        #echo "Deleting $log_file"
        rm -f "$log_file"
    else
        #line=$(tail -n 30 "$log_file"|grep Data|tail -n 1| sed 's/\x1B\[A//g'| tr -d '\r')
        line=$(tail -n 1000 "$log_file"|grep "Images batch"|tail -n 1| sed 's/\x1B\[A//g'| tr -d '\r')
        #line=$(tail -n 1000 "$log_file"|grep "[Data]"|tail -n 1| sed 's/\x1B\[A//g'| tr -d '\r')
        echo $log_file $line
        #grep Error "$log_file" 
    fi
done
#echo "$output"