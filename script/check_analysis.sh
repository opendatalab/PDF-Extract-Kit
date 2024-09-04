
user=`whoami`
jobname='analysis'
runningPID=`squeue -u $user -n $jobname | awk '{print $1}'`

for log_file in .log/convert/*; 
do 
    ### skip if it is not a file
    [ -f "$log_file" ] || continue
    line=$(tail -n 1 "$log_file"| sed 's/\x1B\[A//g'| tr -d '\r')
    echo $log_file $line
    #grep Error "$log_file" 

done
#echo "$output"