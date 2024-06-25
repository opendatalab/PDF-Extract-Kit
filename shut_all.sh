locked=("3370948" "3370950")

all_jobs=`squeue --me`
echo "|==> get all jobs."


for name in $all_jobs
do
    if echo "$name" | grep -q "^[0-9]*$"; then
        len=$(expr length "$name")
        if [ "$len" -gt "5" ]; then
            found=false
            for item in "${locked[@]}"; do
                if [ "$item" == "$name" ]; then
                    found=true
                    break
                fi
            done
            if [ "$found" == true ]; then
                echo "job id '$name' locked, skipped."
            else
                scancel $name
                echo "  => shutting job: $name"
            fi
        fi
    fi
done

echo "|==> all job shutted."