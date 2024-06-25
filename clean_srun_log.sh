all_logs=`ls *.out`
echo "|==> get all logs."

for name in $all_logs
do
    if grep "Finished" $name > /dev/null
    then
        echo "  => job done. $name, removed !"
        rm $name
        continue
    fi

    if grep "srun: error" $name > /dev/null
    then
        echo "  => job killed. $name, removed !"
        rm $name
        continue
    fi

    if grep "slurmstepd: error" $name > /dev/null
    then
        echo "  => job early quited. $name, removed !"
        rm $name
        continue
    fi

    echo "  => job not done. $name"
done