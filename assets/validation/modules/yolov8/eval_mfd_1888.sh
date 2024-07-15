weight=$1
imsize=${2:-1888}

srun -p s2_bigdata --gres=gpu:1 --async \
~/anaconda3/envs/yolov8/bin/python eval_mfd.py \
--weight ${weight} --imsize ${imsize} --cfg1 opendata.yaml

rm batchscript*