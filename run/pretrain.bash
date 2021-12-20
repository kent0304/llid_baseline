# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/pretrain/$name
mkdir -p $output
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./ \
    python train.py \
    --data synthetic \
    --train train \
    --valid valid \
    --visual_embedding attention \
    --bert_embedding mean \
    --hidden_features li-c \
    --concat hadamard \
    --relu relu \
    --batchSize 64 \
    --lr 5e-5 \
    --epochs 30 \
    --tqdm \
    --output $output ${@:3}


