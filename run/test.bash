# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/test_result/$name
mkdir -p $output
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./ \
    python train.py \
    --load /home/tanaka/projects/pdc/baseline/snap/finetune/finetune_l-c_bertfrz/BEST \
    --data raw \
    --test  test \
    --visual_embedding attention \
    --bert_embedding mean \
    --hidden_features l-c \
    --concat hadamard \
    --output $output ${@:3}

