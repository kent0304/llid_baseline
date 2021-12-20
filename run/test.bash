# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/test_result/$name
mkdir -p $output
cp $0 $output/run.bash

# See Readme.md for option details.
# l -> finetune/finetune_l-c_lr1-4/epoch6
# li -> finetune/finetune_li-c/epoch16
# global li -> finetune/finetune_global_li-c/epoch48
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./ \
    python train.py \
    --load /home/tanaka/projects/pdc/baseline/snap/finetune/finetune_global_li-c/epoch48 \
    --data raw \
    --test  test \
    --visual_embedding attention \
    --bert_embedding mean \
    --img_feat global \
    --hidden_features li-c \
    --concat hadamard \
    --output $output ${@:3}


