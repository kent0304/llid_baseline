# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/visualize/$name
mkdir -p $output
mkdir -p $output/attention
cp $0 $output/run.bash

# See Readme.md for option details.
# --load /home/tanaka/projects/pdc/baseline/snap/finetune/no_pretrain_li-c_lr1-4/epoch39 \
# --load /home/tanaka/projects/pdc/baseline/snap/finetune/finetune_li-c/epoch16 \
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./ \
    python visualize.py \
    --load /home/tanaka/projects/pdc/baseline/snap/finetune/finetune_li-c/epoch16 \
    --data raw \
    --test  test \
    --visual_embedding attention \
    --bert_embedding mean \
    --hidden_features li-c \
    --img_feat bottomup \
    --concat hadamard \
    --visualization visualize \
    --output $output ${@:3}
