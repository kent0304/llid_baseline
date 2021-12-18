# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/finetune/$name
mkdir -p $output
cp $0 $output/run.bash
mkdir -p $output/m2

# See Readme.md for option details.
# --loadLSTM /mnt/LSTA5/data/tanaka/pdc/model_weight/lstm_lm_epoch4.pth \
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./ \
    python train.py \
    --data raw \
    --load /home/tanaka/projects/pdc/baseline/snap/pretrain/pretrain_l-c/BEST \
    --train train \
    --valid valid \
    --visual_embedding attention \
    --bert_embedding mean \
    --hidden_features l-c \
    --concat hadamard \
    --relu relu \
    --batchSize 8 \
    --lr 1e-3 \
    --epochs 50 \
    --tqdm \
    --output $output ${@:3}