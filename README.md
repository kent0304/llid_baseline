## About
- Training
    - data.py
    - model.py
    - param.py
    - train.py
    - requirements.txt

- Image Features
    - frcnn: Bottom-up
    - resnet.py: Global

- Attention Visualization
    - visualize.py

## FRCNN
You prepare image features for bottom-up before.
First you move to an independent environment.
```
cd frcnn
pip install -r requirements.txt
python extract_features.py
```


## Installing dependencies
Make sure you setup a virtual environment with Python. Then, install all the dependencies in requirements.txt file
```
pip install -r requirements.txt
```

## Pre-training
```
bash run/pretrain.bash 0[CUDA Number] pretrain_li-c[Project Title]
```

## Finetuneing
```
bash run/finetune.bash 0[CUDA Number] finetune_li-c[Project Title]
```

## Evaluation
## Prepare for Evaluation
```
cd evaluation
python get_result.py [Note for setting project title]
```

### GLEU
```
cd gec-ranking
```

```
python compute_gleu.py -s ../../test_global_li-c/compositions.txt -r ../../test_global_li-c/references.txt -o ../../test_global_li-c/inferences.txt
python compute_gleu.py -s ../../test_li-c/compositions.txt -r ../../test_li-c/references.txt -o ../../test_li-c/inferences.txt
python compute_gleu.py -s ../../no_pretrain_li-c/compositions.txt -r ../../no_pretrain_li-c/references.txt -o ../../no_pretrain_li-c/inferences.txt
python compute_gleu.py -s ../../test_l-c/compositions.txt -r ../../test_l-c/references.txt -o ../../test_l-c/inferences.txt
python compute_gleu.py -s ../../gector/compositions.txt -r ../../gector/references.txt -o ../../gector/inferences.txt
python compute_gleu.py -s ../../bestcrt_finetune_li-c/compositions.txt -r ../../bestcrt_finetune_li-c/references.txt -o ../../bestcrt_finetune_li-c/inferences.txt
```
### M2
- errantのM2算出ツールを利用
- evaluation/errantに移動し
```
source venv/bin/activate
bash val_li-c.sh finetune_li-c 10
```
- 上記コマンドで各epochのloss データでのM2スコアを確認できる．

### Grounding
- l-c: 0.628
- li-c(global): 0.636
- li-c(bottom-up): 0.669
- li-c w/o pretraining(bottom-up): 0.370