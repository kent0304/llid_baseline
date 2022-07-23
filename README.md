# Image Description Dataset for Language Learners
PyTorch code and dataset for our LREC 2022 paper ["Image Description Dataset for Language Learners"](http://www.lsta.media.kyoto-u.ac.jp/mori/research/public/tanaka-LREC22.pdf) by [Kento Tanaka](https://kent34.netlify.app/), Taichi Nishimura, Hiroaki Nanjo, Keisuke Shirai, Hirotaka Kameko, Masatake Dantsuji.

We focus on image description and a corresponding assessment system for language learners. To achieve automatic assessment
of image description, we construct a novel dataset, the Language Learner Image Description (LLID) dataset, which consists of
images, their descriptions, and assessment annotations. Then, we propose a novel task of automatic error correction for image
description, and we develop a baseline model that encodes multimodal information from a learner sentence with an image and
accurately decodes a corrected sentence. Our experimental results show that the developed model can revise errors that cannot
be revised without an image.

Keywords: Image Description, Sentence Error Correction, Language Learning

## Getting started
### FRCNN
You prepare image features for bottom-up before.
First you move to an independent environment.
```
cd frcnn
pip install -r requirements.txt
python extract_features.py
```

### Installing dependencies
Make sure you setup a virtual environment with Python. Then, install all the dependencies in requirements.txt file
```
pip install -r requirements.txt
```

### Pre-training
```
bash run/pretrain.bash 0[CUDA Number] pretrain_li-c[Project Title]
```

### Finetuneing
```
bash run/finetune.bash 0[CUDA Number] finetune_li-c[Project Title]
```

### Evaluation
1. Prepare for Evaluation
```
cd evaluation
python get_result.py [Note for setting project title]
```

2. GLEU
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
 3. ERRANT
- errantのM2算出ツールを利用
- evaluation/errantに移動し
```
source venv/bin/activate
bash val_li-c.sh finetune_li-c 10
```
- 上記コマンドで各epochのloss データでのM2スコアを確認できる．

