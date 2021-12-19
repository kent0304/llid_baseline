## Evaluation
### GLEU
```
python compute_gleu.py -s ../../test_li-c/compositions.txt -r ../../test_li-c/references.txt -o ../../test_li-c/inferences.txt
python compute_gleu.py -s ../../test_l-c/compositions.txt -r ../../test_l-c/references.txt -o ../../test_l-c/inferences.txt
python compute_gleu.py -s ../../gector/compositions.txt -r ../../gector/references.txt -o ../../gector/inferences.txt
```
### M2
- errantのM2算出ツールを利用
- evaluation/errantに移動し
```
source venv/bin/activate
bash val_li-c.sh finetune_li-c 10
```
- 上記コマンドで各epochのloss データでのM2スコアを確認できる．
