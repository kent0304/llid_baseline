import json
import subprocess
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import collections
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from param import args

from matplotlib import pyplot as plt
plt.switch_backend('agg')

from model import Model
from data import PDCDataset, PDCTorchDataset

DataTuple = collections.namedtuple("DataTuple", 'dataset loader')



# データをバッチサイズでまとめる際に使う関数
def collate_fn(data):
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[-1]), reverse=True)
    id, filename, feats, boxes, composition, composition_ids, correction, correction_ids = zip(*data)
    # id, feats, boxes, ans_text, crt, file_name, ans_text_copy, ans = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    feats = torch.stack(feats, 0)
    boxes = torch.stack(boxes, 0)

    lengths = [len(cap) for cap in correction_ids]
    targets = torch.zeros(len(correction_ids), 30).long()
    # ans_lengths = [len(a) for a in composition_ids]
    # ans_targets = torch.zeros(len(composition_ids), 30).long()
    for i, cap in enumerate(correction_ids):
        end = lengths[i]
        if end < 30:
            targets[i, :end] = cap[:end]    
        else:
            targets[i, :30] = cap[:30]   
    # for i, a in enumerate(composition_ids):
    #     end = ans_lengths[i]
    #     if end < 30:
    #         ans_targets[i, :end] = a[:end]    
    #     else:
    #         ans_targets[i, :30] = a[:30]
    return id, filename, feats, boxes, composition, composition_ids, correction, targets, lengths

# drop_lastは最後のミニバッチが余った時にデータを捨てるかどうか
# pin_memoryはGPUメモリにデータを送るかどうか
def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = PDCDataset(splits)
    tset = PDCTorchDataset(dset)
    # (36, 2048)特徴量, (36, 4)位置情報
    if splits == "raw_val":
        bs = len(tset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True, collate_fn=collate_fn
    )

    return DataTuple(dataset=dset, loader=data_loader)


class Correct:
    def __init__(self):
        # Datasets
        if args.data == 'raw':
            self.train_tuple = get_data_tuple(
                "raw_train", bs=args.batch_size, shuffle=True, drop_last=True
            )
            self.valid_tuple = get_data_tuple(
                "raw_val", bs=args.batch_size, shuffle=False, drop_last=False
            ) 
        elif args.data == 'synthetic':
            self.train_tuple = get_data_tuple(
                "synthetic_train", bs=args.batch_size, shuffle=True, drop_last=True
            )
            self.valid_tuple = get_data_tuple(
                "synthetic_val", bs=args.batch_size, shuffle=False, drop_last=False
            )
        self.model = Model()
        
        # # Model
        # if args.attention:
        #     print("CorrectModelAttentionロード")
        #     self.model = CorrectModelAttention()
        # elif args.copyattention:
        #     print("CorrectModelAttentionCopyロード")
        #     self.model = CorrectModelAttentionCopy()
        # else:
        #     self.model = CorrectModel()
        

        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optim = torch.optim.Adam(self.model.parameters(), args.lr)

        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)
    
    def evaluate(self, eval_tuple: DataTuple, epoch: int):
        dset, loader = eval_tuple
        self.model.eval()
        valid_loss = 0
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        for i, (id, filename, feats, boxes, composition, composition_ids, correction, targets, lengths) in iter_wrapper(enumerate(loader)):
            feats, boxes,  targets = feats.cuda(), boxes.cuda(),  targets.cuda() # crt (8,30)
            target = torch.zeros(targets.shape[0], 30).long().cuda()
            target[:,1:targets.shape[1]] = targets[:,1:] # targetもmax_lengthで覆う必要がある (batchsize, max_length)

            inputs = targets[:,:-1]
            with torch.no_grad():
                # if args.attention or args.copyattention:
                #     outputs = self.model(feats, boxes, ans_text, inputs, [l-1 for l in lengths], ans_lengths, ans)
                # else:
                outputs = self.model(feats, boxes, composition, inputs, [l-1 for l in lengths]) 
                # m2 score 
                if args.data == 'raw':
                    sampled_ids = outputs.max(2)[1] # maxの出力はidx0: 値, idx1: 位置
                    sentences = [self.model.tokenizer.decode(sampled_id, skip_special_tokens=True) for sampled_id in sampled_ids]

                    with open(os.path.join(self.output, "m2/epoch{}_inference.txt".format(str(epoch))), "w") as f:
                        for sentence in sentences:
                            f.write(sentence + "\n")
                    if not os.path.isfile(os.path.join(self.output, "m2/compositions.txt")):
                        with open(os.path.join(self.output, "m2/compositions.txt"), "w")as f:
                            for cp in composition:
                                f.write(cp + "\n")
                    if not os.path.isfile(os.path.join(self.output, "m2/references.txt")):
                        with open(os.path.join(self.output, "m2/references.txt"), "w")as f:
                            for cr in correction:
                                f.write(cr + "\n")

            loss = self.criterion(outputs.view(-1, 30522), target.reshape(-1))
            valid_loss += loss.item()
        return valid_loss / len(loader)

    def train(self, train_tuple, eval_tuple):
        dset, loader = train_tuple
        train_losses = []
        valid_losses = []
        best_valid = 100
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        with open(os.path.join(self.output, "loss.txt"), "w") as losstxt:
            for epoch in range(args.epochs):
                train_loss = 0
                self.model.train()
                for i, (id, filename, feats, boxes, composition, composition_ids, correction, targets, lengths) in iter_wrapper(enumerate(loader)):
                    self.optim.zero_grad()
                    feats, boxes, targets = feats.cuda(), boxes.cuda(),  targets.cuda() # crt (8,30)
                    target = torch.zeros(targets.shape[0], 30).long().cuda()

                    target[:,1:targets.shape[1]] = targets[:,1:] # targetもmax_lengthで覆う必要がある (batchsize, max_length) (8,30)

                    inputs = targets[:,:-1]
                    
                    # if args.attention or args.copyattention:
                    #     outputs = self.model(feats, boxes, composition, inputs, [l-1 for l in lengths], ans_lengths, ans)
                    # else:
                    #     outputs = self.model(feats, boxes, composition, inputs, [l-1 for l in lengths], ans_lengths)   
                    outputs = self.model(feats, boxes, composition, inputs, [l-1 for l in lengths]) 
    
                    # print(outputs.shape)
                    
                    loss = self.criterion(outputs.view(-1, 30522), target.reshape(-1))
                    loss.backward()
                    # print(loss.item())
                    self.optim.step()
                    train_loss += loss.item()
                
                train_losses.append(train_loss / len(loader))
                valid_loss = self.evaluate(eval_tuple, epoch)
                valid_losses.append(valid_loss)
                
                print(f"Epoch: {epoch} Train Loss: {train_loss / len(loader):.4f} Valid Loss: {valid_loss:.4f}")
                losstxt.write(f"Epoch: {epoch} Train Loss: {train_loss / len(loader):.4f} Valid Loss: {valid_loss:.4f}\n")
                self.my_plot(train_losses, valid_losses)
                # validation データで高い評価値のものを保存
                if self.valid_tuple is not None:  # Do Validation
                    if args.data == 'raw':
                        self.save("epoch{}".format(epoch))
                        
                    else:
                        if valid_loss < best_valid:
                            best_valid = valid_loss
                            self.save("BEST")


    def predict(self, eval_tuple: DataTuple):
        self.model.eval()
        dset, loader = eval_tuple
        result = {}
        # tokenizer.decoder = decoders.WordPiece()
        for i, (id, filename, feats, boxes, composition, composition_ids, correction, targets, lengths) in enumerate(loader): # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                # if args.attention or args.copyattention:
                #     sampled_ids = self.model.sample(feats, boxes, ans_text, ans_lengths, ans)
                # else:
                #     sampled_ids = self.model.sample(feats, boxes, ans_text)
                sampled_ids = self.model.sample(feats, boxes, composition)
                sampled_ids = torch.tensor(sampled_ids).cpu().numpy() # (1, max_seq_length) -> (max_seq_length)
            # Convert word_ids to words
            sentence = self.model.tokenizer.decode(sampled_ids)
            sentence = sentence.replace(".[SEP]", ". [SEP]") # [SEP]と.が連接しているものを離しておく
            sentence = sentence.replace("[SEP].", "[SEP] .")
            sentence = sentence.replace(" - ", "-")
            sentence_list = sentence.split(" ")
            # sentence_list = tokenizer.tokenize(sentence)
            if "[SEP]" in sentence_list:
                sep_idx = sentence_list.index("[SEP]")
                sentence_list = sentence_list[:sep_idx]
                sentence = " ".join(sentence_list)
                sentence = sentence.replace(" .", ".")
                sentence = sentence.replace(" ,", ",")
                print(sentence)
 
            crt = self.model.tokenizer.decode(targets.squeeze(0))
            crt = crt.replace("[CLS]", "")
            crt = crt.replace("[SEP]", "")
            crt = crt.replace("[PAD]", "")
            crt = crt.replace(" - ", "-")
            crt = crt.strip()
            result[id[0]] = {"filename": filename[0], "学習者作文": composition[0], "専門家による添削": correction[0], "モデルによる添削予測": sentence}
        return result
    



    def save(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)
    
    def my_plot(self, train_losses, valid_losses):
        # グラフの描画先の準備
        fig = plt.figure()
        # 画像描画
        plt.plot(train_losses, label='train')
        plt.plot(valid_losses, label='valid')
        #グラフタイトル
        plt.title('Cross Entropy Loss')
        #グラフの軸
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #グラフの凡例
        plt.legend()
        # グラフ画像保存
        fig.savefig(os.path.join(args.output, 'loss.jpg'))

def main():
    correct = Correct()

    # 学習済写真描写訂正モデルの読み込み
    if args.load is not None:
        correct.load(args.load)


    # Test or Train
    if args.test is not None:
        result = correct.predict(
            get_data_tuple("raw_test", bs=1, shuffle=False, drop_last=False)
        )
        # print("========")
        # print(result)
        with open(os.path.join(args.output, 'test_predict.json'), 'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

    else:
        print('Start training...')
        correct.train(correct.train_tuple, correct.valid_tuple)

    
    

if __name__ == "__main__":
    main()