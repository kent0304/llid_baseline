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

from model import Model, GlobalModel
from data import PDCDataset, PDCTorchDataset

DataTuple = collections.namedtuple("DataTuple", 'dataset loader')



# データをバッチサイズでまとめる際に使う関数
def collate_fn(data):
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[-1]), reverse=True)
    if args.img_feat == 'bottomup':
        id, filename, feats, boxes, composition, composition_ids, correction, correction_ids = zip(*data)
        feats = torch.stack(feats, 0)
        boxes = torch.stack(boxes, 0)
    elif args.img_feat == 'global':
        id, filename, global_feat, composition, composition_ids, correction, correction_ids = zip(*data)
        global_feat = torch.stack(global_feat, 0)
    lengths = [len(cap) for cap in correction_ids]
    targets = torch.zeros(len(correction_ids), 30).long()
    for i, cap in enumerate(correction_ids):
        end = lengths[i]
        if end < 30:
            targets[i, :end] = cap[:end]    
        else:
            targets[i, :30] = cap[:30] 
    if args.img_feat == 'bottomup':
        return id, filename, (feats, boxes), composition, composition_ids, correction, targets, lengths
    elif args.img_feat == 'global':
        return id, filename, global_feat, composition, composition_ids, correction, targets, lengths


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
        if args.img_feat == 'bottomup':
            self.model = Model()
        elif args.img_feat == 'global':
            self.model = GlobalModel()
        
        # GPU options
        self.model = self.model.cuda()

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
        for i, (id, filename, features, composition, composition_ids, correction, targets, lengths) in iter_wrapper(enumerate(loader)):
            if args.img_feat == 'bottomup':
                feats, boxes = features
                feats, boxes, targets = feats.cuda(), boxes.cuda(),  targets.cuda() # crt (8,30)
            elif args.img_feat == 'global':
                global_features = features
                global_features, targets = global_features.cuda(), targets.cuda()
            target = torch.zeros(targets.shape[0], 30).long().cuda()
            target[:,1:targets.shape[1]] = targets[:,1:] # targetもmax_lengthで覆う必要がある (batchsize, max_length)
            inputs = targets[:,:-1]
            with torch.no_grad():
                # if args.attention or args.copyattention:
                #     outputs = self.model(feats, boxes, ans_text, inputs, [l-1 for l in lengths], ans_lengths, ans)
                # else:
                if args.img_feat == 'bottomup':
                    outputs = self.model(feats, boxes, composition, inputs, [l-1 for l in lengths]) 
                elif args.img_feat == 'global':
                    outputs = self.model(global_features, composition, inputs, [l-1 for l in lengths])               
                
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
                for i, (id, filename, features, composition, composition_ids, correction, targets, lengths) in iter_wrapper(enumerate(loader)):
                    self.optim.zero_grad()
        
                    if args.img_feat == 'bottomup':
                        feats, boxes = features
                        feats, boxes, targets = feats.cuda(), boxes.cuda(),  targets.cuda() # crt (8,30)
                    elif args.img_feat == 'global':
                        global_features = features
                        global_features, targets = global_features.cuda(), targets.cuda()
                
                    target = torch.zeros(targets.shape[0], 30).long().cuda()
                    target[:,1:targets.shape[1]] = targets[:,1:] # targetもmax_lengthで覆う必要がある (batchsize, max_length) (8,30)
                    inputs = targets[:,:-1]
                    
                    # if args.attention or args.copyattention:
                    #     outputs = self.model(feats, boxes, composition, inputs, [l-1 for l in lengths], ans_lengths, ans)
                    # else:
                    #     outputs = self.model(feats, boxes, composition, inputs, [l-1 for l in lengths], ans_lengths)   
                    if args.img_feat == 'bottomup':
                        outputs = self.model(feats, boxes, composition, inputs, [l-1 for l in lengths])   
                    elif args.img_feat == 'global':
                        outputs = self.model(global_features, composition, inputs, [l-1 for l in lengths])               
                    loss = self.criterion(outputs.view(-1, 30522), target.reshape(-1))
                    loss.backward()
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
        for i, (id, filename, features, composition, composition_ids, correction, targets, lengths) in enumerate(loader): # Avoid seeing ground truth
            with torch.no_grad():
                if args.img_feat == 'bottomup':
                    feats, boxes = features
                    feats, boxes, targets = feats.cuda(), boxes.cuda(),  targets.cuda() # crt (8,30)
                elif args.img_feat == 'global':
                    global_features = features
                    global_features, targets = global_features.cuda(), targets.cuda()
                # if args.attention or args.copyattention:
                #     sampled_ids = self.model.sample(feats, boxes, ans_text, ans_lengths, ans)
                # else:
                #     sampled_ids = self.model.sample(feats, boxes, ans_text)
                if args.img_feat == 'bottomup':
                    sampled_ids = self.model.sample(feats, boxes, composition)
                elif args.img_feat == 'global':
                    sampled_ids = self.model.sample(global_features, composition)
                sampled_ids = torch.tensor(sampled_ids).cpu().numpy() # (1, max_seq_length) -> (max_seq_length)
            # Convert word_ids to words
            sentence = self.model.tokenizer.decode(sampled_ids)
            sentence_list = sentence.split(' ')
            if "[SEP]" in sentence_list:
                sep_idx = sentence_list.index("[SEP]")
                sentence_list = sentence_list[:sep_idx]
                sentence = " ".join(sentence_list)

            sentence = sentence.replace("[SEP].", "")
            sentence = sentence.replace("[SEP],", "")
            sentence = sentence.replace("[SEP]s", "")
            sentence = sentence.replace(" - ", "-")
            print(sentence)
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
    if args.load is not None:
        correct.load(args.load)

    # Test or Train
    if args.test is not None:
        result = correct.predict(
            get_data_tuple("raw_test", bs=1, shuffle=False, drop_last=False)
        )
        with open(os.path.join(args.output, 'test_predict.json'), 'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    else:
        print('Start training...')
        correct.train(correct.train_tuple, correct.valid_tuple)

    
if __name__ == "__main__":
    main()