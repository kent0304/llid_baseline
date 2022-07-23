import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from param import args

DATA_PATH = "./data/"
MSCOCO_IMGFEAT_ROOT = "./data/mscoco_imgfeats/"
SPLIT2NAME = {
    'synthetic_train': 'synthetic/train17_data',
    'synthetic_val': 'synthetic/val17_data',
    'raw_train': 'raw/train',
    'raw_val': 'raw/valid',
    'raw_test': 'raw/test',
    'konan_train': 'konan/konan_train',
    'konan_val': 'konan/konan_val',
    'konan_test': 'konan/konan_test',
}

class MyDataset:
    """
    example:
    "36": {
        "img_id": "482917",
        "correction": "A dog sitting between its masters feet on a footstool watching tv",
        "composition": "A dog sit between its masters feet about a footstool watching tv"
    },
    """
    def __init__(self, split: str):
        self.name = split
        self.data = []
        self.data.extend(json.load(open(DATA_PATH + "%s.json" % SPLIT2NAME[split])).items())
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            k: v
            for k,v in self.data
        }

    def __len__(self):
        return len(self.data)


class MyTorchDataset(Dataset):
    def __init__(self, dataset: MyDataset):
        super().__init__()
        self.raw_dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.lengths = 20

        # Loading detection features to img_data
        img_data = []
        # loading pickle
        print("Loading image features...")
        if args.img_feat == 'bottomup':
            if 'raw' in self.raw_dataset.name: 
                with open(MSCOCO_IMGFEAT_ROOT + 'raw_features_bbox.pkl', "rb") as f:
                        result = pickle.load(f)
            elif 'konan' in self.raw_dataset.name:
                with open(MSCOCO_IMGFEAT_ROOT + 'konan_features_bbox.pkl', "rb") as f:
                    result = pickle.load(f)
            elif 'synthetic_train' in self.raw_dataset.name:
                with open(MSCOCO_IMGFEAT_ROOT + 'train_synthetic_features_mscoco_half_crt.pkl', "rb") as f:
                    result = pickle.load(f)
            elif 'synthetic_val' in self.raw_dataset.name:
                with open(MSCOCO_IMGFEAT_ROOT + 'valid_synthetic_features_mscoco_half_crt.pkl', "rb") as f:
                    result = pickle.load(f)
        elif args.img_feat == 'global':
            if 'raw' in self.raw_dataset.name: 
                with open(MSCOCO_IMGFEAT_ROOT + 'global_raw_features.pkl', "rb") as f:
                    result = pickle.load(f)
            elif 'synthetic_train' in self.raw_dataset.name:
                with open(MSCOCO_IMGFEAT_ROOT + 'global_train_synthetic_features_mscoco_half_crt.pkl', "rb") as f:
                    result = pickle.load(f)
            elif 'synthetic_val' in self.raw_dataset.name:
                with open(MSCOCO_IMGFEAT_ROOT + 'global_valid_synthetic_features_mscoco_half_crt.pkl', "rb") as f:
                    result = pickle.load(f)

        # Convert img list to dict
        self.imgid2img = {} # imagid => img_feature
        for img_datum in result:
            img_id = img_datum['img_id']
            if args.img_feat == 'bottomup':
                img_id = str(int(img_id.split('_')[-1]))
            self.imgid2img[img_id] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            # 物体検出の画像データを管理するimgid2imgに含まれるimg_idのデータのみを保持する
            img_id = datum[1]['img_id']
            img_id = str(int(img_id.split('_')[-1]))
            if img_id in self.imgid2img:
                self.data.append(datum)
            else:
                pass
    
        print("Use %d data in torch dataset" % (len(self.data)))
        # "torch dataset" is for training
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        """
        {
        "img_id": "30055",
        "filename": "COCO_train2014_000000030055_623044.jpg",
        "category_id": 38,
        "ans": "This is a UFO at sky.",
        "crt": "This is a kite in the sky.",
        "best_crt": "This is a kite flying in the sky above a ground with a car parked."
        },
        """
        
        id = self.data[item][0]
        datum = self.data[item][1]
        img_id = datum['img_id']
        img_id = str(int(img_id.split('_')[-1]))

        if 'raw' in self.raw_dataset.name: 
            composition = datum['ans'].lower()
            correction = datum['crt'].lower()
            if args.best_crt:
                correction = datum['best_crt'].lower()
            # gector = datum["gector"].lower()
            filename = datum['filename']
            # if args.gector:
            #     composition = gector
        elif 'synthetic' in self.raw_dataset.name:
            composition = datum['composition']
            correction = datum['correction']
            filename = ''
            
        # Get image info
        img_info = self.imgid2img[img_id]
        if args.img_feat == 'bottomup':
            obj_num = img_info['num_boxes']
            feats = img_info['features'].copy()
            feats = np.squeeze(feats)
            boxes = img_info['boxes'].copy()
            boxes = np.squeeze(boxes)
            assert obj_num == len(boxes) == len(feats)
            feats = torch.tensor(feats, dtype=torch.float32)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            if args.visualization is not None:
                orig_boxes = img_info["original_boxes"].copy()
                orig_boxes = np.squeeze(orig_boxes)
                orig_boxes = torch.tensor(orig_boxes, dtype=torch.float32)
        elif args.img_feat == 'global':
            global_feat = img_info["global_feat"]
            global_feat = torch.tensor(global_feat, dtype=torch.float32)

        # Convert caption (string) to word ids.
        # ans_tokens = nltk.tokenize.word_tokenize(str(ans_text).lower())


        if correction == "<correct>":
            correction_tokens = ['[CLS]'] + self.tokenizer.tokenize(composition) +['[SEP]']
            correction = composition
        else:
            correction_tokens = ['[CLS]'] + self.tokenizer.tokenize(correction) +['[SEP]']
        correction_ids = self.tokenizer.convert_tokens_to_ids(correction_tokens)
        correction_ids = torch.Tensor(correction_ids).long()

        composition_tokens = ['[CLS]'] + self.tokenizer.tokenize(composition) +['[SEP]']
        composition_ids = self.tokenizer.convert_tokens_to_ids(composition_tokens)
        composition_ids = torch.Tensor(composition_ids).long()

        if args.img_feat == 'bottomup':
            if args.visualization is not None:
                return id, filename, feats, boxes, composition, composition_ids, correction, correction_ids, orig_boxes
            return id, filename, feats, boxes, composition, composition_ids, correction, correction_ids
        elif args.img_feat == 'global':
            return id, filename, global_feat, composition, composition_ids, correction, correction_ids
