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
from train import get_data_tuple, Correct

from typing import List, Tuple

import matplotlib.patches as patches
from PIL import Image


DataTuple = collections.namedtuple("DataTuple", 'dataset loader')

def add_bboxes_to_image(ax, alpha, image: np.ndarray,
                        bboxes: List[Tuple[int, int, int, int]],
                        line_width: int = 0.4,
                        border_color=(1, 0, 0, 0.4)) -> None:
    """
    Add bbox to ax

    :param image: dtype=np.uint8
    :param bbox: [(left, top, right, bottom)]
    :param label: List[str] or None
    :return: ax
    """
    # Display the image
    ax.imshow(image)

    num = len(bboxes)
    assert num == len(alpha)
    for i in range(num):
        bbox = bboxes[i]
        alpha_i = alpha[i]
        # Add bounding box
        top, left, bottom, right = bbox
        rect = patches.Rectangle((left, top), right - left, bottom - top,
                                 linewidth=line_width,
                                 edgecolor=border_color,
                                 facecolor=(1, 0, 0, 0.5*alpha_i))
        ax.add_patch(rect)

    return ax



def visualize(eval_tuple: DataTuple, model):
        model.eval()
        dset, loader = eval_tuple
        result = {}
        # tokenizer.decoder = decoders.WordPiece()
        for i, (id, filename, features, composition, composition_ids, correction, targets, lengths, orig_boxes) in enumerate(loader): # Avoid seeing ground truth
            with torch.no_grad():
                id = filename[0].split("_")[-2]
                path = '/mnt/LSTA6/data/tanaka/mscoco/train2017/'+id+'.jpg'
           
                image = Image.open(path)
                orig_boxes = orig_boxes.squeeze(0).tolist()
                orig_boxes = [tuple(box) for box in orig_boxes]
       
                # orig_boxes = [(321.75225830078125, 209.3018798828125, 480.0000305175781, 545.8798217773438), (51.01492691040039, 351.13885498046875, 276.2181701660156, 637.3016357421875)]
                
             
                feats, boxes = features
                feats, boxes, targets = feats.cuda(), boxes.cuda(),  targets.cuda() # crt (8,30)
                sampled_ids = model.sample(feats, boxes, composition)
                sampled_ids = torch.tensor(sampled_ids).cpu().numpy()
  

                alpha = model.encoder.get_attention(composition, (feats, boxes))
                alpha = alpha.cpu().numpy().squeeze()
                fig, ax = plt.subplots()
                add_bboxes_to_image(ax, alpha, np.uint8(image), orig_boxes)
                fig.savefig(os.path.join(args.output, 'attention/attention_{}_{}.jpg'.format(id, {str(i+1)})))
          

            # Convert word_ids to words
            sentence = model.tokenizer.decode(sampled_ids)
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





def main():
    correct = Correct()
    if args.load is not None:
        correct.load(args.load)

    if args.test is not None:
        result = visualize(
            get_data_tuple("raw_test", bs=1, shuffle=False, drop_last=False), correct.model
        )


    
if __name__ == "__main__":
    main()