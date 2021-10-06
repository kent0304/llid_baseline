import argparse
import torch
import torch.nn as nn
import numpy as np
import os
# import pickle
# from data_loader import get_loader 
# from build_vocab import Vocabulary
# from model import EncoderCNN, DecoderRNN
# from torch.nn.utils.rnn import pack_padded_sequence
# from torchvision import transforms

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    main(parser.parse_args())