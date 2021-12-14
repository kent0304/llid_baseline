# coding=utf-8
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertConfig, BertModel
from param import args

BertLayerNorm = torch.nn.LayerNorm

class VisualConfig(object):
    VISUAL_LOSSES = ['obj', 'attr', 'feat']
    def __init__(self):

        self.visual_feat_dim = 2048
        self.visual_pos_dim = 4

        self.obj_id_num = 1600
        self.attr_id_num = 400

        self.visual_losses = self.VISUAL_LOSSES
        self.visual_loss_config = {
            'obj': (self.obj_id_num, 'ce', (-1,), 1/0.15),
            'attr': (self.attr_id_num, 'ce', (-1,), 1/0.15),
            'feat': (2048, 'l2', (-1, 2048), 1/0.15),
        }

    def set_visual_dims(self, feat_dim, pos_dim):
        self.visual_feat_dim = feat_dim
        self.visual_pos_dim = pos_dim


VISUAL_CONFIG = VisualConfig()

class VisualFeatEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = VISUAL_CONFIG.visual_feat_dim # 2048
        pos_dim = VISUAL_CONFIG.visual_pos_dim # 4

        # object feature encoding
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # position encoding
        self.box_fc = nn.Linear(pos_dim, config.hidden_size)
        self.box_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, visual_feat, visual_pos):
        # visual feature
        visual_feat = self.visn_fc(visual_feat)
        visual_feat = self.visn_layer_norm(visual_feat)

        # box position
        visual_pos = self.box_fc(visual_pos)
        visual_pos = self.box_layer_norm(visual_pos)

        output = (visual_feat + visual_pos) / 2
        output = self.dropout(output)
        return output

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    # 文章を受け取り、tokenizeしてidsにする

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        
        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class Encoder(nn.Module):
    def __init__(self, args, max_seq_length, tokenizer):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.config = BertConfig()
        self.embeddings = BertEmbeddings(config=self.config)
        self.visn_fc = VisualFeatEncoder(config=self.config)


    def forward(self, compositions, feat, pos, visual_attention_mask=None):
        train_features = convert_sents_to_features(compositions, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        embedding_output = self.embeddings(input_ids, token_type_ids=segment_ids)
        visn_feats = self.visn_fc(feat, pos)
        # 平均をとる
        visn_feats = visn_feats.mean(dim=1)
        embedding_output = embedding_output.mean(dim=1)
        # concat
        features = torch.cat((embedding_output, visn_feats), dim=1)

        return features


class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, max_seq_length=30):
        """Set the hyper-parameters and build the layers."""
        super(LSTMDecoder, self).__init__()
        self.max_seq_length = max_seq_length # 30
        self.embed_size = embed_size # 768
        self.hidden_size = hidden_size # 768
        self.num_layers = num_layers # 1
        self.vocab_size = vocab_size # 30522
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, corrections, lengths):
        """Decode feature vectors and generates corrections."""
        # print("corrections",corrections.shape)
        # print(corrections) # 256,29
        lengths = [self.max_seq_length-1 if l > self.max_seq_length-1 else l for l in lengths]
        # print("lengths",lengths)
        embeddings = self.embed(corrections) # 256, 29 -> 256, 29, 1536
        # print("featuresのサイズ")
        # print(features.size()) # 256, 1536
        # print("features.unsqueeze(1)のサイズ")
        # print(features.unsqueeze(1).size()) # 256, 1, 1536
        # print("embeddingsのサイズ") # 256, 29, 1536
        # print(embeddings.size())
        bs = embeddings.shape[0]
        # print("embeddings", embeddings.shape) # 256, 29, 1536
        assert embeddings.shape == (corrections.shape[0], corrections.shape[1], self.embed_size)
         
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed, (features.view(self.num_layers, bs, self.hidden_size), features.view(self.num_layers, bs, self.hidden_size)))
        # print("lstmに突っ込んだ後のhiddenがこれ")
        # print(hiddens)
        # print(hiddens[0].size())

        hiddens = pad_packed_sequence(hiddens, batch_first=True)
        # print(hiddens[0].size())
        outputs = self.linear(hiddens[0])
        total_outputs = torch.zeros(bs, self.max_seq_length, self.vocab_size).cuda()
        # print(total_outputs.size())
        # print("outputs", outputs.shape)
        # print(total_outputs[:,1:outputs.shape[1]+1,:].shape)
        total_outputs[:,1:outputs.shape[1]+1,:] = outputs
        # print("outputs",outputs.size())
        return total_outputs
    
    def samples(self, features, states=None):
        """Generate corrections for given features using greedy search."""
        sampled_ids = []
        inputs = self.embed(torch.tensor([101]).cuda())                      # inputs: (batch_size, embed_size)
        inputs = inputs.unsqueeze(1) 
        bs = 1
        states = (features.view(self.lstm.num_layers, bs, self.lstm.hidden_size), features.view(self.lstm.num_layers, bs, self.lstm.hidden_size))
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            # print("outputs",outputs)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            # print("predicted",predicted)
            sampled_ids.append(predicted)
            # print(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        # sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        # print("sampled_ids",sampled_ids)
        return sampled_ids

class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        vocab_size = len(self.tokenizer.vocab)
        self.encoder = Encoder(args, max_seq_length=30, tokenizer=self.tokenizer)
        self.decoder = LSTMDecoder(embed_size=1536, hidden_size=1536, vocab_size=vocab_size)

    def forward(self, feat, pos, composition, correction, lengths):
        features = self.encoder(composition, feat, pos)
        outputs = self.decoder(features, correction, lengths)
        return outputs


    