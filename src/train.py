#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import datetime
import locale
import inference
import utils
from tqdm import tqdm
from torch.nn.parameter import Parameter
import load
import cnn_embedding as cnn

'''
# Source code for torch.nn.functional
# グラフを考慮した計算
def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2,
              scale_grad_by_freq=False, sparse=False):
    input = input.contiguous()
    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < weight.size(0), 'Padding_idx must be within num_embeddings'
        elif padding_idx < 0:
            assert padding_idx >= -weight.size(0), 'Padding_idx must be within num_embeddings'
            padding_idx = weight.size(0) + padding_idx
    elif padding_idx is None:
            padding_idx = -1
    if max_norm is not None:
        with torch.no_grad():
            torch.embedding_renorm_(weight, input, max_norm, norm_type)
    #print('input:{}'.format(input.shape))
    #print('weight:{}'.format(weight.shape))
    output = torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
    #print('output:{}'.format(output.shape))
    #print('')
    # グラフの考慮はせず，numpyのような単純な計算
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)


# Source code for torch.nn.modules.sparse
# パラーメータを考慮した計算
class Embedding(Module):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight)
        self.sparse = sparse

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def forward(self, input):
        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, sparse=False):
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            sparse=sparse,
        )
        embedding.weight.requires_grad = not freeze
        return embedding
'''


class LSTMTagger(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 vocab_size,
                 tagset_size,
                 target_dir,
                 glove_file):
        super().__init__()
        self.hidden_dim = hidden_dim
        # one_hot と embeds の内積を取って，センテンス中の各単語をベクトル化し，
        # それらを concat したセンテンスベクトルを取得
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # weight の globe による初期化
        glove_loader = load.GloVeLoader(glove_file, target_dir, embedding_dim)
        self.word_embeddings.weight = Parameter(glove_loader.get_weight())

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        #sentence:torch.Size([72])
        #embeds:torch.Size([72, 50])
        embeds = self.word_embeddings.forward(sentence)
        #cnn_embeds = cnn_embedding(sentence)

        # concat(embeds from globe & embeds from CNN)
        # → input the embeds into LSTM
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

'''
def cnn_embedding(sentence):
    return embeds
'''

def train(target_dir,
          embedding_dim,
          hidden_dim,
          glove_file):
    torch.manual_seed(1)
    train_word_to_ix, train_tag_to_ix, train_sents_idx, train_labels_idx = pickle.load(
                                                           open(target_dir + "CoNLL_train.pkl", "rb"))
    test_word_to_ix, test_tag_to_ix, test_sents_idx, test_labels_idx = pickle.load(
                                                           open(target_dir + "CoNLL_test.pkl", "rb"))
    model = LSTMTagger(embedding_dim,
                       hidden_dim,
                       len(train_word_to_ix),
                       len(train_tag_to_ix),
                       target_dir,
                       glove_file)
    criterion = nn.NLLLoss()
    optimizer = optim.RMSprop(model.parameters())

    EPOCHS = 2
    for epoch in range(EPOCHS):
        loss = 0
        for i, (sentence, tags) in tqdm(enumerate(zip(train_sents_idx, train_labels_idx))):
            model.zero_grad()
            model.hidden = model.init_hidden()
            # 単語インデックスの tensor に変換
            sentence_in = utils.prepare_sequence(sentence)
            # Tags インデックスの tensor に変換
            targets = utils.prepare_sequence(tags)
            tag_scores = model.forward(sentence_in)
            loss = criterion(tag_scores, targets)
            loss.backward()
            optimizer.step()
            loss += loss.item()
        f1_score_train_sents_avg = inference.evaluate(model,
                                                      train_sents_idx[:len(test_sents_idx)],
                                                      train_labels_idx[:len(test_sents_idx)])
        f1_score_test_sents_avg = inference.evaluate(model,
                                                     test_sents_idx,
                                                     test_labels_idx)
        print("[{}] EPOCH {} - LOSS: {:.8f} TRAIN_DATA_F1_SCORE: {} TEST_DATA_F1_SCORE: {}".
                format(datetime.datetime.today(), epoch + 1, loss,
                       f1_score_train_sents_avg, f1_score_test_sents_avg))


def main():
    TARGET_DIR = '../corpus/data/'
    GLOVE_FILE = '../corpus/glove.6B/glove.6B.50d.txt'
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 25

    train(target_dir=TARGET_DIR,
          embedding_dim=EMBEDDING_DIM,
          hidden_dim=HIDDEN_DIM,
          glove_file=GLOVE_FILE)


if __name__ == '__main__':
    main()
