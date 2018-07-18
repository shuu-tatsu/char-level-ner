#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
import codecs
import collections
import torch
import pickle
import utils
import torch.nn as nn


class Loader():

    def __init__(self, target_dir):
        self.target_dir = target_dir
        self.word2idx = collections.defaultdict(int)
        self.label2idx = {'O': 0, 'I': 1, 'B': 2}

    def load(self,
             data_file,
             make_word_dict):
        with codecs.open(data_file, 'r', 'utf-8') as r:
            lines = r.readlines()
        # Converting format
        data_features, data_labels = read_corpus(lines)
        if make_word_dict:
            self.word2idx = make_dic(self.word2idx, doc_sent=data_features)
        unk_word_id = len(self.word2idx) - 1
        unk_label_id = len(self.label2idx) - 1
        sents_idx = [[self.word2idx.get(word, unk_word_id) for word in sent] \
                                                           for sent in data_features]
        labels_idx = [[self.label2idx.get(label, unk_label_id) for label in labels] \
                                                               for labels in data_labels]
        pickle.dump([self.word2idx, self.label2idx, sents_idx, labels_idx],
                     open(self.target_dir + "CoNLL_" + data_file[19:] + ".pkl", "wb"))


class GloVeLoader():

    def __init__(self,
                 glove_file,
                 target_dir,
                 embedding_dim):
        self.embedding_dim = embedding_dim
        self.glove_file = glove_file
        self.target_dir = target_dir

    def read_train_vocaburary(self):
        train_word_to_ix, _, _, _ = pickle.load(open(self.target_dir + "CoNLL_train.pkl", "rb"))
        return train_word_to_ix

    def read_glove(self):
        with codecs.open(self.glove_file, 'r', 'utf-8') as r:
            words = r.readlines()
        return words

    def get_train_vocaburary_set(self):
        train_word_to_ix = self.read_train_vocaburary()
        train_vocabulary_set = set(train_word_to_ix.keys())
        return train_vocabulary_set

    def get_train_vocaburary_size(self):
        train_word_to_ix = self.read_train_vocaburary()
        vocab_size = len(train_word_to_ix)
        return vocab_size

    def get_train_vocaburary_dict(self):
        train_word_to_ix = self.read_train_vocaburary()
        return train_word_to_ix

    def get_glove_vocaburary_set(self):
        words = self.read_glove()
        glove_vocabulary = []
        for word in words:
            tok = word.split()
            glove_vocabulary.append(tok[0])
        glove_vocabulary_set = set(glove_vocabulary)
        return glove_vocabulary_set

    def extract_train_and_glove(self,
                                train_vocabulary_set,
                                glove_vocabulary_set):
        return train_vocabulary_set & glove_vocabulary_set

    def get_glove_words_vectors_dict(self):
        words = self.read_glove()
        glove_dict = {}
        for word in words:
            tok = word.split()
            glove_dict[tok[0]] = tok[1:]
        return glove_dict

    def get_common_words_vectors_dict(self, common_words, glove_words_vectors_dict):
        common_words_dict = {}
        for word, vector in glove_words_vectors_dict.items():
            if word in common_words:
                common_words_dict[word] = vector
        return common_words_dict

    def get_weight(self):
        # train_vocabulary_set と glove_vocabulary_set を取得
        train_word_to_ix = self.get_train_vocaburary_dict()
        train_vocab_size = self.get_train_vocaburary_size()
        train_vocabulary_set = self.get_train_vocaburary_set()
        glove_vocabulary_set = self.get_glove_vocaburary_set()

        # ランダム初期化済み weight を取得
        self.word_embeddings = nn.Embedding(train_vocab_size, self.embedding_dim)
        self.weight = self.word_embeddings.weight

        # train と glove の積集合を取得
        common_words = self.extract_train_and_glove(train_vocabulary_set,
                                                        glove_vocabulary_set)
        # glove の単語とベクトル辞書を取得
        glove_words_vectors_dict = self.get_glove_words_vectors_dict()

        # glove と train の共通単語を辞書の key とし，
        # その単語の glove におけるベクトルを value とした辞書
        common_words_dict = self.get_common_words_vectors_dict(common_words, glove_words_vectors_dict)

        for word, index in train_word_to_ix.items():
            if word in common_words:
                # train に出てくる word が common_words に含まれていれば，
                # それの (glove 由来の) vector を取得
                word_vector = [float(vec_element) for vec_element in common_words_dict[word]]

                # 取得した vector で weight における，
                # 該当する index 上の vector を上書き
                self.weight[index] = torch.Tensor(word_vector)

        return self.word_embeddings.weight


def read_corpus(lines):
    """
    convert corpus into features and labels
    """
    features = list()
    labels = list()
    tmp_fl = list()
    tmp_ll = list()
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            tmp_ll.append(line[-1][0])
        elif len(tmp_fl) > 0:
            features.append(tmp_fl)
            labels.append(tmp_ll)
            tmp_fl = list()
            tmp_ll = list()
    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels.append(tmp_ll)
    return features, labels


def make_dic(word2idx, doc_sent):
    # 頻度順にソートしてidをふる
    words =  utils.flatten(doc_sent)
    counter = collections.Counter()
    counter.update(words)
    cnt = 0
    for word, count in counter.most_common():
        # 出現回数３回以上の単語のみ辞書に追加
        if count >= 3:
            word2idx[word] = cnt
            cnt += 1
    word2idx[u'<unk>'] = len(word2idx)
    return word2idx


def main():
    torch.manual_seed(1)
    TARGET_DIR = '../corpus/data/'
    GLOVE_FILE = '../corpus/glove.6B/glove.6B.50d.txt'
    #TRAIN_FILE = TARGET_DIR + 'eng.train' # 14041 sentences
    #TEST_FILE = TARGET_DIR + 'eng.test'
    TRAIN_FILE = TARGET_DIR + 'toy.train' # 143 sentences
    TEST_FILE = TARGET_DIR + 'toy.test'
    #TRAIN_FILE = TARGET_DIR + 'mid.train' # 3153 sentences
    #TEST_FILE = TARGET_DIR + 'mid.test'

    EMBEDDING_DIM = 50

    loader = Loader(target_dir=TARGET_DIR)

    #trainの時は単語の辞書を作成する
    loader.load(data_file=TRAIN_FILE,
                make_word_dict=True)
    #testの時は単語の辞書を作成しない
    loader.load(data_file=TEST_FILE,
                make_word_dict=None)

    glove_loader = GloVeLoader(glove_file=GLOVE_FILE,
                               target_dir=TARGET_DIR,
                               embedding_dim=EMBEDDING_DIM)
    globe_word_embeddings_weight = glove_loader.get_weight()


if __name__ == '__main__':
    main()
