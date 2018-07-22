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
        self.char2idx = collections.defaultdict(int)
        self.label2idx = {'O': 0, 'I': 1, 'B': 2}

    def load(self,
             data_file,
             make_word_dict):
        with codecs.open(data_file, 'r', 'utf-8') as r:
            lines = r.readlines()
        # Converting format
        data_features, data_labels = read_corpus(lines)
        if make_word_dict:
            self.char2idx = make_dic(self.char2idx, doc_sent=data_features)
        unk_char_id = len(self.char2idx) - 1
        unk_label_id = len(self.label2idx) - 1
        sents_idx = [[[self.char2idx.get(char, unk_char_id) for char in word] \
                                                            for word in sent] \
                                                            for sent in data_features]

        '''
        学習データがtoyの場合のデータサンプル
        '''

        '''
        defaultdict(<class 'int'>, {'e': 0, 'a': 1, 'i': 2, 't': 3, 's': 4, 'n': 5,
        'r': 6, 'o': 7, 'h': 8, 'd': 9, 'l': 10, 'c': 11, 'u': 12, 'm': 13, 'p': 14,
        'g': 15, 'f': 16, 'y': 17, 'w': 18, '.': 19, 'S': 20, 'T': 21, 'b': 22, 'E': 23,
        'I': 24, 'A': 25, 'v': 26, ',': 27, 'N': 28, '1': 29, 'P': 30, 'k': 31, 'R': 32,
        'L': 33, '-': 34, '0': 35, '9': 36, 'O': 37, '2': 38, 'B': 39, 'G': 40, 'C': 41,
        'M': 42, 'D': 43, 'U': 44, 'F': 45, '6': 46, 'K': 47, "'": 48, '"': 49, '5': 50,
        'H': 51, 'q': 52, 'W': 53, 'J': 54, '4': 55, '7': 56, '3': 57, '8': 58, 'x': 59,
        'Y': 60, 'V': 61, 'j': 62, '(': 63, ')': 64, '$': 65, '/': 66, '=': 67, 'z': 68,
        '+': 69, 'X': 70, 'Q': 71, '&': 72, 'Z': 73, ':': 74, '<unk>': 75})
        '''

        #print(data_features)
        '''
        [['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'],

        ['Peter', 'Blackburn'], ...
        '''

        #print(sents_idx)
        '''
        [[[23, 44], [6, 0, 62, 0, 11, 3, 4], [40, 0, 6, 13, 1, 5], [11, 1, 10, 10],
        [3, 7], [22, 7, 17, 11, 7, 3, 3], [39, 6, 2, 3, 2, 4, 8], [10, 1, 13, 22], [19]],

        [[30, 0, 3, 0, 6], [39, 10, 1, 11, 31, 22, 12, 6, 5]], ...
        '''

        labels_idx = [[self.label2idx.get(label, unk_label_id) for label in labels] \
                                                               for labels in data_labels]
        #print(labels_idx)
        '''
        [[1, 0, 1, 0, 0, 0, 1, 0, 0],

        [1, 1], ...
        '''

        pickle.dump([self.char2idx, self.label2idx, sents_idx, labels_idx],
                     open(self.target_dir + "CoNLL_char_" + data_file[19:] + ".pkl", "wb"))


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


def make_dic(char2idx, doc_sent):
    # 頻度順にソートしてidをふる
    words =  utils.flatten(doc_sent)
    chars = utils.flatten(words)
    counter = collections.Counter()
    counter.update(chars)
    cnt = 0
    for char, count in counter.most_common():
        # 出現回数1回以上の文字のみ辞書に追加
        if count >= 1:
            char2idx[char] = cnt
            cnt += 1
    char2idx[u'<unk>'] = len(char2idx)
    print(char2idx)
    return char2idx


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


if __name__ == '__main__':
    main()
