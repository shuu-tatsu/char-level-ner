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
import locale
import inference
import utils
from tqdm import tqdm
from torch.nn.parameter import Parameter
import load_char

import torchvision
import torchvision.transforms as transforms
import numpy as np


class CNN(nn.Module):

    def __init__(self, vocab_size):
        super(CNN, self).__init__()
        self.vocab_size = vocab_size

        #(in_features, out_features, bias=True)
        self.fc = nn.Linear(self.vocab_size, 10)

        #(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv = nn.Conv2d(1, 1, 3, padding=1)


    def forward(self, x):
        x = one_hot(x, self.vocab_size)
        print('one_hot')
        print(x)
        print(x.shape)

        x = torch.Tensor([[x]])
        print('tensor')
        print(x)
        print(x.shape)

        x = self.fc(x)
        print('fc')
        print(x)
        print(x.shape)

        x = self.conv(x)
        print('conv')
        print(x)
        print(x.shape)

        return x


def inference(testloader, classes, model):
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    # imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = model(autograd.Variable(images))
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def transform_ix_to_word(inputs, train_ix_to_word):
    sentence = [train_ix_to_word[ix] for ix in inputs]
    return sentence


def one_hot(x, vocab_size):
    one_hot = np.identity(vocab_size)[x]
    return one_hot


def train(target_dir,
          embedding_dim,
          hidden_dim,
          glove_file):
    torch.manual_seed(1)
    train_char2idx, train_label2idx, train_sents_idx, train_labels_idx = pickle.load(
                                                         open(target_dir + "CoNLL_char_train.pkl", "rb"))
    test_char2idx, test_label2idx, test_sents_idx, test_labels_idx = pickle.load(
                                                         open(target_dir + "CoNLL_char_test.pkl", "rb"))
    model = CNN(len(train_char2idx))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters())

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for sentence, labels in zip(train_sents_idx, train_labels_idx):
            for word, label in zip(sentence, labels):
                print('word:    {}'.format(word))
                print('label:    {}'.format(label))

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                output = model.forward(word)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

    print('Finished Training')
    return model


def main():
    TARGET_DIR = '../corpus/data/'
    GLOVE_FILE = '../corpus/glove.6B/glove.6B.50d.txt'
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 25

    model = train(target_dir=TARGET_DIR,
                  embedding_dim=EMBEDDING_DIM,
                  hidden_dim=HIDDEN_DIM,
                  glove_file=GLOVE_FILE)

    #inference(testloader, classes, model)

if __name__ == '__main__':
    main()
