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


class DataLoader():

    def __init__(self, sentences, labels, batch_size, shuffle):
        self.sentences = sentences
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv1d(1, 3, 3)
        self.pool = nn.MaxPool1d(2, 2)  # kernel_size, stride=None
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 50)
        self.fc3 = nn.Linear(50, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
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
    print(sentence)
    return sentence


def train(target_dir,
          embedding_dim,
          hidden_dim,
          glove_file):
    torch.manual_seed(1)
    train_char2idx, train_label2idx, train_sents_idx, train_labels_idx = pickle.load(
                                                           open(target_dir + "CoNLL_char_train.pkl", "rb"))
    test_char2idx, test_label2idx, test_sents_idx, test_labels_idx = pickle.load(
                                                           open(target_dir + "CoNLL_char_test.pkl", "rb"))

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters())

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for sentence, labels in zip(train_sents_idx, train_labels_idx):
            for word, label in zip(sentence, labels):

                print(word)
                print(label)
                word_tensor = torch.LongTensor(word)
                label_tensor = torch.LongTensor(label)
                print(word_tensor)
                print(label_tensor)
                print('')

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                output = model.forward(word_tensor)
                loss = criterion(output, label_tensor)
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
