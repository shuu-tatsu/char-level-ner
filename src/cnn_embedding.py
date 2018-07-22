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
import load

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
        self.conv1 = nn.Conv2d(1, 6, 3)  # in_channels, out_channels, kernel_size
        self.pool = nn.MaxPool2d(2, 2)  # kernel_size, stride=None
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 50)
        self.fc4 = nn.Linear(50, 3)

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


def train(target_dir,
          embedding_dim,
          hidden_dim,
          glove_file):
    torch.manual_seed(1)
    train_word_to_ix, train_tag_to_ix, train_sents_idx, train_labels_idx = pickle.load(
                                                           open(target_dir + "CoNLL_train.pkl", "rb"))
    test_word_to_ix, test_tag_to_ix, test_sents_idx, test_labels_idx = pickle.load(
                                                           open(target_dir + "CoNLL_test.pkl", "rb"))

    trainloader = DataLoader(train_sents_idx, train_labels_idx, batch_size=1, shuffle=True)
    testloader = DataLoader(test_sents_idx, test_labels_idx, batch_size=1, shuffle=False)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters())

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for inputs, labels in zip(trainloader.sentences, trainloader.labels):
            inputs = torch.Tensor(inputs)
            labels = torch.Tensor(labels)
            #inputs: tensor([  64.,  186.,   39.,  186.,    2.,  186.,   50.,  114.,    0.])
            #labels: tensor([ 1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.])

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
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
