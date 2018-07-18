#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn


def prepare_sequence(idxs):
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


def flatten(nested_list):
    #2重のリストをフラットにする関数
    return [e for inner_list in nested_list for e in inner_list]
