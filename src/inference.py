#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
import utils
import torch
import get_span


def infer(model, sent_idx):
    with torch.no_grad():
        inputs = utils.prepare_sequence(sent_idx)
        tag_scores = model.forward(inputs)
        _, pred_tag = torch.max(tag_scores.data, 1)
        return pred_tag


def get_entity_span(sent_tag):
    '''
    label2idx = {'O': 0, 'I': 1, 'B': 2}
    '''
    flag = 0
    named_entity_list = []
    entity_span = ()
    start_index = 0
    end_index = 0
    span_info = [named_entity_list, entity_span, start_index, end_index]

    for index, tag in enumerate(sent_tag):
        if tag == 1: # Iになる
            if flag == 0: # O → I  Starting entity
                flag, span_info = get_span.starting_entity(flag, span_info, index)
            elif flag == 1: # I/B → I Inside entity
                flag, span_info = get_span.inside_entity(flag, span_info, index)

        elif tag == 2: # Bになる
            if flag == 1: # I → B  Starting entity
                flag, span_info = get_span.continue_next_entity(flag, span_info, index)

        elif tag == 0: # Oになる
            if flag == 1: # I/B → O  Ending entity
                flag, span_info = get_span.ending_entity(flag, span_info, index)

    named_entity_list = span_info[0]
    return named_entity_list


def count_tp(pred_span, true_span):
    tp = len(set(pred_span) & set(true_span))
    return tp


def count_fp(pred_span, true_span):
    fp = len(set(pred_span) - set(true_span))
    return fp


def count_fn(pred_span, true_span):
    fn = len(set(true_span) - set(pred_span))
    return fn


def precision_recall(pred_tag, true_tag):
    pred_span = get_entity_span(pred_tag)
    true_span = get_entity_span(true_tag)
    tp = count_tp(pred_span, true_span)
    fp = count_fp(pred_span, true_span)
    fn = count_fn(pred_span, true_span)
    return tp, fp, fn


def evaluate(model, sents_idx, labels_idx):
    f1_total = 0
    for sent_idx, label_idx in zip(sents_idx, labels_idx):
        pred_tag = infer(model, sent_idx)
        tp, fp, fn = precision_recall(pred_tag, label_idx)
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 1
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 1
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        f1_total += f1_score
    f1_average = f1_total / len(labels_idx)
    return f1_average
