#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')


def starting_entity(flag, span_info, index):
    named_entity_list, entity_span, start_index, end_index = span_info
    flag = 1
    start_index = index
    span_info = [named_entity_list, entity_span, start_index, end_index]
    return flag, span_info


def continue_next_entity(flag, span_info, index):
    named_entity_list, entity_span, start_index, end_index = span_info
    end_index = index - 1
    entity_span = (start_index, end_index)
    named_entity_list.append(entity_span)
    start_index = index
    span_info = [named_entity_list, entity_span, start_index, end_index]
    return flag, span_info


def ending_entity(flag, span_info, index):
    named_entity_list, entity_span, start_index, end_index = span_info
    flag = 0
    end_index = index - 1
    entity_span = (start_index, end_index)
    named_entity_list.append(entity_span)
    span_info = [named_entity_list, entity_span, start_index, end_index]
    return flag, span_info


def inside_entity(flag, span_info, index):
    named_entity_list, entity_span, start_index, end_index = span_info
    pass
    span_info = [named_entity_list, entity_span, start_index, end_index]
    return flag, span_info
