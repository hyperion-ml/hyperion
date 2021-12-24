"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np


def to3D_by_class(x, class_ids, max_length=0):
    dim = x.shape[1]
    num_classes = np.max(class_ids) + 1
    if max_length == 0:
        for i in range(num_classes):
            num_i = np.sum(class_ids == i)
            max_length = np.maximum(max_length, num_i)

    x3d = np.zeros((num_classes, max_length, dim), dtype=x.dtype)
    sample_weight = np.zeros((num_classes, max_length), dtype=x.dtype)
    for i in range(num_classes):
        idx = class_ids == i
        num_i = np.sum(idx)
        x3d[i, :num_i, :] = x[idx, :]
        sample_weight[i, :num_i] = 1.0

    return x3d, sample_weight


def to3D_by_seq(x, max_length=0):
    dim = x[0].shape[1]
    num_seqs = len(x)
    if max_length == 0:
        for i in range(num_seqs):
            num_i = x[i].shape[0]
            max_length = np.maximum(max_length, num_i)

    x3d = np.zeros((num_seqs, max_length, dim), dtype=x[0].dtype)
    sample_weight = np.zeros((num_seqs, max_length), dtype=x[0].dtype)
    for i in range(num_seqs):
        num_i = x[i].shape[0]
        x3d[i, :num_i, :] = x[i]
        sample_weight[i, :num_i] = 1.0

    return x3d, sample_weight
