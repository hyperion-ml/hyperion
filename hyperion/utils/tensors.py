from __future__ import absolute_import
from __future__ import print_function

from six.moves import xrange

import numpy as np

def to3D_by_class(x, class_ids, max_length=0):
    dim = x.shape[1]
    n_classes = np.max(class_ids)+1
    if max_length == 0 :
        for i in xrange(n_classes):
            n_i = np.sum(class_ids==i)
            max_length = np.maximum(max_length, n_i)

    x3d = np.zeros((n_classes, max_length, dim), dtype=x.dtype)
    sample_weights = np.zeros((n_classes, max_length), dtype=x.dtype)
    for i in xrange(n_classes):
        idx = class_ids == i
        n_i = np.sum(idx)
        x3d[i,:n_i,:] = x[idx, :]
        sample_weights[i,:n_i] = 1.

    return x3d, sample_weights
