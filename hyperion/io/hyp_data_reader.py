"""
Class to read data from hdf5 files.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import numpy as np
import h5py

from ..hyp_defs import float_cpu
from ..utils.list_utils import list2ndarray, ismember

class HypDataReader(object):

    def __init__(self, file_path):
        self.file_path = file_path
        self.f = h5py.File(file_path, 'r')


    def read(self, keys, field="", return_tensor=False):
        if isinstance(keys, list):
            datasets = [key+field for key in keys]
        else:
            datasets = keys.astype(np.object)+field
            
        if return_tensor:
            # we assume that all datasets have a common shape
            shape_0 = self.f[datasets[0]].shape
            shape = tuple([len(keys)] + list(shape_0))
            X = np.zeros(shape, dtype=float_cpu())
        else:
            X = []
            
        for i in xrange(len(keys)):
            if datasets[i] in self.f:
                X_i = self.f[datasets[i]]
                if return_tensor:
                    X[i] = X_i
                else:
                    X.append(X_i)
            else:
                print('%s not found' % datasets[i])

        return X
