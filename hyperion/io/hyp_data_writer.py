"""
Class to write data to hdf5 files.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import numpy as np
import h5py

from ..hyp_defs import float_save
from ..utils.list_utils import list2ndarray, ismember

class HypDataWriter(object):

    def __init__(self, file_path):
        self.file_path = file_path
        self.f = h5py.File(file_path, 'w')


    def write(self, keys, field, X):
        #datasets = keys.astype(np.object)+field
        datasets = [ key+field for key in keys]
        for i, ds in enumerate(datasets):
            self.f.create_dataset(ds, data=X[i,:].astype(float_save()))

