"""
Class to read input and target features from hdf5 files.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import numpy as np
import h5py

from ..utils.list_utils import list2ndarray, ismember

class HypDataWriter(object):

    def __init__(self, file_path):
        self.f = h5py.File(file_path, 'w')
        self.key = None
        
    def create_dataset(self, dataset):
        pass

    def resize_dataset(self, dataset, max_seq_length, nb_seqs):
        pass

    def increase_dataset(self, new_seq_length, nb_seqs):
        pass
        
    def write(self, keys, dataset, data):
        

    def _write_key(self, key):
        if self.key is None:
            self.key = self.f.create_dataset('key', data = key.astype('S'))
            return
        
        
