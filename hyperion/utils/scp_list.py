"""
Class to read .scp files
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import os.path as path

import numpy as np
import pandas as pd

from .list_utils import *

class SCPList(object):

    def __init__(self, key, file_path):
        self.key = key
        self.file_path = file_path
        self.validate()

        
    def validate(self):
        self.key = list2ndarray(self.key)
        self.file_path = list2ndarray(self.file_path)
        assert(len(self.key) == len(self.file_path))


    def len(self):
        return len(self.key)

    
    def sort(self):
        self.key, idx =  sort(self.key, return_index=True)
        self.file_path = self.file_path[idx]

        
    def save(self, file_path, sep=' '):
        with open(file_path, 'w') as f:
            for item in zip(self.key, self.file_path):
                f.write('%s%s%s\n' % (item[0], sep, item[1]))
        
    @classmethod
    def load(cls, file_path, sep=' '):
        with open(file_path, 'r') as f:
            fields = [line.rstrip().split(sep=sep, maxsplit=1) for line in f]
        key = [i[0] for i in fields]
        file_path = [i[1] for i in fields]
        return cls(key, file_path)

    
    def split(self, idx, num_parts):
        key, idx1 = split_list(self.key, idx, num_parts)
        file_path = self.file_path[idx1]
        return SCPList(key, file_path)

    
    @classmethod
    def merge(cls, scp_lists):
        key_list = [item.key for item in scp_lists]
        file_list = [item.file_path for item in scp_lists]
        key = np.concatenate(tuple(key_list))
        file_path = np.concatenate(tuple(file_list))
        return cls(key, file_path)

    
    def filter(self, filter_key, keep=True):
        if not(keep):
            filter_key = np.setdiff1d(self.key, filter_key)

        f, _ = ismember(filter_key, self.key)
        assert(np.all(f))
        f, _ = ismember(self.key, filter_key)
        key = self.key[f]
        file_path = self.file_path[f]
        return SCPList(key, file_path)


    def filter_paths(self, filter_key, keep=True):
        if not(keep):
            filter_key = np.setdiff1d(self.file_path, filter_key)

        f, _ = ismember(filter_key, self.file_path)
        assert(np.all(f))
        f, _ = ismember(self.file_path, filter_key)
        key = self.key[f]
        file_path = self.file_path[f]
        return SCPList(key, file_path)


    def shuffle(self, seed=1024, rng=None):
        if rng is None:
            rng = np.random.RandomState(seed=seed)
        index = np.arange(len(self.key))
        rng.shuffle(index)
        self.key = self.key[index]
        self.file_path = self.file_path[index]
        return index
    
        
    def __eq__(self, other):
        if self.key.size == 0 and other.key.size == 0:
            return True
        eq = self.key.shape == other.key.shape
        eq = eq and np.all(self.key == other.key)
        eq = eq and (self.file_path.shape == other.file_path.shape)
        eq = eq and np.all(self.file_path == other.file_path)
        return eq

    
    def __cmp__(self, other):
        if self.__eq__(other):
            return 0
        return 1

    
    def test():

        key = ['spk1']+['spk2']*2+['spk3']*3+['spk10']*10
        file_path = np.arange(len(key)).astype('U')
        file_txt = 'test.txt'
        
        scp1 = SCPList(key, file_path)
        scp1.sort()
        
        scp1.save(file_txt)
        scp2 = SCPList.load(file_txt)
        assert(scp1 == scp2)

        num_parts=3
        scp_list = []
        for i in xrange(num_parts):
            scp_i = scp1.split(i+1, num_parts)
            scp_list.append(scp_i)

        assert(scp_list[0].len() == 1)
        assert(scp_list[1].len() == 10)
        assert(scp_list[2].len() == 5)

        scp2 = SCPList.merge(scp_list)
        assert(scp1 == scp2)

        filter_key = ['spk2', 'spk10']
        scp2 = scp1.filter(filter_key)

        f = np.zeros(len(key), dtype='bool')
        f[1:13] = True
        scp3 = SCPList(scp1.key[f], scp1.file_path[f])
        
        assert(scp2 == scp3)
        
        
