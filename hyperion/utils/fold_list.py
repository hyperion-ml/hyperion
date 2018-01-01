"""
Class to make/read/write k-fold x-validation lists
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange
from six import string_types

import os.path as path
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from .list_utils import *

class FoldList(object):

    def __init__(self, fold, key, mask=None):
        self.fold = fold
        self.key = key
        self.mask = mask
        self.validate()


        
    def validate(self):
        self.key = list2ndarray(self.key)
        self.fold = list2ndarray(self.fold)
        if self.fold.dtype != int:
            self.fold = self.fold.astype(int)
        assert len(self.key) == len(self.fold)
        assert len(np.unique(self.fold[self.fold>=0])) == np.max(self.fold)+1
        if self.mask is not None:
            assert len(self.mask) == len(self.fold)

        
    def copy(self):
        return deepcopy(self)


    
    def __len__(self):
        return self.num_folds()


    
    def num_folds(self):
        return np.max(self.fold)+1


    def get_fold_idx(self, fold):
        test_idx = self.fold == fold
        train_idx = np.logical_not(test_idx)
        if self.mask is not None:
            train_idx = np.logical_and(train_idx, self.mask)
            test_idx = np.logical_and(test_idx, self.mask)
        return train_idx, test_idx

    
    
    def get_fold(self, fold):
        train_idx, test_idx = self.get_fold_idx(fold)
        return self.key[train_idx], self.key[test_idx]


    
    def __getitem__(self, fold):
        return self.get_fold(fold)


        
    def save(self, file_path, sep=' '):
        with open(file_path, 'w') as f:
            for f,k in zip(self.fold, self.key):
                f.write('%s%s%s\n' % (f,sep,k))


    @classmethod
    def load(cls, file_path, sep=' '):
        with open(file_path, 'r') as f:
            fields = [line.rstrip().split(sep=sep, maxsplit=1) for line in f]
        fold = np.asarray([int(f[0]) for f in fields], dtype=int)
        key = np.asarray([f[1] for f in fields])
        return cls(fold, key)


    
    @classmethod
    def create(cls, segment_key, num_folds, balance_by_key=None, group_by_key=None, mask=None):

        if group_by_key is None:
            group_by_key = segment_key

        if balance_by_key is None:
            balance_by_key = np.zeros((len(segment_key),), dtype=int)
        else:
            _, balance_by_key = np.unique(balance_by_key, return_inverse=True)

        if mask is not None:
            balance_by_key[mask==False] = -1
            
        folds = - np.ones((len(segment_key),), dtype=int)
            
        num_classes = np.max(balance_by_key) + 1
        for i in xrange(num_classes):
            
            idx_i = (balance_by_key == i).nonzero()[0]
            group_key_i = group_by_key[idx_i]
            _, group_key_i = np.unique(group_key_i, return_inverse=True)
            num_groups_i = np.max(group_key_i) + 1
            delta = float(num_groups_i)/num_folds
            
            for j in xrange(num_folds):
                k1 = int(np.round(j*delta))
                k2 = int(np.round((j+1)*delta))
                idx_ij = np.logical_and(group_key_i>=k1, group_key_i<k2)
                idx_fold = idx_i[idx_ij]
                folds[idx_fold] = j

        if mask is None:
            assert np.all(folds>=0)
        else:
            assert np.all(folds[mask]>=0)
        return cls(folds, segment_key, mask)
        
        
