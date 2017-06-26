"""
Loads data to train LDA, PLDA, PDDA
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import os
import argparse
import time
import copy

import numpy as np

from ..io import HypDataReader
from ..utils.scp_list import SCPList
from ..utils.tensors import to3D_by_class
from ..transforms import TransformList


class VectorClassReader(object):

    def __init__(self, v_file, key_file, preproc=None, scp_sep='=', v_field='',
                 min_spc=1, max_spc=None, spc_pruning_mode='random',
                 csplit_min_spc=1, csplit_max_spc=None, csplit_mode='random',
                 csplit_overlap=0, vcr_seed=1024, csplit_once=True):

        self.r = HypDataReader(v_file)
        self.scp = SCPList.load(key_file, sep=scp_sep)
        self.preproc = preproc
        self.field = v_field
        self.rng = np.random.RandomState(vcr_seed)
        self.csplit_max_spc = csplit_max_spc
        self.csplit_min_spc = csplit_min_spc
        self.csplit_mode = csplit_mode
        self.csplit_overlap = csplit_overlap
        self.csplit_once = csplit_once
        self._samples_per_class = None
        self.scp = self._filter_by_spc(self.scp, min_spc, max_spc, spc_pruning_mode, self.rng)
        if csplit_once:
            self.scp = self._split_classes(self.scp, self.csplit_min_spc, self.csplit_max_spc,
                                           self.csplit_mode, self.csplit_overlap, self.rng)


            
    def read(self, return_3d=False, max_length=0):
        if self.csplit_once:
            scp = self.scp
        else:
            scp = self._split_classes(self.spc, self.csplit_min_spc, self.csplit_max_spc,
                                      self.csplit_mode, self.csplit_overlap, self.rng)
        
        x = self.r.read(scp.file_path, self.field, return_tensor=True)
        if self.preproc is not None:
            x = self.preproc.predict(x)

        _, class_ids=np.unique(scp.key, return_inverse=True)
        if return_3d:
            x, sample_weight = to3D_by_class(x, class_ids, max_length)
            return x, sample_weight
        return x, class_ids


    
    @property
    def samples_per_class(self):
        if self._samples_per_class is None:
            if self.csplit_once:
                scp = self.scp
            else:
                scp = self._split_classes(self.spc, self.csplit_min_spc, self.csplit_max_spc,
                                        self.csplit_mode, self.csplit_overlap, self.rng)
            _, self._samples_per_class=np.unique(scp.key, return_counts=True)

        return self._samples_per_class


    
    @property
    def max_samples_per_class(self):
        num_spc = self.samples_per_class
        return np.max(num_spc)


    
    @staticmethod
    def _filter_by_spc(scp, min_spc=1, max_spc=None, spc_pruning_mode='last', rng=None):
        if min_spc <= 1 and max_spc==None:
            return scp

        if min_spc > 1:
            classes, num_spc = np.unique(scp.key, return_counts=True)
            filter_key = classes[num_spc >= min_spc]
            scp = scp.filter(filter_key)

        if max_spc is not None:
            classes, class_ids, num_spc=np.unique(
                scp.key, return_inverse=True, return_counts=True)
            
            if np.all(num_spc <= max_spc):
                return scp
            f = np.ones_like(class_ids, dtype=bool)
            for i in xrange(np.max(class_ids)+1):
                if num_spc[i] > max_spc:
                    indx = np.where(class_ids == i)[0]
                    num_reject = len(indx) - max_spc
                    if spc_pruning_mode == 'random':
                        #indx = rng.permutation(indx)
                        #indx = indx[-num_reject:]
                        indx = rng.choice(indx, size=num_reject, replace=False)
                    if spc_pruning_mode == 'last':
                        indx = indx[-num_reject:]
                    if spc_pruning_mode == 'first':
                        indx = indx[:num_reject]
                    f[indx] = False

            if np.any(f==False):
                scp = SCPList(scp.key[f], scp.file_path[f])
            
        return scp


    
    @staticmethod
    def _split_classes(scp, min_spc, max_spc, mode='sequential', overlap=0, rng=None):
        if max_spc is None:
            return scp
        if mode == 'random_1part':
            return VectorClassReader._filter_by_scp(scp, min_scp, max_scp, 'random', rng)

        _, class_ids, num_spc = np.unique(scp.key, return_inverse=True, return_counts=True)
        if np.all(num_spc <= max_spc):
            return VectorClassReader._filter_by_spc(scp, min_spc)

        num_classes = np.max(class_ids)+1

        shift = max_spc-overlap
        new_indx = np.zeros(max_spc*int(np.max(num_spc)*num_classes/shift+1), dtype=int)
        new_class_ids = np.zeros_like(new_indx)
        
        j = 0
        new_i = 0
        for i in xrange(num_classes):
            indx_i = np.where(class_ids == i)[0]
            if num_spc[i] > max_spc:
                num_subclass = int(np.ceil((num_spc[i] - max_spc)/shift + 1))
                if mode == 'sequential':
                    l = 0
                    for k in xrange(num_subclass-1):
                        new_indx[j:j+max_spc] = indx_i[l:l+max_spc]
                        new_class_ids[j:j+max_spc] = new_i
                        l += shift
                        j += max_spc
                        new_i += 1
                    n = num_spc[i] - (num_subclass-1)*shift
                    new_indx[j:j+n] = indx_i[l:l+n]
                    new_class_ids[j:j+n] = new_i
                    j += n
                    new_i += 1
                if mode == 'random':
                    for k in xrange(num_subclass):
                        #indx[j:j+max_spc] = rng.permutation(indx_i)[:max_spc]
                        new_indx[j:j+max_spc] = rng.choice(
                            indx_i, size=max_spc, replace=False)
                        new_class_ids[j:j+max_spc] = new_i
                        j += max_spc
                        new_i += 1
            else:
                new_indx[j:j+num_spc[i]] = indx_i
                new_class_ids[j:j+num_spc[i]] = new_i
                new_i += 1
                j += num_spc[i]

        new_indx = new_indx[:j]
        new_class_ids = new_class_ids[:j]
        key = new_class_ids.astype('U')
        file_path = scp.file_path[new_indx]
        scp = SCPList(key, file_path)
        
        return VectorClassReader._filter_by_spc(scp, min_spc)
                     

    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('scp_sep', 'v_field', 
                      'min_spc', 'max_spc', 'spc_pruning_mode',
                      'csplit_min_spc', 'csplit_max_spc',
                      'csplit_mode', 'csplit_overlap',
                      'csplit_once','vcr_seed')
        return dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)

    
    @staticmethod
    def add_argparse_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'
        parser.add_argument(p1+'scp-sep', dest=(p2+'scp_sep'), default='=',
                            help=('scp file field separator'))
        parser.add_argument(p1+'v-field', dest=(p2+'v_field'), default='',
                            help=('dataset field in input vector file'))
        
        parser.add_argument(p1+'min-spc', dest=(p2+'min_spc'), type=int,
                            default=1,
                            help=('minimum samples per class'))
        parser.add_argument(p1+'max-spc', dest=(p2+'max_spc'), type=int,
                            default=None,
                            help=('maximum samples per class'))
        parser.add_argument(p1+'spc-pruning-mode', dest=(p2+'spc_pruning_mode'), 
                            default='random',
                            choices=['random', 'first', 'last'],
                            help=('vector pruning method when spc > max-spc'))
        parser.add_argument(p1+'csplit-min-spc', dest=(p2+'csplit_min_spc'), type=int,
                            default=None,
                            help=('minimum samples per class when doing class spliting'))
        parser.add_argument(p1+'csplit-max-spc', dest=(p2+'csplit_max_spc'), type=int,
                            default=None,
                            help=('split one class into subclasses with '
                                  'spc <= csplit-max-spc'))

        parser.add_argument(p1+'csplit-mode', dest=(p2+'csplit_mode'), 
                            default='random', type=str.lower,
                            choices = ['sequential', 'random', 'random_1subclass'],
                            help=('class splitting mode'))
        parser.add_argument(p1+'csplit-overlap', dest=(p2+'csplit_overlap'), type=float,
                            default=0, help=('overlap between subclasses'))
        parser.add_argument(p1+'csplit-once', dest=(p2+'csplit_once'), 
                            default=True, type=bool,
                            help=('class spliting done only once at the begining'))
        parser.add_argument(p1+'vcr-seed', dest=(p2+'vcr_seed'), type=int,
                            default=1024, help=('seed for rng'))


                            
                    
