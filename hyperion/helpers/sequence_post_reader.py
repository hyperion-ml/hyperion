"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
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
# from ..utils.scp_list import SCPList
from ..utils.tensors import to3D_by_seq
# from ..transforms import TransformList
from ..hyp_defs import float_cpu
from .sequence_reader import SequenceReader


class SequencePostReader(SequenceReader):
    """Class to read sequences and GMM posteriors (deprecated)
    """
    def __init__(self, data_file, key_file, post_file, num_comp=2048, **kwargs):
        super(SequencePostReader, self).__init__(data_file, key_file, **kwargs)

        self.post_r = HypDataReader(post_file)
        self.num_comp = num_comp
        

    @staticmethod
    def to_dense(r_sparse, index, num_comp):
        index = index.astype(int)
        r = np.zeros((r_sparse.shape[0], num_comp), dtype=float_cpu())
        for i in xrange(r_sparse.shape[0]):
            r[i, index[i]] = r_sparse[i]
        return r


    
    @staticmethod
    def to_dense_list(r_sparse, index, num_comp):
        r = []
        for i in xrange(len(r_sparse)):
            r_i = SequencePostReader.to_dense(r_sparse[i], index[i], num_comp)
            r.append(r_i)
        return r


    
    def read(self, return_3d=False,
             max_seq_length=0, return_sample_weight=True):
        
        if self.cur_batch == self.num_batches or self.cur_batch==-1:
            self.reset()
        
        if self.max_seq_length is None:
            x, r, keys = self.read_full_seqs()
        elif self.seq_split_mode == 'sequential':
            x, r, keys = self._read_subseqs_sequential()
        elif (self.seq_split_mode == 'random_slice' or
        self.seq_split_mode == 'random_samples'):
            x, r, keys = self._read_subseqs_random()
        else:
            x, r, keys = self._read_subseqs_random_1seq()
            
        self.cur_batch +=1

        if return_3d:
            x, sample_weight = to3D_by_seq(x, max_length=max_seq_length)
            r, _ = to3D_by_seq(r, max_length=max_seq_length)
            return x, r, sample_weight, keys

        return x, r, keys


    
    def read_next_seq(self):
        key = self.scp.file_path[self.cur_seq]
        self.cur_seq += 1
        x = self.r.read([key], field=self.field, return_tensor=False)[0]
        r = self.post_r.read([key], field='.r', return_tensor=False)[0]
        index = self.post_r.read([key], field='.index', return_tensor=False)[0]
        r = self.to_dense(r, index, self.num_comp)
        return x, r, key
    

    
    def read_full_seqs(self):
        if self.min_seq_length == 0:
            keys = self.scp.file_path[self.cur_seq:self.cur_seq+self.batch_size]
            self.cur_seq += self.batch_size
        else:
            keys = []
            j = 0
            seq_length = self.seq_length
            for i in xrange(self.cur_seq, self.num_seqs):
                self.cur_seq += 1
                if seq_length[i] > self.min_seq_length:
                    keys.append(self.scp.file_path[i])
                    j +=1
                    if j == self.batch_size:
                        break
            assert(len(keys)==self.batch_size)

        x = self.r.read(keys, field=self.field, return_tensor=False)
        r = self.post_r.read(keys, field='.r', return_tensor=False)
        index = self.post_r.read(keys, field='.index', return_tensor=False)
        r = self.to_dense_list(r, index, self.num_comp)
        
        return x, r, keys 
            

    
    def _read_subseqs_sequential(self):
        shift = self.max_seq_length - self.seq_split_overlap
        num_subseqs = self.num_subseqs
        seq_length = self.seq_length
        x = []
        keys = []
        r = []
        for i in xrange(np.sum(num_subseqs)):
            if self.cur_subseq[self.cur_seq] < num_subseqs[self.cur_seq]:
                key = self.scp.file_path[self.cur_seq]
                seq_length_i = min(
                    seq_length[self.cur_seq]-self.cur_frame[self.cur_seq],
                    self.max_seq_length)
                x_i = self.r.read_slice(key, self.cur_frame[self.cur_seq],
                                        seq_length_i, field=self.field)
                r_i = self.post_r.read_slice(key, self.cur_frame[self.cur_seq],
                                             seq_length_i, field='.r')
                index_i = self.post_r.read_slice(key, self.cur_frame[self.cur_seq],
                                                 seq_length_i, field='.index')
                r_i = self.to_dense(r_i, index_i, self.num_comp)
                x.append(x_i)
                r.append(r_i)
                keys.append(key)
                self.cur_frame[self.cur_seq] += shift
                self.cur_subseq[self.cur_seq] += 1

            
            self.cur_seq = (self.cur_seq + 1) % self.num_seqs
            if len(x) == self.batch_size:
                break
        assert(len(x) == self.batch_size)
        return x, r, keys


    
    def _read_post_slice(self, file_path, index, seq_length):
        r = self.post_r.read_slice(file_path, index, seq_length, field='.r')
        index = self.post_r.read_slice(file_path, index, seq_length, field='.index')
        r = self.to_dense(r, index, self.num_comp)
        return r


            
    def _read_subseqs_random_1seq(self):
        if self.seq_split_mode == 'random_slice_1seq':
            read_f = lambda x, y: self.r.read_random_slice(
                x, y, rng=self.rng, field=self.field)
        else:
            pass
            # read_f = lambda x, y: self.r.read_random_samples(
            #     x, y, rng=self.rng, field=self.field)

        
        num_subseqs = self.num_subseqs
        seq_length = self.seq_length
        
        x = []
        keys = []
        r = []
        for i in xrange(self.cur_seq, self.num_seqs):
            if num_subseqs[i] > 0:
                seq_length_i = min(self.max_seq_length, seq_length[i])
                key = self.scp.file_path[i]
                x_i, index = read_f(key, seq_length_i)
                r_i = self._read_post_slice(key, index, seq_length_i)
                x.append(x_i)
                r.append(r_i)
                keys.append(key)
            self.cur_seq +=1
            
            if len(x) == self.batch_size:
                break
            
        assert(len(x) == self.batch_size)
        return x, r, keys


    
    def _read_subseqs_random(self):
        if self.seq_split_mode == 'random_slice':
            read_f = lambda x, y: self.r.read_random_slice(
                x, y, rng=self.rng, field=self.field)
        else:
            pass
            # read_f = lambda x, y: self.r.read_random_samples(
            #     x, y, rng=self.rng, field=self.field)
            
        num_subseqs = self.num_subseqs
        seq_length = self.seq_length
        
        x = []
        keys = []
        r = []
        for i in xrange(np.sum(num_subseqs)):
            if self.cur_subseq[self.cur_seq] < num_subseqs[self.cur_seq]:
                if num_subseqs[self.cur_seq] == 1:
                    seq_length_i = min(self.max_seq_length, seq_length[self.cur_seq])
                else:
                    seq_length_i = self.max_seq_length

                key = self.scp.file_path[self.cur_seq]
                x_i, index = read_f(key, seq_length_i)
                r_i = self._read_post_slice(key, index, seq_length_i)
                
                x.append(x_i)
                r.append(r_i)
                keys.append(key)
                self.cur_subseq[self.cur_seq] += 1

            self.cur_seq = (self.cur_seq + 1) % self.num_seqs
            if len(x) == self.batch_size:
                break
            
        assert(len(x) == self.batch_size)
        return x, r, keys
    

    
    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('scp_sep', 'seq_field', 'shuffle_seqs', 'subsample',
                      'min_seq_length', 'max_seq_length',
                      'seq_split_mode', 'seq_split_overlap',
                      'seqr_seed', 'part_idx', 'num_parts', 'num_comp')
        return dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)


    
    @staticmethod
    def filter_val_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('scp_sep', 'seq_field', 'subsample',
                      'min_seq_length', 'max_seq_length',
                      'seqr_seed', 'part_idx', 'num_parts', 'num_comp')
        return dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)


    @staticmethod
    def add_argparse_args(parser, prefix=None):
        SequenceReader.add_argparse_args(parser, prefix=prefix)
        
        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'
            

        parser.add_argument(p1+'num-comp', dest=(p2+'num_comp'), default=2048,
                            type=int,
                            help=('number of GMM components'))


    @staticmethod
    def filter_eval_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('scp_sep', 'seq_field', 'part_idx', 'num_parts', 'num_comp')
        return dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)

        
        
    @staticmethod
    def add_argparse_eval_args(parser, prefix=None):
        SequenceReader.add_argparse_eval_args(parser, prefix=prefix)
        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'

            parser.add_argument(p1+'num-comp', dest=(p2+'num_comp'), default=2048,
                                type=int,
                                help=('number of GMM components'))
        

    
