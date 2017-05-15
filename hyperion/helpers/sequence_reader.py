#!/usr/bin/env python

"""
Loads data to train UBM, i-vector
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
from ..utils.tensors import to3D_by_seq
from ..transforms import TransformList

class SequenceReader(object):

    def __init__(self, data_file, key_file,
                 feat_norm=None, preproc=None, splicing=None,
                 scp_sep='=', seq_field='',
                 batch_size=1,
                 shuffle_seqs=True, subsample=100,
                 min_seq_length=1, max_seq_length=None,
                 seq_split_mode='random_slice', seq_split_overlap=0,
                 seed=1024):
        self.r = HypDataReader(data_file)
        self.scp = SCPList.load(key_file, sep=scp_sep)
        self.feat_norm = feat_norm
        self.preproc = preproc
        self.splicing = splicing
        self.field = seq_field
        self.batch_size = batch_size
        self.shuffle_seqs = shuffle_seqs
        self.subsample = 100
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_split_mode = seq_split_mode
        self.seq_split_overlap = seq_split_overlap
        self.rng = np.random.RandomState(seed)
        
        self._seq_length = None
        self._num_subseqs = None
        self._num_batches = None
        self.cur_seq = 0
        self.cur_subseq = None
        self.cur_batch = -1
        self.cur_frame = None
        
        if max_seq_length is not None:
            self.cur_subseq = np.zeros((self.scp.len(),), dtype='int64')
            if seq_split_mode == 'sequential':
                self.cur_frame = np.zeros((self.scp.len(),), dtype='int64')


                
    @property
    def num_seqs(self):
        return self.scp.len()


    
    @property
    def seq_length(self):
        if self._seq_length is None:
            self._seq_length = self.r.get_num_rows(self.scp.file_path, self.field)
        return self._seq_length


    
    @property
    def total_length(self):
        return np.sum(self.seq_length)


    
    @property
    def num_subseqs(self):
        if self._num_subseqs is None:
            seq_length = self.seq_length
            if (self.max_seq_length is None or
                self.seq_split_mode=='random_slice_1seq' or
                self.seq_split_mode=='random_samples_1seq'):
                num_subseqs = (seq_length >= self.min_seq_length).astype(int)
            else:
                shift = self.max_seq_length - self.seq_split_overlap
                num_subseqs = (np.floor(seq_length/shift)).astype(int)
                num_subseqs += (seq_length%shift >= self.min_seq_length).astype(int)
            self._num_subseqs = num_subseqs
        return self._num_subseqs


    @property
    def num_total_subseqs(self):
        return np.sum(self.num_subseqs)

    
    @property
    def max_batch_seq_length(self):
        if self.max_seq_length is None:
            return np.max(self.seq_length)
        else:
            return self.max_seq_length


        
    @property
    def num_batches(self):
        if self._num_batches is None:
            num_seqs = np.sum(self.num_subseqs)
            self._num_batches = int(np.floor(num_seqs/self.batch_size))
        return self._num_batches
    

    
    def reset(self):

        self.cur_seq = 0
        self.cur_batch = 0

        if self.max_seq_length is not None:
            self.num_subseqs
            self.cur_subseq[:] = 0
            if self.cur_frame is not None:
                self.cur_frame[:] = 0
                
        if self.shuffle_seqs:
            index = self.scp.shuffle(rng=self.rng)
            if self._seq_length is not None:
                self._seq_length = self._seq_length[index]
            if self._num_subseqs is not None:
                self._num_subseqs = self._num_subseqs[index]


    
    def read(self, return_3d=False,
             max_seq_length=0, return_sample_weights=True):
        
        if self.cur_batch == self.num_batches or self.cur_batch==-1:
            self.reset()
        
        if self.max_seq_length is None:
            x = self._read_full_seqs()
        elif self.seq_split_mode == 'sequential':
            x = self._read_subseqs_sequential()
        elif (self.seq_split_mode == 'random_slice' or
        self.seq_split_mode == 'random_samples'):
            x = self._read_subseqs_random()
        else:
            x = self._read_subseqs_random_1seq()
            
        self.cur_batch +=1

        if return_3d:
            x, sample_weights = to3D_by_seq(x, max_length=max_seq_length)
            return x, sample_weights

        return x


    
    def _read_full_seqs(self):
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
                
        return self.r.read(keys, field=self.field, return_tensor=False)        
            

    
    def _read_subseqs_sequential(self):
        shift = self.max_seq_length - self.seq_split_overlap
        num_subseqs = self.num_subseqs
        seq_length = self.seq_length
        x = []
        for i in xrange(np.sum(num_subseqs)):
            if self.cur_subseq[self.cur_seq] < num_subseqs[self.cur_seq]:
                key = self.scp.file_path[self.cur_seq]
                seq_length_i = min(
                    seq_length[self.cur_seq]-self.cur_frame[self.cur_seq],
                    self.max_seq_length)
                x_i = self.r.read_slice(key, self.cur_frame[self.cur_seq],
                                        seq_length_i, field=self.field)
                x.append(x_i)
                self.cur_frame[self.cur_seq] += shift
                self.cur_subseq[self.cur_seq] += 1

            
            self.cur_seq = (self.cur_seq + 1) % self.num_seqs
            if len(x) == self.batch_size:
                break
        assert(len(x) == self.batch_size)
        return x


    
    def _read_subseqs_random_1seq(self):
        if self.seq_split_mode == 'random_slice_1seq':
            read_f = lambda x, y: self.r.read_random_slice(
                x, y, rng=self.rng, field=self.field)
        else:
            read_f = lambda x, y: self.r.read_random_samples(
                x, y, rng=self.rng, field=self.field)
            
        num_subseqs = self.num_subseqs
        seq_length = self.seq_length
        
        x = []
        for i in xrange(self.cur_seq, self.num_seqs):
            if num_subseqs[i] > 0:
                seq_length_i = min(self.max_seq_length, seq_length[i])
                x_i = read_f(self.scp.file_path[i], seq_length_i)
                x.append(x_i)
            self.cur_seq +=1
            print(len(x))
            
            if len(x) == self.batch_size:
                break
            
        assert(len(x) == self.batch_size)
        return x


    
    def _read_subseqs_random(self):
        if self.seq_split_mode == 'random_slice':
            read_f = lambda x, y: self.r.read_random_slice(
                x, y, rng=self.rng, field=self.field)
        else:
            read_f = lambda x, y: self.r.read_random_samples(
                x, y, rng=self.rng, field=self.field)
            
        num_subseqs = self.num_subseqs
        seq_length = self.seq_length
        
        x = []
        for i in xrange(np.sum(num_subseqs)):
            if self.cur_subseq[self.cur_seq] < num_subseqs[self.cur_seq]:
                if num_subseqs[self.cur_seq] == 1:
                    seq_length_i = min(self.max_seq_length, seq_length[i])
                    x_i = read_f(self.scp.file_path[self.cur_seq], seq_length_i)
                else:
                    x_i = read_f(self.scp.file_path[self.cur_seq], self.max_seq_length)
                x.append(x_i)
                self.cur_subseq[self.cur_seq] += 1

            self.cur_seq = (self.cur_seq + 1) % self.num_seqs
            if len(x) == self.batch_size:
                break
            
        assert(len(x) == self.batch_size)
        return x
    

    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('scp_sep', 'seq_field', 'shuffle_seqs', 'subsample',
                      'min_seq_length', 'max_seq_length',
                      'seq_split_mode', 'seq_split_overlap',
                      'seqr_seed')
        return dict((k, kwargs[p+k])
                    for k in valid if p+k in kwargs)


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
        parser.add_argument(p1+'seq-field', dest=(p2+'seq_field'), default='',
                            help=('dataset field in hdf5 file'))

        parser.add_argument(p1+'shuffle-seqs', dest=(p2+'shuffle_seqs'), default=True,
                            type=bool,
                            help=('shuffles the list of sequences in each epoch'))

        parser.add_argument(p1+'subsample', dest=(p2+'subsample'), default=100,
                            type=float,
                            help=('keeps SUBSAMPLE %% of frames of the seq'))

        parser.add_argument(p1+'min-seq-length', dest=(p2+'min_seq_length'), type=int,
                            default=None,
                            help=('minimum number of frames '
                                  'when we doing sequence splitting'))
        parser.add_argument(p1+'max-seq-length', dest=(p2+'max_seq_length'), type=int,
                            default=None,
                            help=('split one seq into subseqs with '
                                  'seq-length <= max-seq-length'))

        parser.add_argument(p1+'seq-split-mode', dest=(p2+'seq_split_mode'), 
                            default='random_slice', type=str.lower,
                            choices = ['sequential', 'random_slice', 'random_samples',
                                       'random_slice_1seq', 'random_samples_1seq'],
                            help=('seq splitting mode'))
        parser.add_argument(p1+'seq-split-overlap', dest=(p2+'seq_split_overlap'),
                            type=float,
                            default=0, help=('overlap between subsequences'))
        parser.add_argument(p1+'seqr-seed', dest=(p2+'seqr_seed'), type=int,
                            default=1024, help=('seed for rng in sequence reader'))

    

    
