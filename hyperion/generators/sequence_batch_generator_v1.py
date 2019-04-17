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
import logging
import copy

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from ..hyp_defs import float_cpu
from ..io import RandomAccessDataReaderFactory as RF
from ..utils.utt2info import Utt2Info
from ..utils.tensors import to3D_by_seq
from ..transforms import TransformList


class SequenceBatchGeneratorV1(object):

    def __init__(self, rspecifier, key_file,
                 class_list = None,
                 path_prefix=None,
                 batch_size=1,
                 iters_per_epoch='auto',
                 gen_method='random', 
                 min_seq_length=None, max_seq_length=None,
                 seq_overlap=0,
                 prune_min_length=0,
                 return_class = True,
                 class_weight = None,
                 max_class_imbalance=2,
                 seq_weight = 'balanced',
                 shuffle_seqs=True,
                 transform=None,
                 init_epoch=0,
                 sg_seed=1024, reset_rng=False,
                 scp_sep=' ', 
                 part_idx=1, num_parts=1):
        
        self.r = RF.create(rspecifier, path_prefix=path_prefix, transform=transform, scp_sep=scp_sep)
        self.u2c = Utt2Info.load(key_file, sep=scp_sep)
        if num_parts > 1:
            self.u2c = self.u2c.split(part_idx, num_parts, group_by_key=False)
        self.batch_size = batch_size

        self.cur_epoch = init_epoch
        self.gen_method = gen_method
        self.shuffle_seqs = shuffle_seqs
        self._min_seq_length = min_seq_length
        self._max_seq_length = max_seq_length
        self.seq_overlap = seq_overlap

        self._init_seq_lengths = None
        self._seq_lengths = None
        
        if prune_min_length > 0:
            self._prune_min_length(prune_min_length)

        self.class_weight = class_weight
        self.seq_weight = seq_weight

        self.return_class = return_class
        self._prepare_class_info(class_list)
        if class_weight == 'balanced':
            self._balance_class_weight(max_class_imbalance)
        
        self.seed = sg_seed
        self.reset_rng = reset_rng
        self.rng = None

        self.init_u2c = self.u2c
        
        self.cur_seq = 0
        self.cur_step = 0
        
        self.cur_frame = None
        self.cur_subseq = None
        self._init_num_subseqs = None
        self.num_subseqs = None

        self._steps_per_epoch = None

        self._prepare_subseqs()
        if iters_per_epoch == 'auto':
            self._compute_iters_auto()
        else:
            self.iters_per_epoch = iters_per_epoch

        self.reset()

                
    @property
    def num_seqs(self):
        return len(self.u2c)


    
    @property
    def seq_lengths(self):
        if self._seq_lengths is None:
            self._init_seq_lengths = self.r.read_num_rows(self.u2c.key)
            self._seq_lengths = self._init_seq_lengths
        return self._seq_lengths


    
    @property
    def total_length(self):
        return np.sum(self.seq_lengths)


    
    @property
    def min_seq_length(self):
        if self._min_seq_length is None:
            self._min_seq_length = np.min(self.seq_lengths)
        return self._min_seq_length


    @property
    def max_seq_length(self):
        if self._max_seq_length is None:
            self._max_seq_length = np.max(self.seq_lengths)
        return self._max_seq_length
    

    @property
    def steps_per_epoch(self):
        if self._steps_per_epoch is None:
            if self.gen_method == 'sequential':
                if self.seq_weight == 'balanced':
                    seqs_per_iter = self.num_seqs*np.max(self.num_subseqs)
                else:
                    seqs_per_iter = np.sum(self.num_subseqs)
            else:
                seqs_per_iter = self.num_seqs
                
            self._steps_per_epoch = int(np.floor(
                self.iters_per_epoch * seqs_per_iter/self.batch_size))
        
        return self._steps_per_epoch


        
    @property
    def num_total_subseqs(self):
        return self.steps_per_epoch * self.batch_size


    
    def _prune_min_length(self, min_length):
        keep_idx = self.seq_lengths >= min_length
        self.u2c = self.u2c.filter_index(keep_idx)
        self._seq_lengths = None


        
    def _prepare_class_info(self, class_list):
        if class_list is None:
            class_dict = {k:i for i, k in enumerate(np.unique(self.u2c.info))}
        else:
            with open(class_list) as f:
                class_dict={line.rstrip().split()[0]: i for i, line in enumerate(f)}

        self.num_classes = len(class_dict)
        self.key2class = {p: class_dict[k] for k, p in zip(self.u2c.info, self.u2c.key)}



    @staticmethod
    def _balance_class_weight_helper(class_ids, max_class_imbalance):
        num_samples = np.bincount(class_ids)
        max_samples = int(np.ceil(np.max(num_samples)/max_class_imbalance))
        idx = []
        for i, num_samples_i in enumerate(num_samples):
            idx_i = (class_ids == i).nonzero()[0]
            r = float(max_samples)/num_samples_i
            if r > 1:
                idx_i = np.tile(idx_i, int(np.ceil(r)))
                idx_i = idx_i[:max_samples]
            idx.append(idx_i)
            
        idx = np.hstack(tuple(idx))
        assert idx.shape[0] >= len(num_samples)*max_samples
        return idx
    

    
    def _balance_class_weight(self, max_class_imbalance):
        classes, class_ids = np.unique(self.u2c.info, return_inverse=True)
        #class_weights = compute_class_weights('balanced', classes, class_ids)
        #num_samples = class_weights/np.min(class_weights)

        idx = self._balance_class_weight_helper(class_ids, max_class_imbalance)
        self.u2c = self.u2c.filter_index(idx)
        # self.u2c.save('tmp.u2c')
        # self.u2c.save('/tmp.u2c')
        if self._init_seq_lengths is not None:
            self._init_seq_legths = self._init_seq_lengths[idx]
            self._seq_lengths = self._init_seq_legths
        

            
    def _compute_iters_auto(self):
        if self.gen_method == 'random':
            avg_total_length = np.mean(self.seq_lengths)
            avg_seq_length = int((self.max_seq_length + self.min_seq_length)/2)
            self.iters_per_epoch = np.ceil(avg_total_length/avg_seq_length)
        else:
            self.iters_per_epoch = 1
        logging.debug('num iters per epoch: %d' % self.iters_per_epoch)


        
    def _prepare_subseqs(self):
        if self.gen_method == 'full_seqs':
            self._prepare_full_seqs()
        elif self.gen_method == 'random':
            self._prepare_random_subseqs()
        elif self.gen_method == 'sequential':
            self._prepare_sequential_subseqs()
            
            

    def _prepare_full_seqs(self):
        pass


    def _prepare_random_subseqs(self):
        pass


    
    def _prepare_sequential_subseqs(self):
        seq_lengths = self.seq_lengths
        avg_length = int((self.max_seq_length + self.min_seq_length)/2)
        shift = avg_length - self.seq_overlap
        self._init_num_subseqs = np.ceil(seq_lengths/shift).astype(int)
        self.num_subseqs = self._init_num_subseqs
        self.cur_frame = np.zeros((self.num_seqs,), dtype=int)
        self.cur_subseq = np.zeros((self.num_seqs,), dtype=int)

    
            
    def reset(self):

        self.cur_seq = 0
        self.cur_step = 0

        if self.reset_rng:
            self.rng = np.random.RandomState(seed=self.seed)
        else:
            logging.debug('\nreset rng %d' % (self.seed+self.cur_epoch))
            self.rng = np.random.RandomState(seed=self.seed+self.cur_epoch)


        if self.shuffle_seqs:
            if self._init_seq_lengths is None:
                self.seq_lengths
                
            self.u2c = self.init_u2c.copy()
            index = self.u2c.shuffle(rng=self.rng)
            self._seq_lengths = self._init_seq_lengths[index]
            if self._init_num_subseqs is not None:
                self.num_subseqs = self._init_num_subseqs[index]

                
        if self.gen_method == 'sequential':
            self.cur_subseq[:] = 0
            self.cur_frame[:] = 0
            


    
    def read(self, squeeze=True, max_seq_length=None):
        
        if self.gen_method == 'full_seqs':
            keys, x = self._read_full_seqs()
        elif self.gen_method == 'random':
            keys, x = self._read_random_subseqs()
        else:
            keys, x = self._read_sequential_subseqs()

        self.cur_step = (self.cur_step + 1) % self.steps_per_epoch
        if self.cur_step == 0:
            self.reset()
            self.cur_epoch += 1

        r = [keys]
            
        if squeeze:
            max_seq_length = (self.max_seq_length if max_seq_length is None
                              else max_seq_length)
            x, sample_weight = to3D_by_seq(x, max_length=max_seq_length)
            r += [x, sample_weight]
        else:
            r.append(x)

        if self.return_class:
            y=np.zeros((len(keys), self.num_classes), dtype=float_cpu())
            for i,k in enumerate(keys):
                y[i, self.key2class[k]] = 1
            r.append(y)
            
        return tuple(r)


    
    def _read_full_seqs(self):
        keys = list(self.u2c.key[self.cur_seq:self.cur_seq+self.batch_size])
        self.cur_seq += self.batch_size
        
        if len(keys) < self.batch_size:
            delta = self.batch_size - len(keys)
            keys += self.u2c.key[:delta]
            self.cur_seq = delta
            assert len(keys) == self.batch_size

        return keys, self.r.read(keys)


    
    def _read_random_subseqs(self):
        
        keys = []
        seq_lengths =[]
        first_frames = []
        for i in xrange(self.batch_size):
            key = self.u2c.key[self.cur_seq]
            full_seq_length = self.seq_lengths[self.cur_seq]
            max_seq_length = min(full_seq_length, self.max_seq_length)
            min_seq_length = min(full_seq_length, self.min_seq_length)
            
            seq_length = self.rng.randint(low=min_seq_length, high=max_seq_length+1)
            first_frame = self.rng.randint(
                low=0, high=full_seq_length-seq_length+1)

            keys.append(key)
            seq_lengths.append(seq_length)
            first_frames.append(first_frame)

            self.cur_seq = (self.cur_seq + 1) % self.num_seqs

        return keys, self.r.read(keys, row_offset=first_frames,
                                 num_rows=seq_lengths)
            

    
    def _read_sequential_subseqs(self):

        keys = []
        seq_lengths =[]
        first_frames = []
        count = 0
        while count < self.batch_size:
            key = self.u2c.key[self.cur_seq]
            first_frame = self.cur_frame[self.cur_seq]
            full_seq_length = self.seq_lengths[self.cur_seq]
            remainder_seq_length =  full_seq_length - first_frame
            if self.cur_subseq[self.cur_seq] == self.num_subseqs[self.cur_seq]:
                self.cur_seq = (self.cur_seq + 1) % self.num_seqs
                continue
            if self.cur_subseq[self.cur_seq] == self.num_subseqs[self.cur_seq]-1:
                seq_length = min(remainder_seq_length, self.max_seq_length)
                self.cur_frame[self.cur_seq] = 0
            else:
                max_seq_length = min(
                    max(self.min_seq_length,
                        remainder_seq_length-self.min_seq_length),
                    self.max_seq_length)
                min_seq_length = min(remainder_seq_length, self.min_seq_length)
                seq_length = self.rng.randint(low=min_seq_length, high=max_seq_length+1)
                self.cur_frame[self.cur_seq] = min(
                    full_seq_length - self.min_seq_length,
                    first_frame + seq_length - self.seq_overlap)
            
            keys.append(key)
            seq_lengths.append(seq_length)
            first_frames.append(first_frame)

            self.cur_subseq[self.cur_seq] += 1
            if self.seq_weight == 'balanced':
                self.cur_subseq[self.cur_seq] %= self.num_subseqs[self.cur_seq]

            self.cur_seq = (self.cur_seq + 1) % self.num_seqs
            count += 1

        assert len(keys) == self.batch_size
        return keys, self.r.read(keys, row_offset=first_frames,
                                 num_rows=seq_lengths)
        
    

    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'

        if p+'no_shuffle_seqs' in kwargs:
            kwargs[p+'shuffle_seqs'] = not kwargs[p+'no_shuffle_seqs']
            
        valid_args = ('scp_sep', 'path_prefix','batch_size',
                      'iters_per_epoch',
                      'gen_method',
                      'class_list', 'shuffle_seqs', 
                      'min_seq_length', 'max_seq_length',
                      'seq_overlap', 'prune_min_length',
                      'class_weight', 'seq_weight',
                      'init_epoch',
                      'sg_seed', 'part_idx', 'num_parts')
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
            
        parser.add_argument(p1+'scp-sep', dest=(p2+'scp_sep'), default=' ',
                            help=('scp file field separator'))

        parser.add_argument(p1+'path-prefix', dest=(p2+'path_prefix'),
                            default='',
                            help=('path prefix for rspecifier scp file'))

        parser.add_argument(p1+'batch-size', dest=(p2+'batch_size'),
                            default=128, type=int,
                            help=('batch size'))
        parser.add_argument(p1+'class-list', dest=(p2+'class_list'), 
                            default=None,
                            help=('ordered list of classes keys'))

        parser.add_argument(
            p1+'iters-per-epoch', dest=(p2+'iters_per_epoch'),
            default='auto',
            type=lambda x: x if x=='auto' else int(x),
            help=('num of passes through the list in each epoch'))

        # parser.add_argument(p1+'no-shuffle-seqs',
        # dest=(p2+'no_shuffle_seqs'), default=False,
        # action='store_true',
        # help=('shuffles the list of sequences in each epoch'))

        parser.add_argument(p1+'gen-method', dest=(p2+'gen_method'), 
                            default='random', type=str.lower,
                            choices = ['full_seqs', 'sequential', 'random'],
                            help=('seq splitting method'))
        parser.add_argument(p1+'min-seq-length', dest=(p2+'min_seq_length'),
                            type=int, default=None,
                            help=('minimum number of frames '
                                  'for sequence splitting'))
        parser.add_argument(p1+'max-seq-length', dest=(p2+'max_seq_length'),
                            type=int, default=None,
                            help=('split one seq into subseqs with '
                                  'seq-length <= max-seq-length'))
        parser.add_argument(p1+'seq-overlap', dest=(p2+'seq_overlap'),
                            type=int, default=0,
                            help=('overlap between subsequences for '
                                  'sequential method'))
        parser.add_argument(
            p1+'prune-min-length', dest=(p2+'prune_min_length'),
            type=int, default=0,
            help=('remove sequences shorter than min_length'))
        
        parser.add_argument(
            p1+'class-weight', dest=(p2+'class_weight'), 
            default=None, type=str.lower,
            choices = ['balanced', 'unbalanced'],
            help=('balances the number of seqs of each class'))
        parser.add_argument(
            p1+'seq-weight', dest=(p2+'seq_weight'), 
            default='balanced', type=str.lower,
            choices = ['balanced', 'unbalanced'],
            help=('balances the weight of each sequence'))
        
        parser.add_argument(p1+'init-epoch', dest=(p2+'init_epoch'),
                            default=0, type=int,
                            help=('batch size'))
        
        parser.add_argument(p1+'sg-seed', dest=(p2+'sg_seed'), type=int,
                            default=1024,
                            help=('seed for rng in sequence reader'))

        parser.add_argument(p1+'part-idx', dest=(p2+'part_idx'),
                            type=int, default=1,
                            help=('splits the list of files in num-parts and process part_idx'))
        parser.add_argument(p1+'num-parts', dest=(p2+'num_parts'),
                            type=int, default=1,
                            help=('splits the list of files in num-parts and process part_idx'))


    
