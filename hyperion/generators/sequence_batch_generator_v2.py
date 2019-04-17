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
import logging
import argparse
import time
import copy

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from ..hyp_defs import float_cpu
from ..io import RandomAccessDataReaderFactory as RF
from ..utils.utt2info import Utt2Info
from ..utils.tensors import to3D_by_seq
from ..transforms import TransformList


class SequenceBatchGeneratorV2(object):

    def __init__(self, rspecifier, key_file,
                 class_list = None,
                 path_prefix=None,
                 batch_size=1,
                 iters_per_epoch='auto',
                 num_egs_per_class=1,
                 num_egs_per_utt=1,
                 min_seq_length=None, max_seq_length=None,
                 prune_min_length=0,
                 return_class = True,
                 return_utt1hot = False,
                 class_weight = None,
                 shuffle_seqs=True,
                 transform=None,
                 init_epoch=0,
                 sg_seed=1024, reset_rng=False,
                 scp_sep=' ', 
                 part_idx=1, num_parts=1):

        logging.info('opening reader %s' % rspecifier)
        self.r = RF.create(rspecifier, path_prefix=path_prefix, transform=transform, scp_sep=scp_sep)
        logging.info('loading utt2info file %s' % key_file)
        self.u2c = Utt2Info.load(key_file, sep=scp_sep)
        if num_parts > 1:
            self.u2c = self.u2c.split(part_idx, num_parts)
        self.batch_size = batch_size

        self.cur_epoch = init_epoch

        self.num_egs_per_class = num_egs_per_class
        self.num_egs_per_utt = num_egs_per_utt
        self.shuffle_seqs = shuffle_seqs
        self._min_seq_length = min_seq_length
        self._max_seq_length = max_seq_length

        self._init_seq_lengths = None
        self._seq_lengths = None
        
        if prune_min_length > 0:
            self._prune_min_length(prune_min_length)

        self.class_weight = class_weight

        self.return_class = return_class
        self.return_utt1hot = return_utt1hot
        self._prepare_class_info(class_list)
        
        self.seed = sg_seed
        self.reset_rng = reset_rng
        self.rng = None

        self.cur_step = 0
        self._steps_per_epoch = None

        if iters_per_epoch == 'auto':
            logging.debug('iters_auto')
            sys.stdout.flush()
            self._compute_iters_auto()
            logging.debug('iters_auto')
            sys.stdout.flush()
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
    def seqs_per_iter(self):
        #return self.num_classes*self.num_egs_per_class*self.num_egs_per_utt
        return self.num_seqs
    
    @property
    def steps_per_epoch(self):
        if self._steps_per_epoch is None:
            self._steps_per_epoch = int(np.ceil(
                self.iters_per_epoch * self.seqs_per_iter/self.batch_size))
        
        return self._steps_per_epoch


        
    @property
    def num_total_subseqs(self):
        return self.steps_per_epoch * self.batch_size


    
    def _prune_min_length(self, min_length):
        keep_idx = self.seq_lengths >= min_length
        self.u2c = self.u2c.filter_index(keep_idx)
        self._seq_lengths = self.seq_lengths[keep_idx]
        self._init_seq_lengths = self._seq_lengths


        
    def _prepare_class_info(self, class_list):
        if class_list is None:
            classes, class_idx = np.unique(self.u2c.info, return_inverse=True)
            class_dict = {k:i for i, k in enumerate(classes)}
        else:
            with open(class_list) as f:
                class_dict={line.rstrip().split()[0]: i for i, line in enumerate(f)}
                class_idx = np.array([class_dict[k] for k in self.u2c.info], dtype=int)

        self.num_classes = len(class_dict)

        self.class2utt = {}
        self.class2utt_idx = {}
        self.class2num_utt = np.zeros((self.num_classes,), dtype=int)
        for k in xrange(self.num_classes):
            idx = (class_idx == k).nonzero()[0]
            self.class2utt[k] = [f for f in self.u2c.key[idx]]
            self.class2utt_idx[k] = idx
            self.class2num_utt[k] = len(idx)

        self.num_class_zero_utt = np.sum(self.class2num_utt==0)


            
    # def _compute_iters_auto_0(self):
    #     total_length = np.sum(self.seq_lengths)
    #     avg_seq_length = int((self.max_seq_length + self.min_seq_length)/2)
    #     seqs_per_iter = self.num_classes * self.num_egs_per_class * self.num_egs_per_utt
    #     self.iters_per_epoch = int(np.ceil(total_length/avg_seq_length/seqs_per_iter))
    #     logging.debug('num iters per epoch: %d' % self.iters_per_epoch)

        
    # def _compute_iters_auto_1(self):
    #     seqs_per_iter = self.num_classes * self.num_egs_per_class * self.num_egs_per_utt
    #     self.iters_per_epoch = int(np.ceil(self.num_seqs/seqs_per_iter))
    #     logging.debug('num iters per epoch: %d' % self.iters_per_epoch)


    def _compute_iters_auto(self):
        print(self.seq_lengths)
        avg_total_length = np.mean(self.seq_lengths)
        avg_seq_length = int((self.max_seq_length + self.min_seq_length)/2)
        print(avg_total_length,avg_seq_length)
        self.iters_per_epoch = np.ceil(avg_total_length/avg_seq_length)
        logging.debug('num iters per epoch: %d' % self.iters_per_epoch)
    
            
    def reset(self):

        #self.cur_seq = 0
        self.cur_step = 0

        if self.reset_rng:
            self.rng = np.random.RandomState(seed=self.seed)
        else:
            logging.debug('\nreset rng %d' % (self.seed+self.cur_epoch))
            self.rng = np.random.RandomState(seed=self.seed+self.cur_epoch)


    
    def read(self, squeeze=True, max_seq_length=None):
        
        keys, x, classes, utt_idx = self._read_random_subseqs()
        
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
            y = np.zeros((len(keys), self.num_classes), dtype=float_cpu())
            for i in xrange(len(keys)):
                y[i, classes[i]] = 1
            r.append(y)

        if self.return_utt1hot:
            num_utts = int(np.ceil(self.batch_size/self.num_egs_per_utt))
            _, utt_ids = np.unique(utt_idx, return_inverse=True)
            z = np.zeros((len(keys), self.num_utts), dtype=float_cpu())
            for i,k in enumerate(utt_ids):
                z[i,k] = 1
            r.append(z)
            
        return tuple(r)


    
    
    def _read_random_subseqs(self):
        
        keys = []
        seq_lengths =[]
        first_frames = []
        num_classes_per_batch = int(np.ceil(self.batch_size/self.num_egs_per_class/self.num_egs_per_utt))
        if self.num_class_zero_utt == 0:
            classes = self.rng.randint(low=0, high=self.num_classes, size=(num_classes_per_batch,))
        else:
            classes = np.zeros((num_classes_per_batch,), dtype=int)
            count = 0
            while count < num_classes_per_batch:
                c = self.rng.randint(low=0, high=self.num_classes)
                if self.class2num_utt[c] > 0:
                    classes[count] = c
                    count += 1
                
        classes = np.repeat(classes, self.num_egs_per_class)
        utt_idx= np.array([self.class2utt_idx[c][self.rng.randint(low=0, high=self.class2num_utt[c])] for c in classes], dtype=np.int)
        utt_idx = np.repeat(utt_idx, self.num_egs_per_utt)
        classes = np.repeat(classes, self.num_egs_per_utt)

        for i in xrange(self.batch_size):
            key = self.u2c.key[utt_idx[i]]
            full_seq_length = self.seq_lengths[utt_idx[i]]
            max_seq_length = min(full_seq_length, self.max_seq_length)
            min_seq_length = min(full_seq_length, self.min_seq_length)
            
            seq_length = self.rng.randint(low=min_seq_length, high=max_seq_length+1)
            first_frame = self.rng.randint(
                low=0, high=full_seq_length-seq_length+1)

            keys.append(key)
            seq_lengths.append(seq_length)
            first_frames.append(first_frame)



        return keys, self.r.read(keys, row_offset=first_frames,
                                 num_rows=seq_lengths), classes, utt_idx
            
    

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
                      'num_egs_per_class', 'num_egs_per_utt',
                      'class_list', 'shuffle_seqs', 
                      'min_seq_length', 'max_seq_length',
                      'seq_overlap', 'prune_min_length',
                      'class_weight', 
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

        parser.add_argument(p1+'num-egs-per-class',
                            dest=(p2+'num_egs_per_class'),
                            type=int, default=1,
                            help=('number of samples per class in batch'))
        parser.add_argument(p1+'num-egs-per-utt',
                            dest=(p2+'num_egs_per_utt'),
                            type=int, default=1,
                            help=('number of samples per utterance in batch'))

        
        parser.add_argument(p1+'min-seq-length', dest=(p2+'min_seq_length'),
                            type=int, default=None,
                            help=('minimum number of frames '
                                  'for sequence splitting'))
        parser.add_argument(p1+'max-seq-length', dest=(p2+'max_seq_length'),
                            type=int, default=None,
                            help=('split one seq into subseqs with '
                                  'seq-length <= max-seq-length'))
        parser.add_argument(
            p1+'prune-min-length', dest=(p2+'prune_min_length'),
            type=int, default=0,
            help=('remove sequences shorter than min_length'))
        
        parser.add_argument(
            p1+'class-weight', dest=(p2+'class_weight'), 
            default=None, type=str.lower,
            choices = ['balanced', 'unbalanced'],
            help=('balances the number of seqs of each class'))
        
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


    
