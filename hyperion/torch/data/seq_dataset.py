"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import os
import logging
import argparse
import time
import copy

import numpy as np
import pandas as pd

from ..hyp_defs import float_cpu
from ..io import RandomAccessDataReaderFactory as RF
from ..utils.utt2info import Utt2Info

from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, rspecifier, key_file,
                 class_file = None,
                 path_prefix=None,
                 min_chunk_length=0,
                 max_chunk_length=None,
                 return_fullseqs=False,
                 return_class=True):

        logging.info('opening dataset %s' % rspecifier)
        self.r = RF.create(
            rspecifier, path_prefix=path_prefix, scp_sep=' ')
        logging.info('loading utt2info file %s' % key_file)
        self.u2c = Utt2Info.load(key_file, sep=' ')
        logging.info('dataset contains %d seqs' % self.num_seqs)

        self._prepare_class_info(class_file)
        
        self._min_chunk_length = min_chunk_length
        self._max_chunk_length = max_chunk_length

        self.return_fullseqs = return_fullseqs
        self.return_class = return_class
        self._seq_lengths = None

        if min_chunk_length > 0:
            self._prune_short_seqs(min_chunk_length)

        if max_chunk_length is not None:
            self._prune_short_seqs(max_chunk_length)
            
        self.cur_batch_params = None


    @property
    def num_seqs(self):
        return len(self.u2c)


    def __len__(self):
        return self.num_seqs


    @property
    def seq_lengths(self):
        if self._seq_lengths is None:
            self._seq_lengths = self.r.read_num_rows(self.u2c.key)

        return self._seq_lengths

    @property
    def total_length(self):
        return np.sum(self.seq_lengths)


    @property
    def min_chunk_length(self):
        if self.return_fullseqs:
            self._min_chunk_length = np.min(self.seq_lengths)
        return self._min_chunk_length


    @property
    def max_seq_length(self):
        if self._max_chunk_length is None:
            self._max_chunk_length = np.max(self.seq_lengths)
        return self._max_chunk_length


    @property
    def min_seq_length(self):
        return np.min(self.seq_lengths)


    @property
    def max_seq_length(self):
        return np.max(self.seq_lengths)


    def _prune_short_seqs(self, min_length):
        keep_idx = self.seq_lengths >= min_length
        self.u2c = self.u2c.filter_index(keep_idx)
        self._seq_lengths = self.seq_lengths[keep_idx]
        logger.info('pruned seqs with min_length < %d,'
                    'keep %d/%d seqs' % (
                        min_length, self.num_seqs, len(keep_idx)))

        
    def _prepare_class_info(self, class_file):
        class_weights = None
        if class_file is None:
            classes, class_idx = np.unique(self.u2c.info, return_inverse=True)
            class2idx = {k:i for i, k in enumerate(classes)}
        else:
            logger.info('reading class-file %s' % (class_file))
            class_info = pd.read_csv(class_file, header=None, sep=' ')
            class2idx = {k:i for i,k in enumerate(class_info[0])}
            class_idx = np.array([class2idx[k] for k in self.u2c.info], dtype=int)
            if class_info.shape[1]==2:
                class_weights = np.array(class_info[1])
           
        self.num_classes = len(class2idx)

        class2utt_idx = {}
        class2num_utt = np.zeros((self.num_classes,), dtype=int)
        for k in xrange(self.num_classes):
            idx = (class_idx == k).nonzero()[0]
            class2utt_idx[k] = idx
            class2num_utt[k] = len(idx)
            if class2num_utt[k] == 0:
                logging.warning('class doesn\'t have any samples')
                if class_weights is None:
                    class_weights = np.ones((self.num_classes,), dtype=float_cpu())
                class_weights[k] = 0

        self.utt_idx2class = class_idx
        self.class2utt_idx = class2utt_idx
        self.class2num_utt = class2num_utt
        if class_weights is not None:
            class_weights /= np.sum(class_weights)
        self.class_weights = class_weights




    def __getitem__(self, index):

        if return_fullseqs:
            return self._get_fullseq(index)
        else:
            return self._get_random_chunk(index)
        

            
    def _get_fullseq(self, index):
        key = self.u2c.key[index]
        x = self.r.read([key])
        if not self.return_class:
            return x
        
        class_idx = self.utt_idx2class[index]
        return x, class_idx


    def _get_fullseq(self, index):
        
        key = self.u2c.key[index]
        full_seq_length = self.seq_lengths[index]
        seq_length = self.cur_batch_params['seq_length']
        assert seq_lenght <= full_seq_length
        first_frame = torch.randint(
            low=0, high=full_seq_length-seq_length+1, size=(1,)).item()
        x = self.r.read([key], row_offset=first_frame,
                        num_rows=seq_length)

        if not self.return_class:
            return x
        
        class_idx = self.utt_idx2class[index]
        return x, class_idx




    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'

        valid_args = ('path_prefix', 'class_file',
                      'min_chunk_length', 'max_chunk_length',
                      'return_fullseqs',
                      'part_idx', 'num_parts')
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
            

        parser.add_argument(p1+'path-prefix', dest=(p2+'path_prefix'),
                            default='',
                            help=('path prefix for rspecifier scp file'))

        parser.add_argument(p1+'class-file', dest=(p2+'class_file'), 
                            default=None,
                            help=('ordered list of classes keys, it can contain class weights'))

        parser.add_argument(p1+'min-chunk-length', dest=(p2+'min_chunk_length'),
                            type=int, default=None,
                            help=('minimum length of sequence chunks'))
        parser.add_argument(p1+'max-seq-length', dest=(p2+'max_seq_length'),
                            type=int, default=None,
                            help=('maximum length of sequence chunks'))

        parser.add_argument(p1+'return-fullseqs', dest=(p2+'return_fullseqs'),
                            default=False, action='store_true',
                            help=('returns full sequences instead of chuncks'))
        
        
        # parser.add_argument(p1+'part-idx', dest=(p2+'part_idx'),
        #                     type=int, default=1,
        #                     help=('splits the list of files in num-parts and process part_idx'))
        # parser.add_argument(p1+'num-parts', dest=(p2+'num_parts'),
        #                     type=int, default=1,
        #                     help=('splits the list of files in num-parts and process part_idx'))



    

            
