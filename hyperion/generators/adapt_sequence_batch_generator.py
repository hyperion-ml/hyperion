"""
 Copyright 2018 Jesus Villalba (Johns Hopkins University)
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
from sklearn.utils.class_weight import compute_class_weight

from ..hyp_defs import float_cpu
from ..io import RandomAccessDataReaderFactory as RF
from ..utils.scp_list import SCPList
from ..utils.tensors import to3D_by_seq
from ..transforms import TransformList
from .sequence_batch_generator_v1 import SequenceBatchGeneratorV1 as SBG


class AdaptSequenceBatchGenerator(SBG):

    def __init__(self, rspecifier,
                 key_file, key_file_adapt,
                 r_adapt=1,
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
                 seq_weight = 'balanced',
                 shuffle_seqs=True,
                 transform=None,
                 init_epoch=0,
                 sg_seed=1024, reset_rng=False,
                 scp_sep=' ', 
                 part_idx=1, num_parts=1):


        self.scp_adapt = SCPList.load(key_file_adapt, sep=scp_sep)
        if num_parts > 1:
            self.scp_adapt = self.scp_adapt.split(part_idx, num_parts, group_by_key=False)

        assert r_adapt < batch_size
        self.r_adapt = r_adapt

        self._init_seq_lengths_adapt = None
        self._seq_lengths_adapt = None

        self.init_scp_adapt = self.scp_adapt

        self.cur_seq_adapt = 0
        self.cur_frame_adapt = None

        self.cur_subseq = None
        self._init_num_subseqs_adapt = None
        self.num_subseqs_adapt = None

        
        super(AdaptSequenceBatchGenerator, self).__init__(
            rspecifier, key_file, class_list, path_prefix, batch_size,
            iters_per_epoch, gen_method, min_seq_length, max_seq_length, seq_overlap,
            prune_min_length, return_class, class_weight, seq_weight,
            shuffle_seqs, transform, init_epoch, sg_seed, reset_rng, scp_sep,
            part_idx,num_parts)



    @property
    def num_seqs(self):
        return len(self.scp)


    
    @property
    def num_seqs_adapt(self):
        return len(self.scp_adapt)


    
    @property
    def seq_lengths(self):
        if self._seq_lengths is None:
            self._init_seq_lengths = self.r.read_num_rows(self.scp.file_path)
            self._seq_lengths = self._init_seq_lengths
        return self._seq_lengths



    @property
    def seq_lengths_adapt(self):
        if self._seq_lengths_adapt is None:
            self._init_seq_lengths_adapt = self.r.read_num_rows(self.scp_adapt.file_path)
            self._seq_lengths_adapt = self._init_seq_lengths_adapt
        return self._seq_lengths_adapt


    
    @property
    def total_length(self):
        return np.sum(self.seq_lengths)



    @property
    def total_length_adapt(self):
        return np.sum(self.seq_lengths_adapt)



    
    @property
    def min_seq_length(self):
        if self._min_seq_length is None:
            self._min_seq_length = min(np.min(self.seq_lengths), np.min(self.seq_lengths_adapt))
        return self._min_seq_length


    
    @property
    def max_seq_length(self):
        if self._max_seq_length is None:
            self._max_seq_length = max(np.max(self.seq_lengths), np.max(self.seq_lengths_adapt))
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
                self.iters_per_epoch * seqs_per_iter/(self.batch_size-self.r_adapt)))
        
        return self._steps_per_epoch


        
    @property
    def num_total_subseqs(self):
        return self.steps_per_epoch * self.batch_size


    
    def _prune_min_length(self, min_length):
        keep_idx = self.seq_lengths >= min_length
        self.scp = self.scp.filter_index(keep_idx)
        keep_idx = self.seq_lengths_adapt >= min_length
        self.scp_adapt = self.scp_adapt.filter_index(keep_idx)

        self._seq_lengths = None
        self._seq_lengths_adapt = None


        
    def _prepare_class_info(self, class_list):
        if class_list is None:
            class_dict = {k:i for i, k in enumerate(np.unique(self.scp.key))}
            class_dict.update({k:i for i, k in enumerate(np.unique(self.scp_adapt.key))})
        else:
            with open(class_list) as f:
                class_dict={line.rstrip().split()[0]: i for i, line in enumerate(f)}

        self.num_classes = len(class_dict)
        self.key2class = {p: class_dict[k] for k, p in zip(self.scp.key, self.scp.file_path)}
        self.key2class.update({p: class_dict[k] for k, p in zip(self.scp_adapt.key, self.scp_adapt.file_path)})


        
    def _balance_class_weight(self):
        super(AdaptSequenceBatchGenerator, self)._balance_class_weight()
        classes, class_ids = np.unique(self.scp_adapt.key, return_inverse=True)

        idx = self._balance_class_weigth_helper(class_ids)
        
        self.scp_adapt = self.scp_adapt.filter_index(idx)
        assert len(self.scp_adapt) == len(num_samples)*max_samples
                   
        if self._init_seq_lengths_adapt is not None:
            self._init_seq_legths_adapt = self._init_seq_lengths_adapt[idx]
            self._seq_lengths_adapt = self._init_seq_legths_adapt
        

            

    def _prepare_full_seqs(self):
        pass


    def _prepare_random_subseqs(self):
        pass


    
    def _prepare_sequential_subseqs(self):
        super(AdaptSequenceBatchGenerator, self)._prepare_sequential_subseqs()
        seq_lengths = self.seq_lengths_adapt
        avg_length = int((self.max_seq_length + self.min_seq_length)/2)
        shift = avg_length - self.seq_overlap
        self._init_num_subseqs_adapt = np.ceil(seq_lengths/shift).astype(int)
        self.num_subseqs_adapt = self._init_num_subseqs_adapt
        self.cur_frame_adapt = np.zeros((self.num_seqs_adapt,), dtype=int)
        self.cur_subseq_adapt = np.zeros((self.num_seqs_adapt,), dtype=int)

    
            
    def reset(self):
        super(AdaptSequenceBatchGenerator, self).reset()
        
        self.cur_seq_adapt = 0

        if self.shuffle_seqs:
            if self._init_seq_lengths_adapt is None:
                self.seq_lengths_adapt
                
            self.scp_adapt = self.init_scp_adapt.copy()
            index = self.scp_adapt.shuffle(rng=self.rng)
            self._seq_lengths_adapt = self._init_seq_lengths_adapt[index]
            if self._init_num_subseqs_adapt is not None:
                self.num_subseqs_adapt = self._init_num_subseqs_adapt[index]

                
        if self.gen_method == 'sequential':
            self.cur_subseq_adapt[:] = 0
            self.cur_frame_adapt[:] = 0
            


    

    
    def _read_full_seqs(self):
        batch_size = self.batch_size - self.r_adapt
        keys = list(self.scp.file_path[self.cur_seq:self.cur_seq+batch_size])
        self.cur_seq += batch_size
        if len(keys) < batch_size:
            delta = batch_size - len(keys)
            keys += self.scp.file_path[:delta]
            self.cur_seq = delta
            assert len(keys) == batch_size

        batch_size = self.r_adapt
        keys_adapt = list(self.scp_adapt.file_path[self.cur_seq_adapt:self.cur_seq_adapt+batch_size])
        self.cur_seq_adapt += batch_size
        if len(keys_adapt) < batch_size:
            delta = batch_size - len(keys)
            keys_adapt += self.scp_adapt.file_path[:delta]
            self.cur_seq_adapt = delta
            assert len(keys_adapt) == batch_size

        keys += keys_adapt
            
        return keys, self.r.read(keys)


    
    def _read_random_subseqs(self):
        
        keys = []
        seq_lengths =[]
        first_frames = []
        for i in xrange(self.batch_size-self.r_adapt):
            key = self.scp.file_path[self.cur_seq]
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

        for i in xrange(self.r_adapt):
            key = self.scp_adapt.file_path[self.cur_seq_adapt]
            full_seq_length = self.seq_lengths_adapt[self.cur_seq_adapt]
            max_seq_length = min(full_seq_length, self.max_seq_length)
            min_seq_length = min(full_seq_length, self.min_seq_length)
            
            seq_length = self.rng.randint(low=min_seq_length, high=max_seq_length+1)
            first_frame = self.rng.randint(
                low=0, high=full_seq_length-seq_length+1)

            keys.append(key)
            seq_lengths.append(seq_length)
            first_frames.append(first_frame)

            self.cur_seq_adapt = (self.cur_seq_adapt + 1) % self.num_seqs_adapt
            
        return keys, self.r.read(keys, row_offset=first_frames,
                                 num_rows=seq_lengths)
            

    
    def _read_sequential_subseqs(self):

        keys = []
        seq_lengths =[]
        first_frames = []
        count = 0
        while count < self.batch_size - self.r_adapt:
            key = self.scp.file_path[self.cur_seq]
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


        while count < self.batch_size:
            key = self.scp_adapt.file_path[self.cur_seq_adapt]
            first_frame = self.cur_frame_adapt[self.cur_seq_adapt]
            full_seq_length = self.seq_lengths_adapt[self.cur_seq_adapt]
            remainder_seq_length =  full_seq_length - first_frame
            if self.cur_subseq_adapt[self.cur_seq_adapt] == self.num_subseqs_adapt[self.cur_seq_adapt]:
                self.cur_seq_adapt = (self.cur_seq_adapt + 1) % self.num_seqs_adapt
                continue
            if self.cur_subseq_adapt[self.cur_seq_adapt] == self.num_subseqs_adapt[self.cur_seq_adapt]-1:
                seq_length = min(remainder_seq_length, self.max_seq_length)
                self.cur_frame_adapt[self.cur_seq_adapt] = 0
            else:
                max_seq_length = min(
                    max(self.min_seq_length,
                        remainder_seq_length-self.min_seq_length),
                    self.max_seq_length)
                min_seq_length = min(remainder_seq_length, self.min_seq_length)
                seq_length = self.rng.randint(low=min_seq_length, high=max_seq_length+1)
                self.cur_frame_adapt[self.cur_seq_adapt] = min(
                    full_seq_length - self.min_seq_length,
                    first_frame + seq_length - self.seq_overlap)
            
            keys.append(key)
            seq_lengths.append(seq_length)
            first_frames.append(first_frame)

            self.cur_subseq_adapt[self.cur_seq_adapt] += 1
            if self.seq_weight == 'balanced':
                self.cur_subseq_adapt[self.cur_seq_adapt] %= self.num_subseqs_adapt[self.cur_seq_adapt]

            self.cur_seq_adapt = (self.cur_seq_adapt + 1) % self.num_seqs_adapt
            count += 1

            
        assert len(keys) == self.batch_size
        return keys, self.r.read(keys, row_offset=first_frames,
                                 num_rows=seq_lengths)
        
    
    

    @staticmethod
    def filter_args(prefix=None, **kwargs):
        args = super(AdaptSequenceBatchGenerator,
                     AdaptSequenceBatchGenerator).filter_args(prefix, **kwargs)
        
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'

        valid_args = ('r_adapt',)
        new_args = dict((k, kwargs[p+k])
                        for k in valid_args if p+k in kwargs)
        args.update(new_args)
        return args


    
    @staticmethod
    def add_argparse_args(parser, prefix=None):
        args = super(AdaptSequenceBatchGenerator,
                     AdaptSequenceBatchGenerator).add_argparse_args(parser, prefix)
        
        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'

        parser.add_argument(p1+'r-adapt', dest=(p2+'r_adapt'),
                            default=64, type=int,
                            help=('batch size of adaptation data.'))


