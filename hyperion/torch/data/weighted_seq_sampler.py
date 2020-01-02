"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
import logging

import numpy as np

import torch
from torch.utils.data import Sampler


class ClassWeightedSeqSampler(Sampler):

    def __init__(self, dataset, batch_size=1, iters_per_epoch='auto',
                 num_egs_per_class=1, num_egs_per_utt=1):
        
        super(ClassWeightedSeqSampler, self).__init__(None)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_egs_per_class = num_egs_per_class
        self.num_egs_per_utt = num_egs_per_utt
        self.batch = 0
        
        if iters_per_epoch == 'auto':
            self._compute_iters_auto()
        else:
            self.iters_per_epoch = iters_per_epoch

        self._len = int(math.ceil(
            self.iters_per_epoch * dataset.num_seqs / batch_size))

        logging.info('num batches per epoch: %d' % self._len)
        
        self._num_classes_per_batch = int(math.ceil(
            batch_size/num_egs_per_class/num_egs_per_utt))
        logging.info('num classes per batch: %d' % self._num_classes_per_batch)

        #self.weights = torch.as_tensor(dataset.class_weights, dtype=torch.double)
        

        
    def _compute_iters_auto(self):
        dataset = self.dataset
        avg_seq_length = np.mean(dataset.seq_lengths)
        avg_chunk_length = int((dataset.max_seq_length + dataset.min_seq_length)/2)
        self.iters_per_epoch = math.ceil(avg_seq_length/avg_chunk_length)
        logging.debug('num iters per epoch: %d' % self.iters_per_epoch)


    def __len__(self):
        return self._len
        

        
    def __iter__(self):
        self.batch = 0
        return self


    def __next__(self):

        if self.batch == self._len:
            raise StopIteration
        
        dataset = self.dataset
        if dataset.class_weights is None:
            class_idx = torch.randint(low=0, high=dataset.num_classes, 
                                      size=(self._num_classes_per_batch,))
        else:
            class_idx = torch.multinomial(
                dataset.class_weights, 
                num_samples=self._num_classes_per_batch, replacement=True)

        if self.num_egs_per_class > 1:
            class_idx = class_idx.repeat(self.num_egs_per_class)

        utt_idx = torch.as_tensor([
            self.dataset.class2utt_idx[c][
                torch.randint(low=0, high=int(self.dataset.class2num_utt[c]), size=(1,))] 
            for c in class_idx.tolist()])

        if self.num_egs_per_utt > 1:
            utt_idx = utt_idx.repeat(self.num_egs_per_utt)

        if self.batch == 0:
            logging.info('batch 0 classidx=%s', str(class_idx[:10]))
            logging.info('batch 0 uttidx=%s', str(utt_idx[:10]))

        self.batch += 1
  
        self.dataset.set_random_chunk_length()
        return utt_idx.tolist()[:self.batch_size]

    

    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'

        if p+'no_shuffle_seqs' in kwargs:
            kwargs[p+'shuffle_seqs'] = not kwargs[p+'no_shuffle_seqs']
            
        valid_args = ('batch_size',
                      'iters_per_epoch',
                      'num_egs_per_class', 'num_egs_per_utt')
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
            
        parser.add_argument(p1+'batch-size', dest=(p2+'batch_size'),
                            default=128, type=int,
                            help=('batch size'))

        parser.add_argument(
            p1+'iters-per-epoch', dest=(p2+'iters_per_epoch'),
            default='auto',
            type=lambda x: x if x=='auto' else int(x),
            help=('number of times we sample an utterance in each epoch'))

        parser.add_argument(p1+'num-egs-per-class',
                            dest=(p2+'num_egs_per_class'),
                            type=int, default=1,
                            help=('number of samples per class in batch'))
        parser.add_argument(p1+'num-egs-per-utt',
                            dest=(p2+'num_egs_per_utt'),
                            type=int, default=1,
                            help=('number of samples per utterance in batch'))


