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

# from ..io import HypDataReader
# from ..utils.scp_list import SCPList
# from ..utils.tensors import to3D_by_seq
# from ..transforms import TransformList
from ..hyp_defs import float_cpu
from .sequence_reader import SequenceReader

class SequenceClassReader(SequenceReader):
    """Class to read sequences and sequence_class_id (deprecated)
    """
    def __init__(self, data_file, key_file, classes_file, **kwargs):
        super(SequenceClassReader, self).__init__(data_file, key_file, **kwargs)

        self.key_class=None
        self.num_classes=0
        with open(classes_file) as f:
            class_dict={line.rstrip().split()[0]: i for i, line in enumerate(f)}
            self.num_classes=len(class_dict)
            self.key_class={p: class_dict[k] for k, p in zip(self.scp.key, self.scp.file_path)}
            


    def read(self, return_3d=False,
             max_seq_length=0, return_sample_weight=True):

        r = super(SequenceClassReader, self).read(
            return_3d=return_3d, max_seq_length=max_seq_length,
            return_sample_weight=return_sample_weight)
        keys = r[-1]

        y=np.zeros((len(keys), self.num_classes), dtype=float_cpu())
        for i,k in enumerate(keys):
            y[i, self.key_class[k]] = 1

        r += (y,)
        return r

            
