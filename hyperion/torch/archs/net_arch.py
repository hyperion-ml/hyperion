"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

import torch.nn as nn

class NetArch(nn.Module):

    @property
    def context(self):
        return 0

    def get_config(self):
        config = {
            'class_name': self.__class__.__name__}

        return config
