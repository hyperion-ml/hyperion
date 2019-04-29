"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import torch
import torch.nn as nn
import torch.functional as F
from torch.nn import Dropout2d

class Dropout1d(Dropout2d):
    
    def forward(self, input):
        x = torch.unsqueeze(input, dim=-2)
        x = F.dropout2d(x, self.p, self.training, self.inplace)
        return torch.squeeze(x, dim=-2)

    
