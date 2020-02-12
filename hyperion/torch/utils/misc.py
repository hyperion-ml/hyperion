"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import torch
from apex import amp

@amp.float_function
def l2_norm(x, axis=-1):
    norm = torch.norm(x, 2, axis, True) + 1e-10
    y = torch.div(x, norm)
    return y

