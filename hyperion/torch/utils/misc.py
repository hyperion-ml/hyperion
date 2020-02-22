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


def compute_snr(x, n, axis=-1):
    P_x = 10*torch.log10(torch.mean(x**2, dim=axis))
    P_n = 10*torch.log10(torch.mean(n**2, dim=axis))
    return P_x - P_n
