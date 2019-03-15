"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch

_FLOAT_TORCH=torch.float32

def float_torch():
    return _FLOAT_TORCH


def set_float_cpu(float_torch):
    global _FLOAT_TORCH
    _FLOAT_TORCH = float_torch

    
def float_torch_str():
    #TODO
    return 'float32'
