"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch


str2torch_dtype = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
}

torch_dtype2str = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
}


def floatstr_torch():
    return torch_dtype2str[torch.get_default_dtype()]


def float_torch():
    return torch.get_default_dtype()


def set_float_torch(float_torch):
    if isinstance(float_torch, str):
        float_torch = str2torch_dtype[float_torch]

    torch.set_default_dtype(float_torch)
