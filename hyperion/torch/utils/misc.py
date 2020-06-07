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


def compute_stats_adv_attack(x, x_adv):

    if x.dim() > 2:
        x = torch.flatten(x, start_dim=1)
        x_adv = torch.flatten(x_adv, start_dim=1)

    noise = x_adv - x
    P_x = 10*torch.log10(torch.mean(x**2, dim=-1))
    P_n = 10*torch.log10(torch.mean(noise**2, dim=-1))
    snr = P_x - P_n
    #x_l1 = torch.sum(torch.abs(x), dim=-1)
    x_l2 = torch.norm(x, dim=-1)
    x_linf = torch.max(x, dim=-1)[0]
    abs_n = torch.abs(noise)
    n_l0 = torch.sum(abs_n>0, dim=-1).float()
    #n_l1 = torch.sum(abs_n, dim=-1)
    n_l2 = torch.norm(noise, dim=-1)
    n_linf = torch.max(noise, dim=-1)[0]
    return snr, P_x, P_n, x_l2, x_linf, n_l0, n_l2, n_linf
    