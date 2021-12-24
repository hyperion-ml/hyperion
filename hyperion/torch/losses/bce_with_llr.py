"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as nnf


class BCEWithLLR(nn.Module):
    def __init__(self, p_tar=0.5):
        super().__init__()
        self.p_tar = p_tar
        self.logit_ptar = math.log(p_tar / (1 - p_tar))

    # def forward(self, x, y, is_selfsim=False, is_sim=False, y2=None):
    #     # logging.info('{} {}'.format(x.shape[0], y.shape[0]))
    #     if is_selfsim or is_sim:
    #         assert x.dim() > 1
    #         # x is a full score matrix
    #         # y contains the labels of the rows
    #         y1 = y
    #         if is_selfsim:
    #             y2 = y

    #         assert x.shape[0] == y1.shape[0]
    #         assert x.shape[1] == y2.shape[0]
    #         y = y1.unsqueeze(-1) - y2.unsqueeze(0) + 1
    #         y[y!=1] = 0
    #         if is_selfsim:
    #             #if it is selfsim we only use the upper trianglaur
    #             mask=torch.triu(torch.ones_like(x, dtype=torch.bool),
    #                             diagonal=1)
    #             x = x[mask]
    #             y = y[mask]

    #     # y = y.double()
    #     # ntar = torch.sum(y, dim=0).float()
    #     # nnon = torch.sum(1-y, dim=0).float()
    #     y = y.float()
    #     ntar = torch.mean(y, dim=0)
    #     nnon = torch.mean(1-y, dim=0)
    #     # logging.info('{} {} {} {} {} {}'.format(ntar, nnon, ntar+nnon, len(x)-ntar-nnon, x.shape, torch.unique(y)))
    #     weight_tar = self.p_tar / ntar
    #     weight_non = (1 - self.p_tar) / nnon
    #     x = x + self.logit_ptar
    #     weight = y * weight_tar + (1-y) * weight_non
    #     loss = nnf.binary_cross_entropy_with_logits(
    #         x, y, weight=weight, reduction='mean')
    #     return loss

    def forward(self, x, y):
        y = y.float()
        ntar = torch.mean(y, dim=0)
        nnon = torch.mean(1 - y, dim=0)
        weight_tar = self.p_tar / ntar
        weight_non = (1 - self.p_tar) / nnon
        x = x + self.logit_ptar
        weight = y * weight_tar + (1 - y) * weight_non
        loss = nnf.binary_cross_entropy_with_logits(
            x, y, weight=weight, reduction="mean"
        )
        return loss
