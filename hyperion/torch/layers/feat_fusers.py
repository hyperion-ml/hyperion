"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math

import torch
import torch.nn as nn


class FeatFuser(nn.Module):
    def __init__(self):
        super().__init__()


class _ProjFeatFuser(FeatFuser):
    def __init__(self, feat_dim=None, proj_dim=None, proj_bias=True):
        super().__init__()
        self.feat_dim = feat_dim
        self.proj_dim = proj_dim
        self.feat_proj = None
        if feat_dim is not None and proj_dim is not None:
            self.feat_proj = nn.Linear(feat_dim, proj_dim, bias=proj_bias)


class LastFeatFuser(_ProjFeatFuser):
    def __init__(self, feat_dim=None, proj_dim=None, proj_bias=True):
        super().__init__(feat_dim, proj_dim, proj_bias)

    def forward(self, feats):
        feats = feats[-1]
        if self.feat_proj is not None:
            feats = self.feat_proj(feats)

        return feats


class WeightedAvgFeatFuser(_ProjFeatFuser):
    def __init__(self, num_feats, feat_dim=None, proj_dim=None, proj_bias=True):
        super().__init__(feat_dim, proj_dim, proj_bias)
        self.num_feats = num_feats
        self.feat_fuser = nn.Parameter(torch.zeros(num_feats))

    def forward(self, feats):
        feats = torch.stack(feats, dim=-1)
        norm_weights = nn.functional.softmax(self.feat_fuser, dim=-1)
        feats = torch.sum(feats * norm_weights, dim=-1)
        if self.feat_proj is not None:
            feats = self.feat_proj(feats)

        return feats


class LinearFeatFuser(_ProjFeatFuser):
    def __init__(self, num_feats, feat_dim=None, proj_dim=None, proj_bias=True):
        super().__init__(feat_dim, proj_dim, proj_bias)
        self.num_feats = num_feats
        self.feat_fuser = nn.Linear(num_feats, 1, bias=False)
        self.feat_fuser.weight.data = torch.ones(1, num_feats) / num_feats

    def forward(self, feats):
        feats = torch.stack(feats, dim=-1)
        feats = self.feat_fuser(feats).squeeze(dim=-1)
        if self.feat_proj is not None:
            feats = self.feat_proj(feats)

        return feats


class CatFeatFuser(FeatFuser):
    def __init__(self, num_feats, feat_dim, proj_dim=None, proj_bias=True):
        super().__init__()
        self.num_feats = num_feats
        self.feat_dim = feat_dim
        if proj_dim is None:
            proj_dim = feat_dim
        self.proj_dim = proj_dim
        self.proj_bias = proj_bias
        self.feat_fuser = nn.Linear(num_feats * feat_dim, proj_dim, bias=proj_bias)

    def forward(self, feats):
        feats = torch.cat(feats, dim=-1)
        feats = self.feat_fuser(feats)
        return feats
