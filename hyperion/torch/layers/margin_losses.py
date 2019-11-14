"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import torch
import torch.nn as nn


@amp.float_function
def _l2_norm(x, axis=-1):
    norm = torch.norm(x, 2, axis, True)
    y = torch.div(x, norm)
    return y


class ArcLossOutput(nn.Module):

    def __init__(self, in_feats, num_classes, s=64, margin=0.3, margin_inc_epochs=0):
        super(ArcLossOutput, self).__init__()
        self.in_feats = in_feats
        self.num_classes = num_classes
        self.s = s
        self.margin = margin
        self.margin_inc_epochs = margin_inc_epochs
        if margin_inc_epochs == 0:
            self.cur_margin = margin
        else:
            self.cur_margin = 0
        
        self._compute_aux()

        self.kernel = nn.Parameter(torch.Tensor(in_feats, num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)

        
    def _compute_aux(self):

        self.cos_m = math.cos(self.cur_margin)
        self.sin_m = math.sin(self.cur_margin)


    def update_margin(self, epoch):
        
        if self.margin_inc_epochs == 0:
            return

        if epoch < self.margin_inc_epochs:
            self.cur_margin = self.margin*epoch/self.margin_inc_epochs
        else:
            if self.cur_margin != self.margin:
                self.cur_margin = margin
            else:
                return

        self._compute_aux()


    @amp.float_function
    def forward(self, x, y=None):

        s = self.s
        x = _l2_norm(x)
        batch_size = len(x)
        kernel_norm = _l2_norm(self.kernel,axis=0)
        # cos(theta+m)                                      
        cos_theta = torch.mm(x, kernel_norm).float()
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability

        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta

        if y is not None and self.training:
            cos_theta_2 = torch.pow(cos_theta, 2)
            sin_theta_2 = 1 - cos_theta_2
            sin_theta = torch.sqrt(sin_theta_2)
            cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)

            idx_ = torch.arange(0, batch_size, dtype=torch.long)
            output[idx_, y] = cos_theta_m[idx_, y]

        output *= s # scale up in order to make softmax work
        return output



class CosLossOutput(nn.Module):

    def __init__(self, in_feats, num_classes, s=64, margin=0.3, margin_inc_epochs=0):
        super(ArcLossOutput, self).__init__()
        self.in_feats = in_feats
        self.num_classes = num_classes
        self.s = s
        self.margin = margin
        self.margin_inc_epochs = margin_inc_epochs
        if margin_inc_epochs == 0:
            self.cur_margin = margin
        else:
            self.cur_margin = 0
        
        self.kernel = nn.Parameter(torch.Tensor(in_feats, num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        

    def update_margin(self, epoch):
        
        if self.margin_inc_epochs == 0:
            return

        if epoch < self.margin_inc_epochs:
            self.cur_margin = self.margin*epoch/self.margin_inc_epochs
        else:
            if self.cur_margin != self.margin:
                self.cur_margin = margin
            else:
                return


    @amp.float_function
    def forward(self, x, y=None):

        s = self.s
        x = _l2_norm(x)
        batch_size = len(x)
        kernel_norm = _l2_norm(self.kernel,axis=0)
        # cos(theta+m)                                      
        cos_theta = torch.mm(x, kernel_norm).float()
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability

        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        if y is not None and self.training:
            cos_theta_m = cos_theta - self.cur_margin
            idx_ = torch.arange(0, batch_size, dtype=torch.long)
            output[idx_, y] = cos_theta_m[idx_, y]

        output *= s # scale up in order to make softmax work
        return output
