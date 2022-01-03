"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import logging
import math

import torch
import torch.nn as nn
import torch.cuda.amp as amp


def _l2_norm(x, axis=-1):
    with amp.autocast(enabled=False):
        norm = torch.norm(x.float(), 2, axis, True) + 1e-10
        y = torch.div(x, norm)
    return y


class ArcLossOutput(nn.Module):
    def __init__(
        self, in_feats, num_classes, cos_scale=64, margin=0.3, margin_warmup_epochs=0
    ):
        super().__init__()
        self.in_feats = in_feats
        self.num_classes = num_classes
        self.cos_scale = cos_scale
        self.margin = margin
        self.margin_warmup_epochs = margin_warmup_epochs
        if margin_warmup_epochs == 0:
            self.cur_margin = margin
        else:
            self.cur_margin = 0

        self._compute_aux()

        self.kernel = nn.Parameter(torch.Tensor(in_feats, num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "%s(in_feats=%d, num_classes=%d, cos_scale=%.2f, margin=%.2f, margin_warmup_epochs=%d)" % (
            self.__class__.__name__,
            self.in_feats,
            self.num_classes,
            self.cos_scale,
            self.margin,
            self.margin_warmup_epochs,
        )
        return s

    def _compute_aux(self):
        logging.info("updating arc-softmax margin=%.2f" % (self.cur_margin))
        self.cos_m = math.cos(self.cur_margin)
        self.sin_m = math.sin(self.cur_margin)

    def update_margin(self, epoch):
        if self.margin_warmup_epochs == 0:
            return

        if epoch < self.margin_warmup_epochs:
            self.cur_margin = self.margin * epoch / self.margin_warmup_epochs
        else:
            if self.cur_margin != self.margin:
                self.cur_margin = self.margin
            else:
                return

        self._compute_aux()

    def forward(self, x, y=None):
        with amp.autocast(enabled=False):
            s = self.cos_scale
            batch_size = len(x)
            x = _l2_norm(x.float())
            kernel_norm = _l2_norm(self.kernel, axis=0)
            cos_theta = torch.mm(x, kernel_norm).float()
            cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
            output = (
                cos_theta * 1.0
            )  # a little bit hacky way to prevent in_place operation on cos_theta

            if y is not None and self.training:
                cos_theta_2 = torch.pow(cos_theta, 2)
                sin_theta_2 = (1 + 1e-10) - cos_theta_2
                sin_theta = torch.sqrt(sin_theta_2)
                cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

                idx_ = torch.arange(0, batch_size, dtype=torch.long)
                output[idx_, y] = cos_theta_m[idx_, y]

            output *= s  # scale up in order to make softmax work
            return output


class CosLossOutput(nn.Module):
    def __init__(
        self, in_feats, num_classes, cos_scale=64, margin=0.3, margin_warmup_epochs=0
    ):
        super().__init__()
        self.in_feats = in_feats
        self.num_classes = num_classes
        self.cos_scale = cos_scale
        self.margin = margin
        self.margin_warmup_epochs = margin_warmup_epochs
        if margin_warmup_epochs == 0:
            self.cur_margin = margin
        else:
            self.cur_margin = 0

        self.kernel = nn.Parameter(torch.Tensor(in_feats, num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def update_margin(self, epoch):
        if self.margin_warmup_epochs == 0:
            return

        if epoch < self.margin_warmup_epochs:
            self.cur_margin = self.margin * epoch / self.margin_warmup_epochs
            logging.info("updating cos-softmax margin=%.2f" % (self.cur_margin))
        else:
            if self.cur_margin != self.margin:
                self.cur_margin = self.margin
                logging.info("updating cos-softmax margin=%.2f" % (self.cur_margin))
            else:
                return

    def forward(self, x, y=None):
        with amp.autocast(enabled=False):
            s = self.cos_scale
            x = _l2_norm(x.float())
            batch_size = len(x)
            kernel_norm = _l2_norm(self.kernel, axis=0)
            # cos(theta+m)
            cos_theta = torch.mm(x, kernel_norm).float()
            cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

            output = (
                cos_theta * 1.0
            )  # a little bit hacky way to prevent in_place operation on cos_theta
            if y is not None and self.training:
                cos_theta_m = cos_theta - self.cur_margin
                idx_ = torch.arange(0, batch_size, dtype=torch.long)
                output[idx_, y] = cos_theta_m[idx_, y]

            output *= s  # scale up in order to make softmax work
            return output


class SubCenterArcLossOutput(ArcLossOutput):
    def __init__(
        self,
        in_feats,
        num_classes,
        num_subcenters=2,
        cos_scale=64,
        margin=0.3,
        margin_warmup_epochs=0,
    ):
        super().__init__(
            in_feats,
            num_classes * num_subcenters,
            cos_scale,
            margin,
            margin_warmup_epochs,
        )
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters

    def __str__(self):
        s = "%s(in_feats=%d, num_classes=%d, num_subcenters=%d, cos_scale=%.2f, margin=%.2f, margin_warmup_epochs=%d)" % (
            self.__class__.__name__,
            self.in_feats,
            self.num_classes,
            self.num_subcenters,
            self.cos_scale,
            self.margin,
            self.margin_warmup_epochs,
        )
        return s

    def forward(self, x, y=None):
        with amp.autocast(enabled=False):
            s = self.cos_scale
            batch_size = len(x)
            x = _l2_norm(x.float())
            kernel_norm = _l2_norm(self.kernel, axis=0)
            # cos(theta+m)
            cos_theta = torch.mm(x, kernel_norm).float()
            cos_theta = torch.max(
                cos_theta.view(-1, self.num_classes, self.num_subcenters), dim=-1
            )[0]

            cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
            # print(cos_theta)
            output = (
                cos_theta * 1.0
            )  # a little bit hacky way to prevent in_place operation on cos_theta

            if y is not None and self.training:
                cos_theta_2 = torch.pow(cos_theta, 2)
                sin_theta_2 = (1 + 1e-10) - cos_theta_2
                sin_theta = torch.sqrt(sin_theta_2)
                cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

                idx_ = torch.arange(0, batch_size, dtype=torch.long)
                output[idx_, y] = cos_theta_m[idx_, y]

            output *= s  # scale up in order to make softmax work
            return output
