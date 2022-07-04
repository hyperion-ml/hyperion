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


def _cosine_affinity(kernel):
    kernel_norm = _l2_norm(kernel, axis=0)
    return torch.mm(kernel_norm.transpose(0, 1), kernel_norm)


class ArcLossOutput(nn.Module):
    """Additive angular margin softmax (ArcFace) output layer.

    It includes the option to also use InterTopK penalty:
    https://arxiv.org/abs/2109.01989

    Attributes:
      in_feats: input feature dimension.
      num_classes: number of output classes.
      cos_scale: cosine scale.
      margin: angular margin.
      margin_warmup_epochs: number of epochs to warm up the margin from 0 to
                            its final value.
      intertop_k: adds negative angular penalty to k largest negative scores.
      intertop_margin: inter-top-k penalty.
    """

    def __init__(
        self,
        in_feats,
        num_classes,
        cos_scale=64,
        margin=0.3,
        margin_warmup_epochs=0,
        intertop_k=5,
        intertop_margin=0,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.num_classes = num_classes
        self.cos_scale = cos_scale
        self.margin = margin
        self.margin_warmup_epochs = margin_warmup_epochs
        self.intertop_k = intertop_k
        self.intertop_margin = intertop_margin
        if margin_warmup_epochs == 0:
            self.cur_margin = margin
            self.cur_intertop_margin = intertop_margin
        else:
            self.cur_margin = 0
            self.cur_intertop_margin = 0

        self._compute_aux()

        # each column is the prototype vector of a class
        self.kernel = nn.Parameter(torch.Tensor(in_feats, num_classes))
        # we normalize prototypes to have l2 norm = 1
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "%s(in_feats=%d, num_classes=%d, cos_scale=%.2f, margin=%.2f, margin_warmup_epochs=%d, intertop_k=%d, intertop_margin=%f)" % (
            self.__class__.__name__,
            self.in_feats,
            self.num_classes,
            self.cos_scale,
            self.margin,
            self.margin_warmup_epochs,
            self.intertop_k,
            self.intertop_margin,
        )
        return s

    def _compute_aux(self):
        logging.info(
            "updating arc-softmax margin=%.2f intertop-margin=%.2f",
            self.cur_margin,
            self.cur_intertop_margin,
        )
        self.cos_m = math.cos(self.cur_margin)
        self.sin_m = math.sin(self.cur_margin)
        self.intertop_cos_m = math.cos(self.cur_intertop_margin)
        self.intertop_sin_m = math.sin(self.cur_intertop_margin)

    def update_margin(self, epoch):
        """Updates the value of the margin.

        Args:
          epoch: value of current epoch.
        """
        if epoch < self.margin_warmup_epochs:
            self.cur_margin = self.margin * epoch / self.margin_warmup_epochs
            self.cur_intertop_margin = (
                self.intertop_margin * epoch / self.margin_warmup_epochs
            )
        else:
            if self.cur_margin != self.margin:
                self.cur_margin = self.margin
                self.cur_intertop_margin = self.intertop_margin
            else:
                return

        self._compute_aux()

    def forward(self, x, y=None):
        """Computes penalized logits.

        Args:
          x: input feature tensor with shape = (batch, in_feats).
          y: ground truth classes. This is required to penalize the logit of
             the true class at training time.

        Returns:
          Logit tensor with shape = (batch, num_classes)
        """
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
                if self.cur_intertop_margin > 0:
                    # implementation of intertop-K
                    # set positive scores to -inf so they don't appear in the top k
                    cos_aux = cos_theta * 1
                    cos_aux[idx_, y] = -1e10
                    # find topk indices for negative samples
                    topk = torch.topk(cos_aux, k=self.intertop_k, dim=-1, sorted=False)
                    idx_ = (
                        idx_.unsqueeze(-1).expand(batch_size, self.intertop_k).flatten()
                    )
                    topk_idx = topk.indices.flatten()
                    # compute cos(theta-m')
                    cos_theta_m = (
                        cos_theta[idx_, topk_idx] * self.intertop_cos_m
                        + sin_theta[idx_, topk_idx] * self.intertop_sin_m
                    )
                    # take the maximum for the cases where m' is larger than theta to get cos(max(0, theta-m'))
                    output[idx_, topk_idx] = torch.maximum(
                        output[idx_, topk_idx], cos_theta_m
                    )

            output *= s  # scale up in order to make softmax work
            return output

    def compute_prototype_affinity(self):
        return _cosine_affinity(self.kernel)


class CosLossOutput(nn.Module):
    """Additive margin softmax (CosFace) output layer.

    Attributes:
      in_feats: input feature dimension.
      num_classes: number of output classes.
      cos_scale: cosine scale.
      margin: angular margin.
      margin_warmup_epochs: number of epochs to warm up the margin from 0 to
                            its final value.
      intertop_k: adds negative angular penalty to k largest negative scores.
      intertop_margin: inter-top-k penalty.
    """

    def __init__(
        self,
        in_feats,
        num_classes,
        cos_scale=64,
        margin=0.3,
        margin_warmup_epochs=0,
        intertop_k=5,
        intertop_margin=0.0,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.num_classes = num_classes
        self.cos_scale = cos_scale
        self.margin = margin
        self.margin_warmup_epochs = margin_warmup_epochs
        self.intertop_k = intertop_k
        self.intertop_margin = intertop_margin
        if margin_warmup_epochs == 0:
            self.cur_margin = margin
            self.cur_intertop_margin = intertop_margin
        else:
            self.cur_margin = 0
            self.cur_intertop_margin = 0

        self.kernel = nn.Parameter(torch.Tensor(in_feats, num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "%s(in_feats=%d, num_classes=%d, cos_scale=%.2f, margin=%.2f, margin_warmup_epochs=%d, intertop_k=%d, intertop_margin=%f)" % (
            self.__class__.__name__,
            self.in_feats,
            self.num_classes,
            self.cos_scale,
            self.margin,
            self.margin_warmup_epochs,
            self.intertop_k,
            self.intertop_margin,
        )
        return s

    def update_margin(self, epoch):
        """Updates the value of the margin.

        Args:
          epoch: value of current epoch.
        """
        # if self.margin_warmup_epochs == 0:
        #    return

        if epoch < self.margin_warmup_epochs:
            self.cur_margin = self.margin * epoch / self.margin_warmup_epochs
            logging.info(
                "updating cos-softmax margin=%.2f intertop-margin=%.2f",
                self.cur_margin,
                self.cur_intertop_margin,
            )
            self.cur_intertop_margin = (
                self.intertop_margin * epoch / self.margin_warmup_epochs
            )
        else:
            if self.cur_margin != self.margin:
                self.cur_margin = self.margin
                self.cur_intertop_margin = self.intertop_margin
                logging.info(
                    "updating cos-softmax margin=%.2f intertop-margin=%.2f",
                    self.cur_margin,
                    self.cur_intertop_margin,
                )
            else:
                return

    def forward(self, x, y=None):
        """Computes penalized logits.

        Args:
          x: input feature tensor with shape = (batch, in_feats).
          y: ground truth classes. This is required to penalize the logit of
             the true class at training time.

        Returns:
          Logit tensor with shape = (batch, num_classes)
        """
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
                if self.cur_intertop_margin > 0:
                    # implementation of intertop-K
                    # set positive scores to -inf so they don't appear in the top k
                    cos_aux = cos_theta * 1
                    cos_aux[idx_, y] = -1e10
                    # find topk indices for negative samples
                    topk = torch.topk(cos_aux, k=self.intertop_k, dim=-1, sorted=False)
                    idx_ = (
                        idx_.unsqueeze(-1).expand(batch_size, self.intertop_k).flatten()
                    )
                    topk_idx = topk.indices.flatten()
                    # compute cos(theta) + m'
                    cos_theta_m = cos_theta[idx_, topk_idx] + self.cur_intertop_margin
                    # clamp so cos cannt be larger than 1.
                    output[idx_, topk_idx] = cos_theta_m.clamp(max=1.0)

            output *= s  # scale up in order to make softmax work
            return output

    def compute_prototype_affinity(self):
        return _cosine_affinity(self.kernel)


class SubCenterArcLossOutput(ArcLossOutput):
    """Sub-Center Additive angular margin softmax (ArcFace) output layer.

    Attributes:
      in_feats: input feature dimension.
      num_classes: number of output classes.
      num_subcenters: number of subcenters.
      cos_scale: cosine scale.
      margin: angular margin.
      margin_warmup_epochs: number of epochs to warm up the margin from 0 to
                            its final value.
      intertop_k: adds negative angular penalty to k largest negative scores.
      intertop_margin: inter-top-k penalty.
    """

    def __init__(
        self,
        in_feats,
        num_classes,
        num_subcenters=2,
        cos_scale=64,
        margin=0.3,
        margin_warmup_epochs=0,
        intertop_k=5,
        intertop_margin=0.0,
    ):
        super().__init__(
            in_feats,
            num_classes * num_subcenters,
            cos_scale,
            margin,
            margin_warmup_epochs,
            intertop_k,
            intertop_margin,
        )
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters
        # this variable counts which subcenter is used more time during training
        # Therefore, which subscenter correspond to the clean label.
        self.register_buffer(
            "subcenter_counts", torch.zeros(num_classes, num_subcenters)
        )

    def __str__(self):
        s = "%s(in_feats=%d, num_classes=%d, num_subcenters=%d, cos_scale=%.2f, margin=%.2f, margin_warmup_epochs=%d, intertop_k=%d, intertop_margin=%f)" % (
            self.__class__.__name__,
            self.in_feats,
            self.num_classes,
            self.num_subcenters,
            self.cos_scale,
            self.margin,
            self.margin_warmup_epochs,
            self.intertop_k,
            self.intertop_margin,
        )
        return s

    def _update_counts(self, y, proto_idx):
        self.subcenter_counts[y, proto_idx] += 1
        # we make counts relative to avoid risk of overflowing the integers
        min_counts, _ = torch.min(self.subcenter_counts, dim=1, keepdim=True)
        self.subcenter_counts -= min_counts

    def forward(self, x, y=None):
        """Computes penalized logits.

        Args:
          x: Input feature tensor with shape = (batch, in_feats).
          y: Ground truth classes. This is required to penalize the logit of
             the true class at training time.

        Returns:
          Logit tensor with shape = (batch, num_classes)
        """
        with amp.autocast(enabled=False):
            s = self.cos_scale
            batch_size = len(x)
            x = _l2_norm(x.float())
            kernel_norm = _l2_norm(self.kernel, axis=0)
            # cos(theta+m)
            cos_theta = torch.mm(x, kernel_norm).float()
            cos_theta, proto_idx = torch.max(
                cos_theta.view(-1, self.num_classes, self.num_subcenters), dim=-1
            )
            cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

            output = (
                cos_theta * 1.0
            )  # a little bit hacky way to prevent in_place operation on cos_theta

            if y is not None and self.training:
                self._update_counts(y, proto_idx)
                cos_theta_2 = torch.pow(cos_theta, 2)
                sin_theta_2 = (1 + 1e-10) - cos_theta_2
                sin_theta = torch.sqrt(sin_theta_2)
                cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

                idx_ = torch.arange(0, batch_size, dtype=torch.long)
                output[idx_, y] = cos_theta_m[idx_, y]
                if self.cur_intertop_margin > 0:
                    # implementation of intertop-K
                    # set positive scores to -inf so they don't appear in the top k
                    cos_aux = cos_theta * 1
                    cos_aux[idx_, y] = -1e10
                    # find topk indices for negative samples
                    topk = torch.topk(cos_aux, k=self.intertop_k, dim=-1, sorted=False)
                    idx_ = (
                        idx_.unsqueeze(-1).expand(batch_size, self.intertop_k).flatten()
                    )
                    topk_idx = topk.indices.flatten()
                    # compute cos(theta-m')
                    cos_theta_m = (
                        cos_theta[idx_, topk_idx] * self.intertop_cos_m
                        + sin_theta[idx_, topk_idx] * self.intertop_sin_m
                    )
                    # take the maximum for the cases where m' is larger than theta to get cos(max(0, theta-m'))
                    output[idx_, topk_idx] = torch.maximum(
                        output[idx_, topk_idx], cos_theta_m
                    )

            output *= s  # scale up in order to make softmax work
            return output

    def get_main_prototype_kernel(self):
        _, idx2 = torch.max(
            self.subcenter_counts, dim=-1
        )  # get indices for the main prototype
        idx1 = torch.arange(self.num_classes)
        kernel = kernel.view(-1, self.num_classes, self.num_subcenters)[:, idx1, idx2]
        return kernel

    def compute_prototype_affinity(self):
        kernel = self.get_main_prototype_kernel()
        return _cosine_affinity(kernel)

    def to_arc_loss(self):
        loss = ArcLossOutput(
            in_feats=self.in_feats,
            num_classes=self.num_classes,
            cos_scale=self.cos_scale,
            margin=self.margin,
            margin_warmup_epochs=self.margin_warmup_epochs,
            intertop_k=self.intertop_k,
            intertop_margin=self.intertop_margin,
        )
        kernel = self.get_main_prototype_kernel()
        loss.kernel.data = kernel
        return loss

    def to_cos_loss(self):
        loss = CosLossOutput(
            in_feats=self.in_feats,
            num_classes=self.num_classes,
            cos_scale=self.cos_scale,
            margin=self.margin,
            margin_warmup_epochs=self.margin_warmup_epochs,
            intertop_k=self.intertop_k,
            intertop_margin=self.intertop_margin,
        )
        kernel = self.get_main_prototype_kernel()
        loss.kernel.data = kernel
        return loss
