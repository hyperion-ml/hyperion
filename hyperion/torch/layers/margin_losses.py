"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F


def _cosine_affinity(kernel: torch.Tensor) -> torch.Tensor:
    """Computes pairwise cosine affinity between class prototypes.

    Args:
      kernel: Prototype matrix with shape ``(in_feats, num_classes)``.

    Returns:
      Cosine affinity matrix with shape ``(num_classes, num_classes)``.
    """
    kernel_norm = F.normalize(kernel, dim=0, eps=1e-10)
    return torch.mm(kernel_norm.transpose(0, 1), kernel_norm)


def _normalized_prototypes(kernel: torch.Tensor) -> torch.Tensor:
    """Returns unit-norm class prototypes as rows.

    Args:
      kernel: Prototype matrix with shape ``(in_feats, num_classes)``.

    Returns:
      Prototype matrix with shape ``(num_classes, in_feats)``.
    """
    return F.normalize(kernel, dim=0, eps=1e-10).transpose(0, 1)


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
        in_feats: int,
        num_classes: int,
        cos_scale: float = 64,
        margin: float = 0.3,
        margin_warmup_epochs: int = 0,
        margin_warmup_steps: int = 0,
        intertop_k: int = 5,
        intertop_margin: float = 0,
    ) -> None:
        """Initializes ArcFace output parameters.

        Args:
          in_feats: Input feature dimension.
          num_classes: Number of output classes.
          cos_scale: Scale factor applied to cosine logits.
          margin: Angular margin.
          margin_warmup_epochs: Number of warmup epochs for margin scheduling.
          margin_warmup_steps: Number of warmup steps for margin scheduling.
          intertop_k: Number of hardest negative classes for InterTopK.
          intertop_margin: InterTopK angular margin.

        Returns:
          None.
        """
        super().__init__()
        self.in_feats = in_feats
        self.num_classes = num_classes
        self.cos_scale = cos_scale
        self.margin = margin
        self.margin_warmup_epochs = margin_warmup_epochs
        self.margin_warmup_steps = margin_warmup_steps
        self.intertop_k = intertop_k
        self.intertop_margin = intertop_margin
        self._update_on_step = margin_warmup_steps > 0
        if margin_warmup_epochs == 0 and margin_warmup_steps == 0:
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

    @property
    def prototypes(self) -> torch.Tensor:
        """Returns the class prototypes.

        Returns:
          Class prototypes with shape ``(num_classes, in_feats)``.
        """
        return _normalized_prototypes(self.kernel)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        s = (
            "%s(in_feats=%d, num_classes=%d, cos_scale=%.2f, margin=%.2f, margin_warmup_epochs=%d, intertop_k=%d, intertop_margin=%f)"
            % (
                self.__class__.__name__,
                self.in_feats,
                self.num_classes,
                self.cos_scale,
                self.margin,
                self.margin_warmup_epochs,
                self.intertop_k,
                self.intertop_margin,
            )
        )
        return s

    def _compute_aux(self) -> None:
        """Updates cached trigonometric values for current margins.

        Returns:
          None.
        """
        self.cos_m = math.cos(self.cur_margin)
        self.sin_m = math.sin(self.cur_margin)
        self.intertop_cos_m = math.cos(self.cur_intertop_margin)
        self.intertop_sin_m = math.sin(self.cur_intertop_margin)

    def _effective_intertop_k(self) -> int:
        """Returns a valid InterTopK value constrained by class count.

        Returns:
          Effective ``k`` used for InterTopK selection.
        """
        return min(self.intertop_k, max(self.num_classes - 1, 0))

    def update_margin(self, step: int) -> None:
        """Updates the value of the margin.

        Args:
          step: value of current epoch or step.
        """
        if self._update_on_step:
            margin_warmup = self.margin_warmup_steps
        else:
            margin_warmup = self.margin_warmup_epochs

        if step < margin_warmup:
            self.cur_margin = self.margin * step / margin_warmup
            self.cur_intertop_margin = self.intertop_margin * step / margin_warmup
        else:
            if self.cur_margin != self.margin:
                self.cur_margin = self.margin
                self.cur_intertop_margin = self.intertop_margin
            else:
                return

        if step % 1000 == 0 or step == margin_warmup or not self._update_on_step:
            logging.info(
                "updating arc-softmax margin=%.2f intertop-margin=%.2f",
                self.cur_margin,
                self.cur_intertop_margin,
            )

        self._compute_aux()

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes penalized logits.

        Args:
          x: input feature tensor with shape = (batch, in_feats).
          y: ground truth classes. This is required to penalize the logit of
             the true class at training time.

        Returns:
          Logit tensor with shape = (batch, num_classes)
        """
        with amp.autocast(enabled=False, device_type=x.device.type):
            s = self.cos_scale
            batch_size = len(x)
            x = F.normalize(x.float(), dim=-1, eps=1e-10)
            kernel_norm = F.normalize(self.kernel, dim=0, eps=1e-10)
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

                idx_ = torch.arange(
                    0, batch_size, dtype=torch.long, device=cos_theta_m.device
                )
                output[idx_, y] = cos_theta_m[idx_, y]
                k = self._effective_intertop_k()
                if self.cur_intertop_margin > 0 and k > 0:
                    # implementation of intertop-K
                    # set positive scores to -inf so they don't appear in the top k
                    cos_aux = cos_theta * 1
                    cos_aux[idx_, y] = -1e10
                    # find topk indices for negative samples
                    topk = torch.topk(cos_aux, k=k, dim=-1, sorted=False)
                    idx_ = idx_.unsqueeze(-1).expand(batch_size, k).flatten()
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

    def compute_prototype_affinity(self) -> torch.Tensor:
        """Computes cosine affinity between current class prototypes.

        Returns:
          Prototype affinity matrix.
        """
        return _cosine_affinity(self.kernel)


class CosLossOutput(nn.Module):
    """Additive margin softmax (CosFace) output layer.

    Attributes:
      in_feats: input feature dimension.
      num_classes: number of output classes.
      cos_scale: cosine scale.
      margin: additive margin.
      margin_warmup_epochs: number of epochs to warm up the margin from 0 to
                            its final value.
      margin_warmup_steps: number of steps to warm up the margin from 0 to
                           its final value.
      intertop_k: adds negative penalty to k largest negative scores.
      intertop_margin: inter-top-k penalty.
    """

    def __init__(
        self,
        in_feats: int,
        num_classes: int,
        cos_scale: float = 64,
        margin: float = 0.3,
        margin_warmup_epochs: int = 0,
        margin_warmup_steps: int = 0,
        intertop_k: int = 5,
        intertop_margin: float = 0.0,
    ) -> None:
        """Initializes CosFace output parameters.

        Args:
          in_feats: Input feature dimension.
          num_classes: Number of output classes.
          cos_scale: Scale factor applied to cosine logits.
          margin: Additive cosine margin.
          margin_warmup_epochs: Number of warmup epochs for margin scheduling.
          margin_warmup_steps: Number of warmup steps for margin scheduling.
          intertop_k: Number of hardest negative classes for InterTopK.
          intertop_margin: InterTopK additive margin.

        Returns:
          None.
        """
        super().__init__()
        self.in_feats = in_feats
        self.num_classes = num_classes
        self.cos_scale = cos_scale
        self.margin = margin
        self.margin_warmup_epochs = margin_warmup_epochs
        self.margin_warmup_steps = margin_warmup_steps
        self.intertop_k = intertop_k
        self.intertop_margin = intertop_margin
        self._update_on_step = margin_warmup_steps > 0
        if margin_warmup_epochs == 0 and margin_warmup_steps == 0:
            self.cur_margin = margin
            self.cur_intertop_margin = intertop_margin
        else:
            self.cur_margin = 0
            self.cur_intertop_margin = 0

        self.kernel = nn.Parameter(torch.Tensor(in_feats, num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    @property
    def prototypes(self) -> torch.Tensor:
        """Returns the class prototypes.

        Returns:
          Class prototypes with shape ``(num_classes, in_feats)``.
        """
        return _normalized_prototypes(self.kernel)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        s = (
            "%s(in_feats=%d, num_classes=%d, cos_scale=%.2f, margin=%.2f, margin_warmup_epochs=%d, intertop_k=%d, intertop_margin=%f)"
            % (
                self.__class__.__name__,
                self.in_feats,
                self.num_classes,
                self.cos_scale,
                self.margin,
                self.margin_warmup_epochs,
                self.intertop_k,
                self.intertop_margin,
            )
        )
        return s

    def update_margin(self, step: int) -> None:
        """Updates the value of the margin.

        Args:
          step: value of current epoch or step.
        """
        if self._update_on_step:
            margin_warmup = self.margin_warmup_steps
        else:
            margin_warmup = self.margin_warmup_epochs

        if step < margin_warmup:
            self.cur_margin = self.margin * step / margin_warmup
            self.cur_intertop_margin = self.intertop_margin * step / margin_warmup
        else:
            if self.cur_margin != self.margin:
                self.cur_margin = self.margin
                self.cur_intertop_margin = self.intertop_margin
            else:
                return

        if not self._update_on_step or step % 1000 == 0 or step == margin_warmup:
            logging.info(
                "updating cos-softmax margin=%.2f intertop-margin=%.2f",
                self.cur_margin,
                self.cur_intertop_margin,
            )

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes penalized logits.

        Args:
          x: input feature tensor with shape = (batch, in_feats).
          y: ground truth classes. This is required to penalize the logit of
             the true class at training time.

        Returns:
          Logit tensor with shape = (batch, num_classes)
        """
        with amp.autocast(enabled=False, device_type=x.device.type):
            s = self.cos_scale
            x = F.normalize(x.float(), dim=-1, eps=1e-10)
            batch_size = len(x)
            kernel_norm = F.normalize(self.kernel, dim=0, eps=1e-10)
            # cos(theta+m)
            cos_theta = torch.mm(x, kernel_norm).float()
            cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

            output = (
                cos_theta * 1.0
            )  # a little bit hacky way to prevent in_place operation on cos_theta
            if y is not None and self.training:
                cos_theta_m = cos_theta - self.cur_margin
                idx_ = torch.arange(
                    0, batch_size, dtype=torch.long, device=cos_theta_m.device
                )
                output[idx_, y] = cos_theta_m[idx_, y]
                k = min(self.intertop_k, max(self.num_classes - 1, 0))
                if self.cur_intertop_margin > 0 and k > 0:
                    # implementation of intertop-K
                    # set positive scores to -inf so they don't appear in the top k
                    cos_aux = cos_theta * 1
                    cos_aux[idx_, y] = -1e10
                    # find topk indices for negative samples
                    topk = torch.topk(cos_aux, k=k, dim=-1, sorted=False)
                    idx_ = idx_.unsqueeze(-1).expand(batch_size, k).flatten()
                    topk_idx = topk.indices.flatten()
                    # compute cos(theta) + m'
                    cos_theta_m = cos_theta[idx_, topk_idx] + self.cur_intertop_margin
                    # clamp so cos cannt be larger than 1.
                    output[idx_, topk_idx] = cos_theta_m.clamp(max=1.0)

            output *= s  # scale up in order to make softmax work
            return output

    def compute_prototype_affinity(self) -> torch.Tensor:
        """Computes cosine affinity between current class prototypes.

        Returns:
          Prototype affinity matrix.
        """
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
      margin_warmup_steps: number of steps to warm up the margin from 0 to its
                           final value.
      intertop_k: adds negative angular penalty to k largest negative scores.
      intertop_margin: inter-top-k penalty.
    """

    def __init__(
        self,
        in_feats: int,
        num_classes: int,
        num_subcenters: int = 2,
        cos_scale: float = 64,
        margin: float = 0.3,
        margin_warmup_epochs: int = 0,
        margin_warmup_steps: int = 0,
        intertop_k: int = 5,
        intertop_margin: float = 0.0,
    ) -> None:
        """Initializes Sub-Center ArcFace output parameters.

        Args:
          in_feats: Input feature dimension.
          num_classes: Number of output classes.
          num_subcenters: Number of prototypes per class.
          cos_scale: Scale factor applied to cosine logits.
          margin: Angular margin.
          margin_warmup_epochs: Number of warmup epochs for margin scheduling.
          margin_warmup_steps: Number of warmup steps for margin scheduling.
          intertop_k: Number of hardest negative classes for InterTopK.
          intertop_margin: InterTopK angular margin.

        Returns:
          None.
        """
        super().__init__(
            in_feats=in_feats,
            num_classes=num_classes * num_subcenters,
            cos_scale=cos_scale,
            margin=margin,
            margin_warmup_epochs=margin_warmup_epochs,
            margin_warmup_steps=margin_warmup_steps,
            intertop_k=intertop_k,
            intertop_margin=intertop_margin,
        )
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters
        # this variable counts which subcenter is used more time during training
        # Therefore, which subscenter correspond to the clean label.
        self.register_buffer(
            "subcenter_counts", torch.zeros(num_classes, num_subcenters)
        )

    @property
    def prototypes(self) -> torch.Tensor:
        """Returns the class prototypes.

        Returns:
          Class prototypes with shape ``(num_classes, in_feats)``.
        """
        return _normalized_prototypes(self.get_main_prototype_kernel())

    def __str__(self) -> str:
        s = (
            "%s(in_feats=%d, num_classes=%d, num_subcenters=%d, cos_scale=%.2f, margin=%.2f, margin_warmup_epochs=%d, intertop_k=%d, intertop_margin=%f)"
            % (
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
        )
        return s

    def _update_counts(self, y: torch.Tensor, proto_idx: torch.Tensor) -> None:
        """Updates class-wise usage counts for selected subcenters.

        Args:
          y: Ground-truth class indices.
          proto_idx: Per-sample selected subcenter indices.

        Returns:
          None.
        """
        idx1 = torch.arange(y.size(0), device=y.device, dtype=torch.long)
        proto_idx = proto_idx[idx1, y]
        self.subcenter_counts.index_put_(
            (y, proto_idx),
            torch.ones_like(y, dtype=self.subcenter_counts.dtype),
            accumulate=True,
        )
        # we make counts relative to avoid risk of overflowing the integers
        min_counts, _ = torch.min(self.subcenter_counts, dim=1, keepdim=True)
        self.subcenter_counts -= min_counts

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes penalized logits.

        Args:
          x: Input feature tensor with shape = (batch, in_feats).
          y: Ground truth classes. This is required to penalize the logit of
             the true class at training time.

        Returns:
          Logit tensor with shape = (batch, num_classes)
        """
        with amp.autocast(enabled=False, device_type=x.device.type):
            s = self.cos_scale
            batch_size = len(x)
            x = F.normalize(x.float(), dim=-1, eps=1e-10)
            kernel_norm = F.normalize(self.kernel, dim=0, eps=1e-10)
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

                idx_ = torch.arange(
                    0, batch_size, dtype=torch.long, device=cos_theta_m.device
                )
                output[idx_, y] = cos_theta_m[idx_, y]
                k = self._effective_intertop_k()
                if self.cur_intertop_margin > 0 and k > 0:
                    # implementation of intertop-K
                    # set positive scores to -inf so they don't appear in the top k
                    cos_aux = cos_theta * 1
                    cos_aux[idx_, y] = -1e10
                    # find topk indices for negative samples
                    topk = torch.topk(cos_aux, k=k, dim=-1, sorted=False)
                    idx_ = idx_.unsqueeze(-1).expand(batch_size, k).flatten()
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

    def get_main_prototype_kernel(self) -> torch.Tensor:
        """Returns one representative prototype per class.

        Returns:
          Kernel tensor containing the most-used subcenter of each class.
        """
        _, idx2 = torch.max(
            self.subcenter_counts, dim=-1
        )  # get indices for the main prototype
        idx1 = torch.arange(
            self.num_classes, device=self.kernel.device, dtype=torch.long
        )
        kernel = self.kernel.view(-1, self.num_classes, self.num_subcenters)[
            :, idx1, idx2
        ]
        return kernel

    def compute_prototype_affinity(self) -> torch.Tensor:
        """Computes cosine affinity between main class prototypes.

        Returns:
          Prototype affinity matrix.
        """
        kernel = self.get_main_prototype_kernel()
        return _cosine_affinity(kernel)

    def to_arc_loss(self) -> ArcLossOutput:
        """Converts this module into a standard ``ArcLossOutput``.

        Returns:
          ArcFace loss layer initialized from main prototypes.
        """
        loss = ArcLossOutput(
            in_feats=self.in_feats,
            num_classes=self.num_classes,
            cos_scale=self.cos_scale,
            margin=self.margin,
            margin_warmup_epochs=self.margin_warmup_epochs,
            margin_warmup_steps=self.margin_warmup_steps,
            intertop_k=self.intertop_k,
            intertop_margin=self.intertop_margin,
        )
        kernel = self.get_main_prototype_kernel()
        loss.kernel.data = kernel
        return loss

    def to_cos_loss(self) -> CosLossOutput:
        """Converts this module into a standard ``CosLossOutput``.

        Returns:
          CosFace loss layer initialized from main prototypes.
        """
        loss = CosLossOutput(
            in_feats=self.in_feats,
            num_classes=self.num_classes,
            cos_scale=self.cos_scale,
            margin=self.margin,
            margin_warmup_epochs=self.margin_warmup_epochs,
            margin_warmup_steps=self.margin_warmup_steps,
            intertop_k=self.intertop_k,
            intertop_margin=self.intertop_margin,
        )
        kernel = self.get_main_prototype_kernel()
        loss.kernel.data = kernel
        return loss
