"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributions as pdf
import torch.nn as nn
import torch.nn.functional as nnf


class Tensor2PDF(nn.Module):
    """Base class for layers that map tensors to probability distributions.

    Args:
        pdf_feats: Feature dimension of the output distribution parameters.
        project: If ``True``, create a learnable projection from input features.
        in_feats: Input feature dimension used when ``project=True``.
        in_dim: Number of input tensor dimensions used when ``project=True``.
    """

    def __init__(
        self,
        pdf_feats: int,
        project: bool = True,
        in_feats: Optional[int] = None,
        in_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.pdf_feats = pdf_feats
        self.project = project
        if project:
            assert (
                in_feats is not None
            ), "input channels must be given to make the projection"
            assert (
                in_dim is not None
            ), "input tensor dim must be given to make the projection"

        self.in_feats = in_feats
        self.in_dim = in_dim

    def _make_proj(self, in_feats: int, out_feats: int, ndims: int) -> nn.Module:
        """Create a 1x1 projection matching the input tensor rank."""
        if ndims == 2:
            return nn.Linear(in_feats, out_feats)
        if ndims == 3:
            return nn.Conv1d(in_feats, out_feats, kernel_size=1)
        if ndims == 4:
            return nn.Conv2d(in_feats, out_feats, kernel_size=1)
        if ndims == 5:
            return nn.Conv3d(in_feats, out_feats, kernel_size=1)
        raise ValueError("ndim=%d is not supported" % ndims)


class Tensor2NormalICov(Tensor2PDF):
    """Map a tensor to a Normal distribution with identity covariance."""

    def __init__(
        self,
        pdf_feats: int,
        project: bool = True,
        in_feats: Optional[int] = None,
        in_dim: Optional[int] = None,
    ) -> None:
        super().__init__(pdf_feats, project=project, in_feats=in_feats, in_dim=in_dim)

        if self.project:
            self._proj = self._make_proj(self.in_feats, self.pdf_feats, self.in_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        prior: Optional[pdf.normal.Normal] = None,
        squeeze_dim: Optional[int] = None,
    ) -> pdf.normal.Normal:
        """Create a Normal posterior from ``inputs``.

        Args:
            inputs: Input tensor containing posterior means.
            prior: Unused, present for interface compatibility.
            squeeze_dim: Optional dimension to squeeze on output parameters.

        Returns:
            Normal distribution with mean ``inputs`` and unit scale.
        """
        if self.project:
            inputs = self._proj(inputs)

        loc = inputs
        scale = torch.ones_like(inputs)
        if squeeze_dim is not None:
            loc = loc.squeeze(dim=squeeze_dim)
            scale = scale.squeeze(dim=squeeze_dim)

        return pdf.normal.Normal(loc, scale)


class Tensor2NormalGlobDiagCov(Tensor2PDF):
    """Map a tensor to a Normal distribution with global trainable diagonal covariance."""

    def __init__(
        self,
        pdf_feats: int,
        project: bool = True,
        in_feats: Optional[int] = None,
        in_dim: Optional[int] = None,
    ) -> None:
        super().__init__(pdf_feats, project=project, in_feats=in_feats, in_dim=in_dim)

        if self.project:
            self._proj = self._make_proj(self.in_feats, self.pdf_feats, self.in_dim)

        pdf_shape = [1] * self.in_dim
        pdf_shape[1] = pdf_feats
        pdf_shape = tuple(pdf_shape)

        self.logvar = nn.Parameter(torch.zeros(pdf_shape))

    def forward(
        self,
        inputs: torch.Tensor,
        prior: Optional[pdf.normal.Normal] = None,
        squeeze_dim: Optional[int] = None,
    ) -> pdf.normal.Normal:
        """Create a Normal posterior from ``inputs``.

        Args:
            inputs: Input tensor containing posterior means.
            prior: Optional prior used to upper-bound posterior scale.
            squeeze_dim: Optional dimension to squeeze on output parameters.

        Returns:
            Normal distribution with inferred mean and global trainable scale.
        """
        if self.project:
            inputs = self._proj(inputs)

        loc = inputs
        scale = torch.exp(0.5 * self.logvar)
        if prior is not None:
            # Force posterior variance not to exceed prior variance.
            scale = torch.min(scale, prior.scale)

        if squeeze_dim is not None:
            loc = loc.squeeze(dim=squeeze_dim)
            scale = scale.squeeze(dim=squeeze_dim)

        return pdf.normal.Normal(loc, scale)


class Tensor2NormalDiagCov(Tensor2PDF):
    """Map a tensor to a Normal distribution with per-sample diagonal covariance."""

    def __init__(
        self,
        pdf_feats: int,
        project: bool = True,
        in_feats: Optional[int] = None,
        in_dim: Optional[int] = None,
    ) -> None:
        super().__init__(pdf_feats, project=project, in_feats=in_feats, in_dim=in_dim)

        if self.project:
            self._proj = self._make_proj(self.in_feats, self.pdf_feats * 2, self.in_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        prior: Optional[pdf.normal.Normal] = None,
        squeeze_dim: Optional[int] = None,
    ) -> pdf.normal.Normal:
        """Create a Normal posterior from ``inputs``.

        Args:
            inputs: Input tensor containing concatenated mean/log-variance.
            prior: Optional prior used to upper-bound posterior scale.
            squeeze_dim: Optional dimension to squeeze on output parameters.

        Returns:
            Normal distribution with learned mean and diagonal scale.
        """
        if self.project:
            inputs = self._proj(inputs)

        loc, logvar = inputs.chunk(2, dim=1)
        scale = torch.exp(0.5 * logvar)

        if prior is not None:
            # Force posterior variance not to exceed prior variance.
            scale = torch.min(scale, prior.scale)

        if squeeze_dim is not None:
            loc = loc.squeeze(dim=squeeze_dim)
            scale = scale.squeeze(dim=squeeze_dim)

        return pdf.normal.Normal(loc, scale)


class Tensor2BayNormalICovGivenNormalPrior(Tensor2PDF):
    """Bayesian interpolation between ML mean and Normal prior with identity covariance."""

    def __init__(
        self,
        pdf_feats: int,
        project: bool = True,
        in_feats: Optional[int] = None,
        in_dim: Optional[int] = None,
    ) -> None:
        super().__init__(pdf_feats, project=project, in_feats=in_feats, in_dim=in_dim)

        if self.project:
            self._proj = self._make_proj(self.in_feats, self.pdf_feats, self.in_dim)

        self._alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        inputs: torch.Tensor,
        prior: Optional[pdf.normal.Normal] = None,
        squeeze_dim: Optional[int] = None,
    ) -> pdf.normal.Normal:
        """Create a Normal posterior from ``inputs`` and optional prior.

        Args:
            inputs: Input tensor containing ML means.
            prior: Optional Normal prior for Bayesian mean interpolation.
            squeeze_dim: Optional dimension to squeeze on output parameters.

        Returns:
            Normal distribution with interpolated mean and unit scale.
        """
        if self.project:
            inputs = self._proj(inputs)

        loc = inputs
        scale = torch.ones_like(inputs)
        if prior is not None:
            alpha = nnf.sigmoid(self._alpha)
            loc = alpha * loc + (1 - alpha) * prior.loc

        if squeeze_dim is not None:
            loc = loc.squeeze(dim=squeeze_dim)
            scale = scale.squeeze(dim=squeeze_dim)

        return pdf.normal.Normal(loc, scale)


class Tensor2BayNormalGlobDiagCovGivenNormalPrior(Tensor2PDF):
    """Bayesian interpolation with global trainable diagonal covariance and Normal prior."""

    def __init__(
        self,
        pdf_feats: int,
        project: bool = True,
        in_feats: Optional[int] = None,
        in_dim: Optional[int] = None,
    ) -> None:
        super().__init__(pdf_feats, project=project, in_feats=in_feats, in_dim=in_dim)

        if self.project:
            self._proj = self._make_proj(self.in_feats, self.pdf_feats, self.in_dim)

        pdf_shape = [1] * self.in_dim
        pdf_shape[1] = pdf_feats
        pdf_shape = tuple(pdf_shape)

        self.logvar = nn.Parameter(torch.zeros(pdf_shape))

        self._alpha = nn.Parameter(torch.zeros(1))
        self._beta = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        inputs: torch.Tensor,
        prior: Optional[pdf.normal.Normal] = None,
        squeeze_dim: Optional[int] = None,
    ) -> pdf.normal.Normal:
        """Create a Normal posterior from ``inputs`` and optional prior.

        Args:
            inputs: Input tensor containing ML means.
            prior: Optional Normal prior used for MAP interpolation.
            squeeze_dim: Optional dimension to squeeze on output parameters.

        Returns:
            Normal distribution with MAP-updated parameters.
        """
        if self.project:
            inputs = self._proj(inputs)

        loc = inputs
        scale = torch.exp(0.5 * self.logvar)

        if prior is not None:
            # MAP estimation for Gaussian mean and variance.
            alpha = nnf.sigmoid(self._alpha)
            beta = nnf.sigmoid(self._beta)
            delta_loc = loc - prior.loc
            loc = alpha * loc + (1 - alpha) * prior.loc
            var = (
                beta * scale**2
                + (1 - beta) * prior.scale**2
                + beta * (1 - alpha) * delta_loc**2
            )
            scale = torch.sqrt(var)

        if squeeze_dim is not None:
            loc = loc.squeeze(dim=squeeze_dim)
            scale = scale.squeeze(dim=squeeze_dim)

        return pdf.normal.Normal(inputs, scale)


class Tensor2BayNormalDiagCovGivenNormalPrior(Tensor2PDF):
    """Bayesian interpolation with per-sample diagonal covariance and Normal prior."""

    def __init__(
        self,
        pdf_feats: int,
        project: bool = True,
        in_feats: Optional[int] = None,
        in_dim: Optional[int] = None,
    ) -> None:
        super().__init__(pdf_feats, project=project, in_feats=in_feats, in_dim=in_dim)

        if self.project:
            self._proj = self._make_proj(self.in_feats, self.pdf_feats * 2, self.in_dim)

        self._alpha = nn.Parameter(torch.zeros(1))
        self._beta = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        inputs: torch.Tensor,
        prior: Optional[pdf.normal.Normal] = None,
        squeeze_dim: Optional[int] = None,
    ) -> pdf.normal.Normal:
        """Create a Normal posterior from ``inputs`` and optional prior.

        Args:
            inputs: Input tensor containing concatenated ML mean/log-variance.
            prior: Optional Normal prior used for MAP interpolation.
            squeeze_dim: Optional dimension to squeeze on output parameters.

        Returns:
            Normal distribution with MAP-updated parameters.
        """
        if self.project:
            inputs = self._proj(inputs)

        loc, logvar = inputs.chunk(2, dim=1)
        scale = torch.exp(0.5 * logvar)
        if prior is not None:
            # MAP estimation for Gaussian mean and variance.
            alpha = nnf.sigmoid(self._alpha)
            beta = nnf.sigmoid(self._beta)
            delta_loc = loc - prior.loc
            loc = alpha * loc + (1 - alpha) * prior.loc
            var = (
                beta * scale**2
                + (1 - beta) * prior.scale**2
                + beta * (1 - alpha) * delta_loc**2
            )
            scale = torch.sqrt(var)

        if squeeze_dim is not None:
            loc = loc.squeeze(dim=squeeze_dim)
            scale = scale.squeeze(dim=squeeze_dim)

        return pdf.normal.Normal(loc, scale)
