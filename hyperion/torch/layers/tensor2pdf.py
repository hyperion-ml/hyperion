"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.distributions as pdf


class Tensor2PDF(nn.Module):
    """Base class for layers that create a prob distribution
    from an input tensor

    Attributes:
      pdf_feats: Feature dimension of the probability distribution.
      project:   If True, it applies a projection to the input tensor.
      in_feats:  Feature dimension of the input tensor.
      in_dim:    Number of dimensions of the input tensor.
    """

    def __init__(self, pdf_feats, project=True, in_feats=None, in_dim=None):
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

    def _make_proj(self, in_feats, out_feats, ndims):
        if ndims == 2:
            return nn.Linear(in_feats, out_feats)
        elif ndims == 3:
            return nn.Conv1d(in_feats, out_feats, kernel_size=1)
        elif ndims == 4:
            return nn.Conv2d(in_feats, out_feats, kernel_size=1)
        elif ndims == 5:
            return nn.Conv3d(in_feats, out_feats, kernel_size=1)
        else:
            raise ValueError("ndim=%d is not supported" % ndims)


class Tensor2NormalICov(Tensor2PDF):
    """Transforms a Tensor into Normal distribution with identitiy variance

    Attributes:
      pdf_feats: Feature dimension of the probability distribution.
      project:   If True, it applies a projection to the input tensor.
      in_feats:  Feature dimension of the input tensor.
      in_dim:    Number of dimensions of the input tensor.
    """

    def __init__(self, pdf_feats, project=True, in_feats=None, in_dim=None):
        super().__init__(pdf_feats, project=project, in_feats=in_feats, in_dim=in_dim)

        if self.project:
            self._proj = self._make_proj(self.in_feats, self.pdf_feats, self.in_dim)

    def forward(self, inputs, prior=None, squeeze_dim=None):
        """Creates a Normal distribution from input tensor.

        Args:
          inputs: Input tensor.
          prior:  Not used.
          squeeze_dim: Squeezes pdf parameters dimensions.

        Returns:
          torch.distributions.normal.Normal object.
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
    """Transforms a Tensor into Normal distribution

    Input tensor will be the mean of the distribution and
    the standard deviation is a global trainable parameter.

    Attributes:
      pdf_feats: Feature dimension of the probability distribution.
      project:   If True, it applies a projection to the input tensor.
      in_feats:  Feature dimension of the input tensor.
      in_dim:    Number of dimensions of the input tensor.
    """

    def __init__(self, pdf_feats, project=True, in_feats=None, in_dim=None):
        super().__init__(pdf_feats, project=project, in_feats=in_feats, in_dim=in_dim)

        if self.project:
            self._proj = self._make_proj(self.in_feats, self.pdf_feats, self.in_dim)

        pdf_shape = [1] * self.in_dim
        pdf_shape[1] = pdf_feats
        pdf_shape = tuple(pdf_shape)

        self.logvar = nn.Parameter(torch.zeros(pdf_shape))

    def forward(self, inputs, prior=None, squeeze_dim=None):
        """Creates a Normal distribution from input tensor.

        Args:
          inputs: Input tensor.
          Args:
          inputs: Input tensor.
          prior:  prior pdf object.
          squeeze_dim: Squeezes pdf parameters dimensions.

        Returns:
          torch.distributions.normal.Normal object.
        """
        if self.project:
            inputs = self._proj(inputs)

        # stddev
        loc = inputs
        scale = torch.exp(0.5 * self.logvar)
        if prior is not None:
            # we force the variance of the posterior smaller than
            # the variance of the prior
            scale = torch.min(scale, prior.scale)

        if squeeze_dim is not None:
            loc = loc.squeeze(dim=squeeze_dim)
            scale = scale.squeeze(dim=squeeze_dim)

        return pdf.normal.Normal(loc, scale)


class Tensor2NormalDiagCov(Tensor2PDF):
    """Transforms a Tensor into Normal distribution

    Applies two linear transformation to the tensors to
    obtain the mean and the log-variance.

    Attributes:
      pdf_feats: Feature dimension of the probability distribution.
      project:   If True, it applies a projection to the input tensor.
      in_feats:  Feature dimension of the input tensor.
      in_dim:    Number of dimensions of the input tensor.
    """

    def __init__(self, pdf_feats, project=True, in_feats=None, in_dim=None):
        super().__init__(pdf_feats, project=project, in_feats=in_feats, in_dim=in_dim)

        if self.project:
            self._proj = self._make_proj(self.in_feats, self.pdf_feats * 2, self.in_dim)

    def forward(self, inputs, prior=None, squeeze_dim=None):
        """Creates a Normal distribution from input tensor.

        Args:
          inputs: Input tensor.
          Args:
          inputs: Input tensor.
          prior:  prior pdf object.
          squeeze_dim: Squeezes pdf parameters dimensions.

        Returns:
          torch.distributions.normal.Normal object.
        """
        if self.project:
            inputs = self._proj(inputs)

        loc, logvar = inputs.chunk(2, dim=1)
        scale = torch.exp(0.5 * logvar)

        if prior is not None:
            # we force the variance of the posterior smaller than
            # the variance of the prior
            scale = torch.min(scale, prior.scale)

        if squeeze_dim is not None:
            loc = loc.squeeze(dim=squeeze_dim)
            scale = scale.squeeze(dim=squeeze_dim)

        return pdf.normal.Normal(loc, scale)


class Tensor2BayNormalICovGivenNormalPrior(Tensor2PDF):
    """Transforms a Tensor into Normal distribution with identitiy variance

    Uses Bayesian interpolation between Gaussian prior and Maximum Likelihood estimation.

    Attributes:
      pdf_feats: Feature dimension of the probability distribution.
      project:   If True, it applies a projection to the input tensor.
      in_feats:  Feature dimension of the input tensor.
      in_dim:    Number of dimensions of the input tensor.
    """

    def __init__(self, pdf_feats, project=True, in_feats=None, in_dim=None):
        super().__init__(pdf_feats, project=project, in_feats=in_feats, in_dim=in_dim)

        if self.project:
            self._proj = self._make_proj(self.in_feats, self.pdf_feats, self.in_dim)

        # interpolation factors between prior and ML estimation
        self._alpha = nn.Parameter(torch.zeros(1))

    def forward(self, inputs, prior=None, squeeze_dim=None):
        """Creates a Normal distribution from input tensor.

        Args:
          inputs: Input tensor.
          Args:
          inputs: Input tensor.
          prior:  prior pdf object.
          squeeze_dim: Squeezes pdf parameters dimensions.

        Returns:
          torch.distributions.normal.Normal object.
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
    """Transforms a Tensor into Normal distribution

    Input tensor will be the ML mean of the distribution and
    the ML standard deviation is a global trainable parameter.

    Uses Bayesian interpolation between Gaussian prior and Maximum Likelihood estimation.

    Attributes:
      pdf_feats: Feature dimension of the probability distribution.
      project:   If True, it applies a projection to the input tensor.
      in_feats:  Feature dimension of the input tensor.
      in_dim:    Number of dimensions of the input tensor.
    """

    def __init__(self, pdf_feats, project=True, in_feats=None, in_dim=None):
        super().__init__(pdf_feats, project=project, in_feats=in_feats, in_dim=in_dim)

        if self.project:
            self._proj = self._make_proj(self.in_feats, self.pdf_feats, self.in_dim)

        pdf_shape = [1] * self.in_dim
        pdf_shape[1] = pdf_feats
        pdf_shape = tuple(pdf_shape)

        self.logvar = nn.Parameter(torch.zeros(pdf_shape))

        # interpolation factors between prior and ML estimation
        self._alpha = nn.Parameter(torch.zeros(1))
        self._beta = nn.Parameter(torch.zeros(1))

    def forward(self, inputs, prior=None, squeeze_dim=None):
        """Creates a Normal distribution from input tensor.

        Args:
          inputs: Input tensor.
          Args:
          inputs: Input tensor.
          prior:  prior pdf object.
          squeeze_dim: Squeezes pdf parameters dimensions.

        Returns:
          torch.distributions.normal.Normal object.
        """
        if self.project:
            inputs = self._proj(inputs)

        # stddev
        loc = inputs
        scale = torch.exp(0.5 * self.logvar)

        if prior is not None:
            # MAP estimation of Gaussian mean and var
            # Eq. from Bishop2006 (10.60-10.63)
            # were we renamed
            # alpha <- N/(beta_0+N)
            # beta <- N/(nu_0+N)
            # where beta_0 and nu_0 are MAP relevance factor for mean and var
            alpha = nnf.sigmoid(self._alpha)
            beta = nnf.sigmoid(self._beta)
            delta_loc = loc - prior.loc
            loc = alpha * loc + (1 - alpha) * prior.loc
            var = (
                beta * scale ** 2
                + (1 - beta) * prior.scale ** 2
                + beta * (1 - alpha) * delta_loc ** 2
            )
            scale = torch.sqrt(var)

        if squeeze_dim is not None:
            loc = loc.squeeze(dim=squeeze_dim)
            scale = scale.squeeze(dim=squeeze_dim)

        return pdf.normal.Normal(inputs, scale)


class Tensor2BayNormalDiagCovGivenNormalPrior(Tensor2PDF):
    """Transforms a Tensor into Normal distribution

    Applies two linear transformation to the tensors to
    obtain the maximum likelihood mean and the log-variance.

    Uses Bayesian interpolation between Gaussian prior and Maximum Likelihood estimation.

    Attributes:
      pdf_feats: Feature dimension of the probability distribution.
      project:   If True, it applies a projection to the input tensor.
      in_feats:  Feature dimension of the input tensor.
      in_dim:    Number of dimensions of the input tensor.
    """

    def __init__(self, pdf_feats, project=True, in_feats=None, in_dim=None):
        super().__init__(pdf_feats, project=project, in_feats=in_feats, in_dim=in_dim)

        if self.project:
            self._proj = self._make_proj(self.in_feats, self.pdf_feats * 2, self.in_dim)

        # interpolation factors between prior and ML estimation
        self._alpha = nn.Parameter(torch.zeros(1))
        self._beta = nn.Parameter(torch.zeros(1))

    def forward(self, inputs, prior=None, squeeze_dim=None):
        """Creates a Normal distribution from input tensor.

        Args:
          inputs: Input tensor.
          Args:
          inputs: Input tensor.
          prior:  prior pdf object.
          squeeze_dim: Squeezes pdf parameters dimensions.

        Returns:
          torch.distributions.normal.Normal object.
        """
        if self.project:
            inputs = self._proj(inputs)

        loc, logvar = inputs.chunk(2, dim=1)
        scale = torch.exp(0.5 * logvar)
        if prior is not None:
            # MAP estimation of Gaussian mean and var
            # Eq. from Bishop2006 (10.60-10.63)
            # were we renamed
            # alpha <- N/(beta_0+N)
            # beta <- N/(nu_0+N)
            # where beta_0 and nu_0 are MAP relevance factor for mean and var
            alpha = nnf.sigmoid(self._alpha)
            beta = nnf.sigmoid(self._beta)
            delta_loc = loc - prior.loc
            loc = alpha * loc + (1 - alpha) * prior.loc
            var = (
                beta * scale ** 2
                + (1 - beta) * prior.scale ** 2
                + beta * (1 - alpha) * delta_loc ** 2
            )
            scale = torch.sqrt(var)

        if squeeze_dim is not None:
            loc = loc.squeeze(dim=squeeze_dim)
            scale = scale.squeeze(dim=squeeze_dim)

        return pdf.normal.Normal(loc, scale)
