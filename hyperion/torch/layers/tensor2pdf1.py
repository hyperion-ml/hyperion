"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn
import torch.distributions as pdf


class Tensor2PDF(nn.Module):
    """Base class for layers that create a prob distribution
    from an input tensor
    """

    def __init__(self):
        super(Tensor2PDF, self).__init__()
        self.tensor2pdfparam_factor = 1


class Tensor2NormalICov(Tensor2PDF):
    """Transforms a Tensor into Normal distribution with identitiy variance"""

    def __init__(self):
        super(Tensor2NormalGlobDiagCov, self).__init__()

    def forward(self, loc, prior=None):
        scale = torch.ones_like(loc)
        return pdf.normal.Normal(loc, scale)


class Tensor2NormalGlobDiagCov(Tensor2PDF):
    """Transforms a Tensor into Normal distribution

    Input tensor will be the mean of the distribution and
    the standard deviation is a global trainable parameter.
    """

    def __init__(self, shape):
        super(Tensor2NormalGlobDiagCov, self).__init__()
        self.logvar = nn.Parameter(torch.zeros(shape))

    def forward(self, loc, prior=None):
        # stddev
        scale = torch.exp(0.5 * self.logvar)
        if prior is not None:
            # the variance of the posterior should be smaller than
            # the variance of the prior
            scale = torch.min(scale, prior.scale)

        return pdf.normal.Normal(loc, scale)


class Tensor2NormalDiagCov(Tensor2PDF):
    """Transforms a Tensor into Normal distribution

    Applies two linear transformation to the tensors to
    obtain the mean and the log-variance.
    """

    def __init__(self):
        super(Tensor2NormalDiagCov, self).__init__()
        self.tensor2pdfparam_factor = 2

    def forward(self, x, prior=None):
        # stddev
        loc, logvar = x.chunk(2, dim=1)
        logvar = self.logvar(x)
        scale = torch.exp(0.5 * logvar)
        if prior is not None:
            # the variance of the posterior should be smaller than
            # the variance of the prior
            scale = torch.min(scale, prior.scale)

        return pdf.normal.Normal(loc, scale)


# class Tensor2NormalDiagCovLin(Tensor2PDF):
#     """Transforms a Tensor into Normal distribution

#        Applies two linear transformation to the tensors to
#        obtain the mean and the log-variance.
#     """

#     def __init__(self, in_shape, out_shape):
#         super(Tensor2NormalDiagCovLin, self).__init__()
#         ndim = len(in_shape)
#         assert ndim == len(out_shape)
#         if ndim == 2:
#             self.loc = nn.Linear(in_shape[-1], out_shape[-1])
#             self.logvar = nn.Linear(in_shape[-1], out_shape[-1])
#         elif ndim == 3:
#             self.loc = nn.Conv1d(in_shape[-1], out_shape[-1], kernel_size=1)
#             self.logvar = nn.Conv1d(in_shape[-1], out_shape[-1], kernel_size=1)
#         elif ndim == 4:
#             self.loc = nn.Conv2d(in_shape[-1], out_shape[-1], kernel_size=1)
#             self.logvar = nn.Conv2d(in_shape[-1], out_shape[-1], kernel_size=1)
#         elif ndim == 5:
#             self.loc = nn.Conv3d(in_shape[-1], out_shape[-1], kernel_size=1)
#             self.logvar = nn.Conv3d(in_shape[-1], out_shape[-1], kernel_size=1)
#         else:
#             raise ValueError('ndim=%d is not supported' % ndim)


#     def forward(self, x, prior=None):
#         # stddev
#         loc = self.loc(x)
#         logvar = self.logvar(x)
#         scale = torch.exp(0.5*logvar)
#         if prior is not None:
#             # the variance of the posterior should be smaller than
#             # the variance of the prior
#             scale = torch.min(scale, prior.scale)

#         return pdf.normal.Normal(loc, scale)
