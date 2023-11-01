"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ..utils import seq_lengths_to_mask

SQRT_EPS = 1e-5


class MeanVarianceNorm(nn.Module):
    """Class to apply short-time mean-variance normalization to features.

    Attributes:
      norm_mean:    if True, it normalizes the mean.
      norm_var:     if True, is also normalized the variance.
      left_context:  left context for the window that computes the normalization stats.
      right_context: right context for the window that computes the normalization stats.
      dim:           normalization dimension (time dimension).

    If left_context = right_context = 0, it computes the stats on the whole utterance.
    """

    def __init__(
        self, norm_mean=True, norm_var=False, left_context=0, right_context=0, dim=1
    ):
        super().__init__()
        self.norm_mean = norm_mean
        self.norm_var = norm_var
        self.left_context = left_context
        self.right_context = right_context
        self.dim = dim

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}(norm_mean={}, norm_var={}, left_context={}, right_context={}, dim={})".format(
            self.__class__.__name__,
            self.norm_mean,
            self.norm_var,
            self.left_context,
            self.right_context,
            self.dim,
        )
        return s

    def forward(self, x, x_lengths=None, x_mask=None):
        """Short-time mean-var normalizes feature tensor.

        Args:
          x: feature tensor.

        Returns:
          Normalized feature tensor.
        """
        if not self.norm_mean and not self.norm_var:
            return x

        if self.dim != 1:
            x = x.transpose(x, 1, self.dim)

        max_length = x.size(1)
        if x_lengths is not None and x_mask is None:
            x_mask = seq_lengths_to_mask(
                x_lengths,
                max_length,
                dtype=x.dtype,
                none_if_all_max=True,
            )

        if (self.left_context == 0 and self.right_context == 0) or (
            max_length <= self.left_context + self.right_context + 1
        ):
            x = self.normalize_global(x, x_mask)
        else:
            x = self.normalize_cumsum(x, x_mask)

        if self.dim != 1:
            x = x.transpose(x, 1, self.dim).contiguous()

        return x

    def _normalize_global_nomask(self, x):
        """Applies global mean-var normalization."""
        # Global mean/var norm.

        if self.norm_mean:
            m_x = torch.mean(x, dim=1, keepdim=True)
            x = x - m_x

        if self.norm_var:
            s_x = torch.std(x, dim=1, keepdim=True).clamp(min=1e-5)
            x = x / s_x

        return x

    def _normalize_global_mask(self, x, x_mask):
        """Applies global mean-var normalization with masking."""
        # Global mean/var norm.
        den = torch.mean(x_mask, dim=1, keepdim=True)
        x = x * x_mask
        m_x = torch.mean(x, dim=1, keepdim=True) / den
        if self.norm_mean:
            x = x - m_x
            if self.norm_var:
                s2_x = torch.mean(x**2, dim=1, keepdim=True) / den
                s_x = torch.sqrt(s2_x.clamp(min=SQRT_EPS))
                x = x / s_x
        elif self.norm_var:
            s2_x = torch.mean((x - m_x) ** 2, dim=1, keepdim=True) / den
            s_x = torch.sqrt(s2_x.clamp(min=SQRT_EPS))
            x = x / s_x

        return x

    def normalize_global(self, x, x_mask=None):
        """Applies global mean-var normalization."""
        # Global mean/var norm.
        if x_mask is None:
            return self._normalize_global_nomask(x)
        else:
            return self._normalize_global_mask(x, x_mask)

    def _prenormalize_cumsum(self, x, x_mask):
        """substract first global mean
        it will help cumsum numerical stability
        and set masked values to the global mean"""
        if self.norm_mean or x_mask is not None:
            # substract first global mean
            # it will help cumsum numerical stability
            if x_mask is not None:
                x = x * x_mask
                den = torch.mean(x_mask, dim=1, keepdim=True)
            else:
                den = 1
            m_x = torch.mean(x, dim=1, keepdim=True) / den

        if self.norm_mean:
            x = x - m_x
            if x_mask is not None:
                x = x * x_mask
        elif x_mask is not None:
            x = x * x_mask + m_x * (1 - x_mask)

        return x

    def normalize_cumsum(self, x, x_mask=None):
        """Applies short-time mean-var normalization using cumulative sums."""

        x = self._prenormalize_cumsum(x, x_mask)
        total_context = self.left_context + self.right_context + 1

        xx = nn.functional.pad(
            x.transpose(1, -1), (self.left_context, self.right_context), mode="reflect"
        ).transpose(1, -1)

        if self.norm_mean or self.norm_var:
            c_x = torch.cumsum(xx, dim=1)
            m_x = (
                c_x[:, total_context - 1 :] - c_x[:, : -total_context + 1]
            ) / total_context

        if self.norm_var:
            c_x = torch.cumsum(xx**2, dim=1)
            m_x2 = (
                c_x[:, total_context - 1 :] - c_x[:, : -total_context + 1]
            ) / total_context

        if self.norm_mean:
            x = x - m_x

        if self.norm_var:
            s_x = torch.sqrt((m_x2 - m_x**2).clamp(min=SQRT_EPS))
            x = x / s_x

        return x.contiguous()

    @staticmethod
    def filter_args(**kwargs):
        """Filters ST-MVN args from arguments dictionary.

        Args:
          kwargs: Arguments dictionary.

        Returns:
          Dictionary with ST-MVN options.
        """

        valid_args = (
            "no_norm_mean",
            "norm_mean",
            "norm_var",
            "left_context",
            "right_context",
            "context",
        )
        d = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        if "no_norm_mean" in d:
            d["norm_mean"] = not d["no_norm_mean"]
            del d["no_norm_mean"]

        if "context" in d:
            if d["context"] is not None:
                d["left_context"] = d["context"]
                d["right_context"] = d["context"]
            del d["context"]

        return d

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds ST-CMVN options to parser.

        Args:
          parser: Arguments parser
          prefix: Options prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--no-norm-mean",
            default=False,
            action="store_true",
            help="don't center the features",
        )

        parser.add_argument(
            "--norm-var",
            default=False,
            action="store_true",
            help="normalize the variance of the features",
        )

        parser.add_argument(
            "--left-context",
            type=int,
            default=150,
            help="past context in number of frames",
        )

        parser.add_argument(
            "--right-context",
            type=int,
            default=150,
            help="future context in number of frames",
        )

        parser.add_argument(
            "--context",
            type=int,
            default=None,
            help=(
                "past/future context in number of frames, "
                "overwrites left-context and right-context options"
            ),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='mean-var norm. options')

    add_argparse_args = add_class_args
