"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, ActionParser

import torch
import torch.nn as nn


class MeanVarianceNorm(nn.Module):
    def __init__(
        self, norm_mean=True, norm_var=False, left_context=0, right_context=0, dim=1
    ):

        super(MeanVarianceNorm, self).__init__()
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

    def forward(self, x):

        T = x.shape[self.dim]
        if (self.left_context == 0 and self.right_context == 0) or (
            T <= self.left_context + self.right_context + 1
        ):
            return self.normalize_global(x)

        return self.normalize_cumsum(x)

    def normalize_global(self, x):
        # Global mean/var norm.
        if self.norm_mean:
            m_x = torch.mean(x, dim=self.dim, keepdim=True)
            x = x - m_x

        if self.norm_var:
            s_x = torch.std(x, dim=self.dim, keepdim=True).clamp(min=1e-5)
            x = x / s_x

        return x

    def normalize_cumsum(self, x):

        if self.norm_mean:
            # substract first global mean
            # it will help cumsum numerical stability
            m_x = torch.mean(x, dim=self.dim, keepdim=True)
            x = x - m_x

        if self.dim != 1:
            x = x.transpose(self.dim, 1)

        total_context = self.left_context + self.right_context + 1

        xx = nn.functional.pad(
            x.transpose(1, -1), (self.left_context, self.right_context), mode="reflect"
        ).transpose(1, -1)

        if self.norm_mean:
            c_x = torch.cumsum(xx, dim=1)
            m_x = (
                c_x[:, total_context - 1 :] - c_x[:, : -total_context + 1]
            ) / total_context

        if self.norm_var:
            c_x = torch.cumsum(xx ** 2, dim=1)
            m_x2 = (
                c_x[:, total_context - 1 :] - c_x[:, : -total_context + 1]
            ) / total_context

        if self.norm_mean:
            x = x - m_x

        if self.norm_var:
            s_x = torch.sqrt((m_x2 - m_x ** 2).clamp(min=1e-5))
            x = x / s_x

        if self.dim != 1:
            x = x.transpose(self.dim, 1)

        return x.contiguous()

    @staticmethod
    def filter_args(**kwargs):
        """Filters ST-CMVN args from arguments dictionary.

        Args:
          kwargs: Arguments dictionary.

        Returns:
          Dictionary with ST-CMVN options.
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
