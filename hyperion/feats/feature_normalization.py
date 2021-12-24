"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
from jsonargparse import ArgumentParser, ActionParser
from scipy.signal import convolve2d

from ..hyp_defs import float_cpu


class MeanVarianceNorm(object):
    """Class to perform mean and variance normalization

    Attributes:
       norm_mean: normalize mean
       norm_var: normalize variance
       left_context: past context of the sliding window, if None all past frames.
       right_context: future context of the sliding window, if None all future frames.

    If left_context==right_context==None, it will apply global mean/variance normalization.
    """

    def __init__(
        self, norm_mean=True, norm_var=False, left_context=None, right_context=None
    ):
        self.norm_mean = norm_mean
        self.norm_var = norm_var
        self.left_context = left_context
        self.right_context = right_context

    def normalize(self, x):
        return self.normalize_cumsum(x)

    def normalize_global(self, x):
        # Global mean/var norm.
        if self.norm_mean:
            m_x = np.mean(x, axis=0, keepdims=True)
            x = x - m_x

        if self.norm_var:
            s_x = np.std(x, axis=0, keepdims=True)
            x = x / s_x

        return x

    def normalize_conv(self, x):
        """Normalize featurex in x
           Uses convolution operator
        Args:
          x: Input feature matrix.

        Returns:
          Normalized feature matrix.
        """

        x = self.normalize_global(x)

        if self.right_context is None and self.left_context is None:
            return x

        if self.left_context is None:
            left_context = x.shape[0]
        else:
            left_context = self.left_context

        if self.right_context is None:
            right_context = x.shape[0]
        else:
            right_context = self.right_context

        total_context = left_context + right_context + 1

        if x.shape[0] <= min(right_context, left_context) + 1:
            # if context is larger than the signal we still return global normalization
            return x

        v1 = np.ones((x.shape[0], 1), dtype=float_cpu())
        h = np.ones((total_context, 1), dtype=float_cpu())

        counts = convolve2d(v1, h)[right_context : right_context + x.shape[0]]
        m_x = convolve2d(x, h)[right_context : right_context + x.shape[0]]
        m_x /= counts

        if self.norm_var:
            m2_x = convolve2d(x * x, h)[right_context : right_context + x.shape[0]]
            m2_x /= counts
            s2_x = m2_x - m_x ** 2
            s2_x[s2_x < 1e-5] = 1e-5
            s_x = np.sqrt(s2_x)

        if self.norm_mean:
            x -= m_x

        if self.norm_var:
            x /= s_x

        return x

    def normalize_cumsum(self, x):
        """Normalize featurex in x
           Uses cumsum
        Args:
          x: Input feature matrix.

        Returns:
          Normalized feature matrix.
        """

        x = self.normalize_global(x)

        if self.right_context is None and self.left_context is None:
            return x

        if self.left_context is None:
            left_context = x.shape[0]
        else:
            left_context = self.left_context

        if self.right_context is None:
            right_context = x.shape[0]
        else:
            right_context = self.right_context

        total_context = left_context + right_context + 1

        if x.shape[0] <= min(right_context, left_context) + 1:
            # if context is larger than the signal we still return global normalization
            return x

        c_x = np.zeros(
            (
                x.shape[0] + total_context,
                x.shape[1],
            ),
            dtype=float_cpu(),
        )
        counts = np.zeros(
            (
                x.shape[0] + total_context,
                1,
            ),
            dtype=float_cpu(),
        )

        c_x[left_context + 1 : left_context + x.shape[0] + 1] = np.cumsum(x, axis=0)
        c_x[left_context + x.shape[0] + 1 :] = c_x[left_context + x.shape[0]]
        counts[left_context + 1 : left_context + x.shape[0] + 1] = np.arange(
            1, x.shape[0] + 1, dtype=float_cpu()
        )[:, None]
        counts[left_context + x.shape[0] + 1 :] = x.shape[0]

        if self.norm_var:
            c2_x = np.zeros(
                (
                    x.shape[0] + total_context,
                    x.shape[1],
                ),
                dtype=float_cpu(),
            )
            c2_x[left_context + 1 : left_context + x.shape[0] + 1] = np.cumsum(
                x * x, axis=0
            )
            c2_x[left_context + x.shape[0] + 1 :] = c2_x[left_context + x.shape[0]]

        counts = counts[total_context:] - counts[:-total_context]
        m_x = (c_x[total_context:] - c_x[:-total_context]) / counts

        if self.norm_mean:
            x -= m_x

        if self.norm_var:
            m2_x = (c2_x[total_context:] - c2_x[:-total_context]) / counts
            s2_x = m2_x - m_x ** 2
            s2_x[s2_x < 1e-5] = 1e-5
            s_x = np.sqrt(s2_x)
            x /= s_x

        return x

    def normalize_slow(self, x):

        x = self.normalize_global(x)

        if self.right_context is None and self.left_context is None:
            return x

        if self.left_context is None:
            left_context = x.shape[0]
        else:
            left_context = self.left_context

        if self.right_context is None:
            right_context = x.shape[0]
        else:
            right_context = self.right_context

        m_x = np.zeros_like(x)
        s_x = np.zeros_like(x)

        for i in range(x.shape[0]):
            idx1 = max(i - left_context, 0)
            idx2 = min(i + right_context, x.shape[0] - 1) + 1
            denom = idx2 - idx1
            m_x[i] = np.mean(x[idx1:idx2], axis=0)
            s_x[i] = np.std(x[idx1:idx2], axis=0)

        if self.norm_mean:
            x -= m_x
        if self.norm_var:
            s_x[s_x < np.sqrt(1e-5)] = np.sqrt(1e-5)
            x /= s_x

        return x

    @staticmethod
    def filter_args(**kwargs):
        """Filters ST-CMVN args from arguments dictionary.

        Args:
          prefix: Options prefix.
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

        neg_args1 = ("no_norm_mean",)
        neg_args2 = ("norm_mean",)

        for a, b in zip(neg_args1, neg_args2):
            d[b] = not d[a]
            del d[a]

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
            # help='mean-var norm options')

    add_argparse_args = add_class_args
