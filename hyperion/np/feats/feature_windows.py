"""
 Copyright 2018 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

import numpy as np
from scipy.signal import blackman, hamming, hann

from ...hyp_defs import float_cpu


class FeatureWindowFactory(object):
    """Factory class to create windowing functions."""

    @staticmethod
    def create(window_type, N, sym=False):
        """Creates a windowing function.

        Args:
          window_type: window type in ["povey", "hamming", "hanning", "blackman", "rectangular"]
          N: num samples.
          sym: if True, the window is symmetric, otherwise non-symmetric.

        Returns:
          Window as (N,) numpy array.
        """

        if window_type == "povey":
            return np.power(
                0.5 - 0.5 * np.cos(2 * np.pi / N * np.arange(N, dtype=float_cpu())),
                0.85,
            )
        if window_type == "hamming":
            return hamming(N, sym).astype(float_cpu(), copy=False)
        if window_type == "hanning":
            return hann(N, sym).astype(float_cpu(), copy=False)
        if window_type == "blackman":
            return blackman(N, sym).astype(float_cpu(), copy=False)
        if window_type == "rectangular":
            return np.ones((N,), dtype=float_cpu())

        raise Exception("Invalid window type %s" % window_type)

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds feature window options to parser.

        Args:
          parser: Arguments parser
          prefix: Options prefix.
        """
        if prefix is None:
            p1 = "--"
        else:
            p1 = "--" + prefix + "."

        parser.add_argument(
            p1 + "window-type",
            default="povey",
            choices=["hamming", "hanning", "povey", "rectangular", "blackman"],
            help=(
                'Type of window ("hamming"|"hanning"|"povey"|"rectangular"|"blackmann")'
            ),
        )

        # parser.add_argument(
        #     p1+'blackman-coeff', type=float,
        #     default=0.42,
        #     help='Constant coefficient for generalized Blackman window. (default = 0.42)')

    add_argparse_args = add_class_args
