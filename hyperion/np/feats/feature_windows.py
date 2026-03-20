"""
Copyright 2018 Jesus Villalba (Johns Hopkins University)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional

import numpy as np
from jsonargparse import ActionParser, ArgumentParser
from scipy.signal.windows import blackman, hamming, hann

from ...hyp_defs import float_cpu


class FeatureWindowFactory:
    """Factory class to create windowing functions."""

    @staticmethod
    def create(window_type: str, N: int, sym: bool = False) -> np.ndarray:
        """Creates a windowing function.

        Args:
          window_type: window type in ["povey", "hamming", "hanning", "blackman", "rectangular"]
          N: num samples.
          sym: if True, the window is symmetric, otherwise non-symmetric.

        Returns:
          Window as (N,) numpy array.
        """
        if not isinstance(N, int) or N < 1:
            raise ValueError(f"N must be a positive integer, got {N!r}")

        window_type = window_type.lower()

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

        raise ValueError(f"invalid window type {window_type!r}")

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Adds feature window options to parser.

        Args:
          parser: Arguments parser
          prefix: Options prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--window-type",
            default="povey",
            choices=["hamming", "hanning", "povey", "rectangular", "blackman"],
            help=(
                'Type of window ("hamming"|"hanning"|"povey"|"rectangular"|"blackman")'
            ),
        )
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    add_argparse_args = add_class_args
