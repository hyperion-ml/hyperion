"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ..utils import seq_lengths_to_mask

SQRT_EPS = 1e-5


class MeanVarianceNorm(nn.Module):
    """Short-time mean-variance normalization for sequence features.

    Attributes:
        norm_mean: If ``True``, applies mean normalization.
        norm_var: If ``True``, applies variance normalization.
        left_context: Left context (frames) for local normalization windows.
        right_context: Right context (frames) for local normalization windows.
        dim: Time dimension used for normalization.
    """

    def __init__(
        self,
        norm_mean: bool = True,
        norm_var: bool = False,
        left_context: int = 0,
        right_context: int = 0,
        dim: int = 1,
    ) -> None:
        """Initializes the MVN layer.

        Args:
            norm_mean: If ``True``, normalizes the sequence mean.
            norm_var: If ``True``, normalizes the sequence variance.
            left_context: Left context (frames) for local window statistics.
            right_context: Right context (frames) for local window statistics.
            dim: Time dimension in the input tensor.

        Returns:
            None.
        """
        super().__init__()
        self.norm_mean = norm_mean
        self.norm_var = norm_var
        self.left_context = left_context
        self.right_context = right_context
        self.dim = dim

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        s = "{}(norm_mean={}, norm_var={}, left_context={}, right_context={}, dim={})".format(
            self.__class__.__name__,
            self.norm_mean,
            self.norm_var,
            self.left_context,
            self.right_context,
            self.dim,
        )
        return s

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies short-time MVN to an input feature tensor.

        Args:
            x: Feature tensor.
            x_lengths: Sequence lengths used to build a mask when ``x_mask`` is ``None``.
            x_mask: Mask of valid frames. If provided, ``x_lengths`` is ignored.

        Returns:
            Normalized feature tensor.
        """
        if not self.norm_mean and not self.norm_var:
            return x

        if self.dim != 1:
            x = x.transpose(1, self.dim)
            if x_mask is not None and x_mask.dim() == x.dim():
                x_mask = x_mask.transpose(1, self.dim)

        max_length = x.size(1)
        if x_lengths is not None and x_mask is None:
            x_mask = seq_lengths_to_mask(
                x_lengths,
                max_length,
                dtype=x.dtype,
                ndim=x.dim(),
                none_if_all_max=True,
            )

        if (self.left_context == 0 and self.right_context == 0) or (
            max_length <= self.left_context + self.right_context + 1
        ):
            x = self.normalize_global(x, x_mask)
        else:
            x = self.normalize_cumsum(x, x_mask)

        if self.dim != 1:
            x = x.transpose(1, self.dim).contiguous()

        return x

    def _normalize_global_nomask(self, x: torch.Tensor) -> torch.Tensor:
        """Applies utterance-level normalization without masking.

        Args:
            x: Input feature tensor.

        Returns:
            Normalized feature tensor.
        """
        # Global mean/var norm.

        if self.norm_mean:
            m_x = torch.mean(x, dim=1, keepdim=True)
            x = x - m_x

        if self.norm_var:
            s_x = torch.std(x, dim=1, keepdim=True).clamp(min=1e-5)
            x = x / s_x

        return x

    def _normalize_global_mask(
        self, x: torch.Tensor, x_mask: torch.Tensor
    ) -> torch.Tensor:
        """Applies utterance-level normalization with a frame mask.

        Args:
            x: Input feature tensor.
            x_mask: Valid-frame mask broadcastable to ``x``.

        Returns:
            Normalized feature tensor.
        """
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

    def normalize_global(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Applies utterance-level normalization.

        Args:
            x: Input feature tensor.
            x_mask: Optional valid-frame mask.

        Returns:
            Normalized feature tensor.
        """
        # Global mean/var norm.
        if x_mask is None:
            return self._normalize_global_nomask(x)
        return self._normalize_global_mask(x, x_mask)

    def _prenormalize_cumsum(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Preprocesses input before cumulative-sum local normalization.

        Args:
            x: Input feature tensor.
            x_mask: Optional valid-frame mask.

        Returns:
            Pre-normalized tensor for cumulative-sum processing.
        """
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

    def normalize_cumsum(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Applies local mean/variance normalization via cumulative sums.

        Args:
            x: Input feature tensor.
            x_mask: Optional valid-frame mask.

        Returns:
            Normalized feature tensor.
        """

        x = self._prenormalize_cumsum(x, x_mask)
        total_context = self.left_context + self.right_context + 1

        xx = nn.functional.pad(
            x.transpose(1, -1),
            (self.left_context + 1, self.right_context),
            mode="reflect",
        ).transpose(1, -1)
        xx[:, 0] = 0

        if self.norm_mean or self.norm_var:
            c_x = torch.cumsum(xx, dim=1)
            m_x = (c_x[:, total_context:] - c_x[:, :-total_context]) / total_context

        if self.norm_var:
            c_x = torch.cumsum(xx**2, dim=1)
            m_x2 = (c_x[:, total_context:] - c_x[:, :-total_context]) / total_context

        if self.norm_mean:
            x = x - m_x

        if self.norm_var:
            s_x = torch.sqrt((m_x2 - m_x**2).clamp(min=SQRT_EPS))
            x = x / s_x

        return x.contiguous()

    @staticmethod
    def filter_args(**kwargs: Any) -> dict[str, Any]:
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
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Adds ST-CMVN options to parser.

        Args:
            parser: Arguments parser.
            prefix: Optional nested option prefix.

        Returns:
            None.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--norm-mean",
            default=True,
            action=ActionYesNo,
            help="center the features",
        )

        parser.add_argument(
            "--norm-var",
            default=False,
            action=ActionYesNo,
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
