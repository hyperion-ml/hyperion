"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from .vq import VectorQuantizerOutput, VQDistanceType
from .vq_factory import VectorQuantizerFactory, vq_dict


class GroupVectorQuantizer(nn.Module):
    """
    Group Vector Quantizer (GVQ).

    Runs multiple vector quantizers on disjoint channel partitions
    of the input. Each stage receives a slice of the feature dimension and
    quantizes it independently; the outputs are concatenated back along the
    same feature dimension.

    The per-stage quantizers are created via `VectorQuantizerFactory` and can be
    any of your variants (NN / EMA-NN / Gumbel / EMA-Gumbel). Masking/lengths
    are forwarded to each stage unchanged.

    Attributes:
        in_feats (int): Input feature dimension (D_in).
        num_quantizers (int): Number of grouped quantizers (M).
        codebook_sizes (List[int]): Codebook size per stage (length M).
        codebook_dims (List[Optional[int]]): Codebook dim per stage (length M).
        quantizers (nn.ModuleList): Instantiated per-stage quantizers.
        base_vq_type (str): Factory key used for per-stage quantizers.
    """

    def __init__(
        self,
        in_feats: int,
        num_quantizers: int,
        codebook_sizes: Union[int, List[int]],
        codebook_dims: Union[int, List[int], None] = None,
        base_vq_type: str = "nn_vq",
        **kwargs,
    ) -> None:
        """
        Initializes a bank of grouped vector quantizers.

        Args:
            in_feats (int): Input feature dimension (D_in).
            num_quantizers (int): Number of grouped quantizers (M).
            codebook_sizes (int | List[int]): Single K (broadcasted) or list of length M.
            codebook_dims (int | List[int] | None): Single D (broadcasted), list of
                length M, or None to use `in_feats` per stage.
            base_vq_type (str): One of {"nn_vq", "ema_nn_vq", "gumbel_vq", "ema_gumbel_vq"}.
            **kwargs: Extra args forwarded to `VectorQuantizerFactory.create`
                (e.g., distance metric, EMA decay/eps, Gumbel temperature, layout).
        """
        super().__init__()

        self.in_feats = in_feats
        self.num_quantizers = num_quantizers
        if isinstance(codebook_sizes, int):
            codebook_sizes = [codebook_sizes] * num_quantizers
        if isinstance(codebook_dims, int) or codebook_dims is None:
            codebook_dims = [codebook_dims] * num_quantizers

        assert num_quantizers == len(codebook_sizes), (
            "num_quantizers must be equal to the length of codebook_sizes"
            f"({num_quantizers} != {len(codebook_sizes)})"
        )
        assert num_quantizers == len(codebook_dims), (
            "num_quantizers must be equal to the length of codebook_dims"
            f"({num_quantizers} != {len(codebook_dims)})"
        )

        self.codebook_sizes = codebook_sizes
        self.codebook_dims = codebook_dims
        self.base_vq_type = base_vq_type
        in_feats_i = in_feats // self.num_quantizers
        assert in_feats_i * self.num_quantizers == in_feats, (
            "in_feats must be divisible by num_quantizers"
            f"({in_feats} % {num_quantizers} != 0)"
        )
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizerFactory.create(
                    base_vq_type,
                    in_feats_i,
                    codebook_sizes[i],
                    codebook_dims[i],
                    **kwargs,
                )
                for i in range(num_quantizers)
            ]
        )

    def __str__(self) -> str:
        s = f"{self.__class__.__name__}({self.in_feats}, {self.num_quantizers}, {self.codebook_sizes}, {self.codebook_dims})"
        for i, q in enumerate(self.quantizers):
            s += f"\n  [{i}] " + q.__str__().replace("\n", "\n      ")
        return s

    def __repr__(self) -> str:
        return self.__str__()

    def get_config(self) -> dict[str, Any]:
        """Returns module configuration for serialization."""
        cfg = {
            "in_feats": self.in_feats,
            "num_quantizers": self.num_quantizers,
            "codebook_sizes": self.codebook_sizes,
            "codebook_dims": self.codebook_dims,
            "base_vq_type": self.base_vq_type,
        }
        # Add base VQ type and args (assumed same for all stages)
        base_vq = self.quantizers[0]
        for k, v in base_vq.get_config().items():
            if k not in {"codebook_size", "codebook_dim", "in_feats", "vq_type"}:
                cfg[k] = v

        return cfg

    def forward(
        self,
        z: torch.Tensor,
        z_lengths: torch.Tensor | None = None,
        z_mask: torch.Tensor | None = None,
        return_codes: bool = False,
    ) -> VectorQuantizerOutput:
        """
        Runs M grouped quantizers on channel partitions and concatenates outputs.

        Args:
            z (Tensor): Input of shape (B, ..., D_in). If the per-stage quantizers
                use `channels_last=True`, features are taken from the last dim;
                otherwise from dim=1.
            z_lengths (Tensor, optional): Sequence lengths (B,). Used by stages to
                derive masks when `z_mask` is not provided.
            z_mask (Tensor, optional): Boolean/float mask of shape (B, T).
            return_codes (bool, optional): If True, returns per-stage code indices.

        Returns:
            VectorQuantizerOutput:
                - z_q (Tensor): Concatenated quantized output, same shape as `z`.
                - codebook_loss (Tensor): Sum of per-stage codebook losses (EMA stages
                  typically output zeros).
                - commitment_loss (Tensor): Sum of per-stage commitment losses.
                - perplexity (Tensor): Per-stage perplexities stacked as shape (M,).
                - codes (Tensor | None): If requested, stacked codes of shape (B, M, T, ...).
        """

        commitment_loss = 0.0
        codebook_loss = 0.0
        z_q = []
        ppl = []
        codes = []

        num_quantizers = self.num_quantizers
        chunk_dim = -1 if self.quantizers[0].channels_last else 1
        z = z.chunk(num_quantizers, dim=chunk_dim)

        for i, quantizer in enumerate(self.quantizers):

            vq_output = quantizer(z[i], z_lengths, z_mask, return_codes=return_codes)

            z_q.append(vq_output.z_q)
            commitment_loss_i = vq_output.commitment_loss
            codebook_loss_i = vq_output.codebook_loss

            commitment_loss += commitment_loss_i.mean()
            codebook_loss += codebook_loss_i.mean()

            ppl.append(vq_output.perplexity)
            if return_codes:
                codes.append(vq_output.codes)

        if return_codes:
            codes = torch.stack(codes, dim=1)
        else:
            codes = None

        z_q = torch.cat(z_q, dim=chunk_dim)
        ppl = torch.stack(ppl, dim=0)

        output = VectorQuantizerOutput(
            z_q=z_q,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            perplexity=ppl,
            codes=codes,
        )

        return output

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None, skip: Optional[set] = None
    ) -> None:
        """
        Register  CLI arguments.

        Args:
            parser (ArgumentParser): The parser to which arguments are added.
            prefix (str, optional): If provided, a nested parser is created and
                attached under ``--{prefix}``.
            skip (set, optional): Set of argument names to skip.

        Adds:
            --num-quantizers (int): Number of residual stages (M).
            --base-vq-type (str): Quantizer type for each stage.
            --codebook-sizes (List[int]): Codebook size K for each stage.
            --codebook-dims (List[int]): Codebook dimension D for each stage
                (defaults to in_feats if omitted).

        Notes:
            Also adds the standard arguments of individual quantizers by calling
            :func:`VectorQuantizerFactory.add_class_args`.
        """
        if skip is None:
            skip = set()

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--num-quantizers",
            type=int,
            required=True,
            help="Number of quantizers (M) in the grouped vector quantizer",
        )
        parser.add_argument(
            "--base-vq-type",
            choices=list(vq_dict.keys()),
            default="nn_vq",
            help="Type of vector quantizer to use in each stage",
        )
        parser.add_argument(
            "--codebook-sizes",
            type=int,
            nargs="+",
            required=True,
            help="List of codebook sizes (K) for each quantizer",
        )
        parser.add_argument(
            "--codebook-dims",
            type=int,
            nargs="+",
            default=None,
            help="List of codebook dimensions (D) for each quantizer. "
            "If not provided, defaults to `in_feats`",
        )

        skip = skip | {"vq_type", "codebook_size", "codebook_dim"}
        VectorQuantizerFactory.add_class_args(parser, skip=skip)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
