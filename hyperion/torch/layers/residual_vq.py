"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from .vq import VectorQuantizerOutput, VQDistanceType
from .vq_factory import VectorQuantizerFactory, vq_dict


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ).

    Implements multi-stage residual quantization: at each stage a base vector
    quantizer maps the current residual to a codebook vector, that vector is
    accumulated into the output, and the residual is updated.

    This RVQ composes any of your vector quantizer variants (NN / Gumbel /
    EMA-NN / EMA-Gumbel) created via :class:`VectorQuantizerFactory`. It supports
    rank-2 to rank-5 inputs, and optional variable-length masks/lengths that are
    forwarded to each stage.

    Attributes:
        in_feats (int): Input feature dimension (D_in).
        num_quantizers (int): Number of residual stages (M).
        codebook_sizes (List[int]): List of codebook sizes (K per stage).
        codebook_dims (List[Optional[int]]): List of codebook dimensions (D per stage).
        quantizers (nn.ModuleList): The instantiated per-stage quantizers.
        quantizer_dropout (float): Fraction of examples using a truncated depth K<M
            in training (applied deterministically to the first ⌊B·p⌋ examples).
    """

    def __init__(
        self,
        in_feats: int,
        num_quantizers: int,
        codebook_sizes: Union[int, List[int]],
        codebook_dims: Union[int, List[int], None] = None,
        base_vq_type: str = "nn_vq",
        quantizer_dropout: float = 0.0,
        **kwargs,
    ):
        """
        Initializes a residual stack of vector quantizers.

        Args:
            in_feats (int): Input feature dimension.
            num_quantizers (int): Number of residual stages (M).
            codebook_sizes (int or List[int]): Either a single K (broadcasted to all
                stages) or a list of length M.
            codebook_dims (int or List[int] or None): Either a single D (broadcasted),
                a list of length M, or None to use ``in_feats`` per stage.
            base_vq_type (str): Type of vector quantizer to use for each stage.
                Options are {"nn_vq", "ema_nn_vq", "gumbel_vq", "ema_gumbel_vq"}.
            quantizer_dropout (float, optional): Fraction ∈ [0,1]. During training,
                a proportion of examples (⌊B·p⌋) will use fewer than M stages,
                with their depth sampled uniformly from 1..M.
            **kwargs: Extra keyword arguments passed to
                :func:`VectorQuantizerFactory.create` (e.g., distance metric,
                EMA decay, Gumbel temperature settings).
        """
        super().__init__()

        self.in_feats = in_feats
        self.num_quantizers = num_quantizers
        if isinstance(codebook_sizes, int):
            codebook_sizes = [codebook_sizes] * num_quantizers
        else:
            assert isinstance(codebook_sizes, list)
            if len(codebook_sizes) == 1:
                codebook_sizes = codebook_sizes * num_quantizers

        if isinstance(codebook_dims, int) or codebook_dims is None:
            codebook_dims = [codebook_dims] * num_quantizers
        else:
            assert isinstance(codebook_dims, list)
            if len(codebook_dims) == 1:
                codebook_dims = codebook_dims * num_quantizers

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

        self.quantizers = nn.ModuleList(
            [
                VectorQuantizerFactory.create(
                    base_vq_type,
                    in_feats,
                    codebook_sizes[i],
                    codebook_dims[i],
                    **kwargs,
                )
                for i in range(num_quantizers)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def __str__(self):
        s = f"{self.__class__.__name__}({self.in_feats}, {self.num_quantizers}, {self.codebook_sizes}, {self.codebook_dims}, quantizer_dropout={self.quantizer_dropout})"
        for i, q in enumerate(self.quantizers):
            s += f"\n  [{i}] " + q.__str__().replace("\n", "\n      ")
        return s

    def __repr__(self):
        return self.__str__()

    def get_config(self):
        """Returns module configuration for serialization."""
        cfg = {
            "in_feats": self.in_feats,
            "num_quantizers": self.num_quantizers,
            "codebook_sizes": self.codebook_sizes,
            "codebook_dims": self.codebook_dims,
            "quantizer_dropout": self.quantizer_dropout,
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
        num_quantizers: int = None,
        return_codes: bool = False,
    ):
        """
        Run multi-stage residual quantization.

        At each stage i:
          * The stage quantizer outputs a codebook vector z_q_i.
          * z_q_i is added to the output accumulator (masked if dropped).
          * The residual is always updated as residual ← residual − z_q_i.

        Args:
            z (Tensor): Input tensor of shape (B, ..., D_in).
            z_lengths (Tensor, optional): Sequence lengths of shape (B,). Used
                to build masks if `z_mask` is not provided.
            z_mask (Tensor, optional): Boolean/float mask of shape (B, T) marking
                valid positions.
            num_quantizers (int, optional): In evaluation, use only the first N
                stages. Defaults to all M. Ignored in training if dropout is active.
            return_codes (bool, optional): If True, also return per-stage code indices.

        Returns:
            VectorQuantizerOutput: Dataclass with fields:
                - z_q (Tensor): Quantized output tensor, same shape as `z`.
                - codebook_loss (Tensor): Scalar loss; for EMA variants, this is zero.
                - commitment_loss (Tensor): Scalar loss encouraging encoder outputs
                  to match their codes.
                - perplexity (Tensor): Tensor of per-stage perplexities, shape (M,).
                - codes (Tensor, optional): If requested, code indices stacked as
                  (B, M, T,...).
        """
        z_q = torch.zeros_like(z)
        residual = z
        commitment_loss = 0.0
        codebook_loss = 0.0
        ppl = []
        codes = []

        if num_quantizers is None:
            num_quantizers = self.num_quantizers

        # --- Quantizer dropout (training only) ---
        if self.training and self.quantizer_dropout > 0.0 and self.num_quantizers > 1:
            with torch.no_grad():
                # determine how many quantizers to use for each example in the batch
                B = z.shape[0]
                # default keep = full depth
                per_ex_K = torch.full(
                    (B,), self.num_quantizers, dtype=torch.long, device=z.device
                )
                # choose which examples to drop stages for
                num_dropout = int(round(B * self.quantizer_dropout))
                if num_dropout > 0:
                    per_ex_K[:num_dropout] = torch.randint(
                        1, self.num_quantizers + 1, (num_dropout,), device=z.device
                    )
        else:
            per_ex_K = None  # not used

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= num_quantizers:
                break

            vq_output = quantizer(
                residual, z_lengths, z_mask, return_codes=return_codes
            )

            z_q_i = vq_output.z_q
            # Create mask to apply quantizer dropout
            if per_ex_K is not None:
                dropout_mask = torch.full((z.shape[0],), i, device=z.device) < per_ex_K

                z_q = z_q + z_q_i * dropout_mask.view(-1, *([1] * (z_q_i.dim() - 1)))
                commitment_loss_i = vq_output.commitment_loss * dropout_mask
                codebook_loss_i = vq_output.codebook_loss * dropout_mask
            else:
                z_q = z_q + z_q_i
                commitment_loss_i = vq_output.commitment_loss
                codebook_loss_i = vq_output.codebook_loss

            commitment_loss += commitment_loss_i.mean()
            codebook_loss += codebook_loss_i.mean()

            residual = residual - z_q_i
            ppl.append(vq_output.perplexity)
            if return_codes:
                codes.append(vq_output.codes)

        if return_codes:
            codes = torch.stack(codes, dim=1)
        else:
            codes = None

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
    ):
        """
        Register RVQ-specific CLI arguments.

        Args:
            parser (ArgumentParser): The parser to which arguments are added.
            prefix (str, optional): If provided, a nested parser is created and
                attached under ``--{prefix}``.
            skip (set, optional): Set of argument names to skip.

        Adds:
            --num-quantizers (int): Number of residual stages (M).
            --quantizer-dropout (float): Dropout probability for stages during training.
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
            help="Number of quantizers (M) to use in the residual vector quantizer",
        )
        parser.add_argument(
            "--quantizer-dropout",
            type=float,
            default=0.0,
            help="Probability of dropping each quantizer during training for regularization",
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
