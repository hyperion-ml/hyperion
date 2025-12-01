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
        base_vq_type (str): Type of vector quantizer used in each stage.
        quantizer_dropout (float): Fraction of examples using a truncated depth K<M
            in training (applied deterministically to the first ⌊B·p⌋ examples).
        bypass_prob (Tensor): Current probability of bypassing quantization entirely
            (output = input) during training.
        bypass_final_prob (float): Final bypass probability after annealing.
        bypass_anneal_steps (int): Number of steps over which to anneal the bypass
            probability from `bypass_init_prob` to `bypass_final_prob`.
    """

    def __init__(
        self,
        in_feats: int,
        num_quantizers: int,
        codebook_sizes: Union[int, List[int]],
        codebook_dims: Union[int, List[int], None] = None,
        base_vq_type: str = "nn_vq",
        quantizer_dropout: float = 0.0,
        bypass_init_prob: float = 0.0,
        bypass_final_prob: float = 0.0,
        bypass_anneal_steps: int = 10000,
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
            bypass_init_prob (float, optional): Initial probability of bypassing
                quantization entirely (output = input). Default is 0.0.
            bypass_final_prob (float, optional): Final probability of bypassing
                quantization entirely (output = input). Default is 0.0.
            bypass_anneal_steps (int, optional): Number of steps over which to
                anneal the bypass probability from `bypass_init_prob` to
                `bypass_final_prob`. Default is 10k.
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
        kwargs["loss_reduction"] = "none"
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
        assert 0.0 <= quantizer_dropout <= 1.0, "quantizer_dropout must be in [0, 1]"
        self.bypass_anneal_steps = bypass_anneal_steps
        assert 0.0 <= bypass_init_prob <= 1.0, "bypass_init_prob must be in [0, 1]"
        assert 0.0 <= bypass_final_prob < 1.0, "bypass_final_prob must be in [0, 1)"
        self.register_buffer("bypass_prob", torch.tensor(bypass_init_prob))
        self.bypass_final_prob = bypass_final_prob

    @torch.no_grad()
    def update_bypass_prob(self, global_step: int):
        """
        Anneal the bypass probability linearly from `bypass_init_prob` to
        `bypass_final_prob` over `bypass_anneal_steps`.

        Args:
            global_step (int, optional): Current training step. If None,
                uses the module's internal `_step` counter.
        """
        if self.bypass_final_prob == self.bypass_prob.item():
            return

        if self.bypass_anneal_steps > 0 and global_step < self.bypass_anneal_steps:
            new_prob = self.bypass_prob.item() + (
                (self.bypass_final_prob - self.bypass_prob.item())
                * (global_step / self.bypass_anneal_steps)
            )
            self.bypass_prob.fill_(new_prob)
        else:
            self.bypass_prob.fill_(self.bypass_final_prob)

    @torch.no_grad()
    def update_hyperparams(self, global_step: int):
        """
        Update any internal parameters, e.g., for annealing.

        Args:
            global_step (int): Current training step.
        """
        self.update_bypass_prob(global_step)
        for quantizer in self.quantizers:
            quantizer.update_hyperparams(global_step)

    def __str__(self):
        s = f"{self.__class__.__name__}({self.in_feats}, {self.num_quantizers}, {self.codebook_sizes}, {self.codebook_dims},  base_vq_type={self.base_vq_type}, quantizer_dropout={self.quantizer_dropout}), bypass_prob={self.bypass_prob.item():.4f}->{self.bypass_final_prob} over {self.bypass_anneal_steps} steps)"
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
            "bypass_init_prob": self.bypass_prob.item(),
            "bypass_final_prob": self.bypass_final_prob,
            "bypass_anneal_steps": self.bypass_anneal_steps,
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
        compute_diversity_loss = self.quantizers[0]._compute_diversity_loss
        if compute_diversity_loss:
            diversity_loss = 0.0
        else:
            diversity_loss = None

        compute_orthogonality_loss = self.quantizers[0]._compute_orthogonality_loss
        if compute_orthogonality_loss:
            orthogonality_loss = 0.0
        else:
            orthogonality_loss = None
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

            if compute_diversity_loss:
                diversity_loss_i = vq_output.diversity_loss

            if compute_orthogonality_loss:
                orthogonality_loss_i = vq_output.orthogonality_loss

            commitment_loss += commitment_loss_i.mean()
            codebook_loss += codebook_loss_i.mean()
            if compute_diversity_loss:
                diversity_loss += diversity_loss_i.mean()
            if compute_orthogonality_loss:
                orthogonality_loss += orthogonality_loss_i.mean()

            residual = residual - z_q_i.detach()
            ppl.append(vq_output.perplexity)
            if return_codes:
                codes.append(vq_output.codes)

        if return_codes:
            codes = torch.stack(codes, dim=1)
        else:
            codes = None

        ppl = torch.stack(ppl, dim=0)

        if self.training and self.bypass_prob.item() > 0.0:
            # During training, randomly bypass quantization entirely with prob p
            B = z.shape[0]
            num_bypass = int(round(B * self.bypass_prob.item()))
            if num_bypass > 0:
                with torch.no_grad():
                    bypass_mask = torch.zeros(B, dtype=torch.bool, device=z.device)
                    bypass_mask[:num_bypass] = 1
                    bypass_mask = bypass_mask[torch.randperm(B)].view(
                        -1, *([1] * (z_q.dim() - 1))
                    )

                z_q = z * bypass_mask + z_q * (~bypass_mask)

        output = VectorQuantizerOutput(
            z_q=z_q,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            diversity_loss=diversity_loss,
            orthogonality_loss=orthogonality_loss,
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
        parser.add_argument(
            "--bypass-init-prob",
            type=float,
            default=0.0,
            help="Initial probability of bypassing quantization entirely (output = input)",
        )
        parser.add_argument(
            "--bypass-final-prob",
            type=float,
            default=0.0,
            help="Final probability of bypassing quantization entirely (output = input)",
        )
        parser.add_argument(
            "--bypass-anneal-steps",
            type=int,
            default=10000,
            help="Number of steps over which to anneal the bypass probability "
            "from `bypass_init_prob` to `bypass_final_prob`",
        )

        skip = skip | {"vq_type", "codebook_size", "codebook_dim", "loss_reduction"}
        VectorQuantizerFactory.add_class_args(parser, skip=skip)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
