"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from hyperion.utils.misc import filter_func_args

from .vq import BinarySplittingGMMVectorQuantizer, VectorQuantizerOutput, VQDistanceType
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
        latent_dim (int, optional): Optional latent dimension for the RVQ stack.
            If set, inputs are projected to this dimension before quantization.
        num_quantizers (int): Number of residual stages (M).
        codebook_sizes (List[int]): List of codebook sizes (K per stage).
        codebook_dims (List[Optional[int]]): List of codebook dimensions (D per stage).
        base_vq_type (str): Type of vector quantizer used in each stage.
        quantizer_dropout (float): Fraction of examples using a truncated depth K<M
            in training (applied deterministically to the first ⌊B·p⌋ examples).
        quantizer_grad_frac (float): Fraction of gradient passed through the
            residual update at each stage (0.0 stops, 1.0 full).
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
        latent_dim: Optional[int] = None,
        quantizer_dropout: float = 0.0,
        quantizer_grad_frac: float = 0.0,
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
            latent_dim (int, optional): Optional latent dimension for the RVQ stack.
                If provided, inputs are projected to `latent_dim` before quantization.
            quantizer_dropout (float, optional): Fraction ∈ [0,1]. During training,
                a proportion of examples (⌊B·p⌋) will use fewer than M stages,
                with their depth sampled uniformly from 1..M.
            quantizer_grad_frac (float, optional): Fraction ∈ [0,1] of gradient
                passed through the residual update (0.0 detaches, 1.0 full).
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

        if latent_dim is None:
            latent_dim = in_feats

        self.latent_dim = latent_dim
        if self.in_feats != self.latent_dim:
            self.in_proj = nn.Linear(in_feats, self.latent_dim)
            self.out_proj = nn.Linear(self.latent_dim, in_feats)
            use_weight_norm = (
                kwargs["use_weight_norm"] if "use_weight_norm" in kwargs else False
            )
            if use_weight_norm:
                self.in_proj = weight_norm(self.in_proj, name="weight")
                self.out_proj = weight_norm(self.out_proj, name="weight")
        else:
            self.in_proj = None
            self.out_proj = None

        self.codebook_sizes = codebook_sizes
        self.codebook_dims = codebook_dims
        self.base_vq_type = base_vq_type
        kwargs["losses_reduction"] = "none"
        quantizers = []
        split_start_steps = kwargs.pop("split_start_steps", 0)
        for i in range(num_quantizers):
            quantizer = VectorQuantizerFactory.create(
                base_vq_type,
                latent_dim,
                codebook_sizes[i],
                codebook_dims[i],
                split_start_steps=split_start_steps,
                **kwargs,
            )
            if isinstance(quantizer, BinarySplittingGMMVectorQuantizer):
                split_start_steps = quantizer.total_split_steps()

            quantizers.append(quantizer)

        self.quantizers = nn.ModuleList(quantizers)

        self.quantizer_dropout = quantizer_dropout
        self.quantizer_grad_frac = float(quantizer_grad_frac)
        assert 0.0 <= quantizer_dropout <= 1.0, "quantizer_dropout must be in [0, 1]"
        assert (
            0.0 <= self.quantizer_grad_frac <= 1.0
        ), "quantizer_grad_frac must be in [0, 1]"
        self.bypass_anneal_steps = bypass_anneal_steps
        assert 0.0 <= bypass_init_prob <= 1.0, "bypass_init_prob must be in [0, 1]"
        assert 0.0 <= bypass_final_prob < 1.0, "bypass_final_prob must be in [0, 1)"
        self.register_buffer("bypass_prob", torch.tensor(bypass_init_prob))
        self.bypass_final_prob = bypass_final_prob

    def change_config(
        self,
        quantizer_dropout: float = 0.0,
        quantizer_grad_frac: float = 0.0,
        **kwargs,
    ):
        """
        Change internal configuration of the RVQ.

        Args:
            quantizer_dropout (float, optional): Fraction ∈ [0,1]. During training,
                a proportion of examples (⌊B·p⌋) will use fewer than M stages,
                with their depth sampled uniformly from 1..M.
            quantizer_grad_frac (float, optional): Fraction ∈ [0,1] of gradient
                passed through the residual update (0.0 detaches, 1.0 full).
            **kwargs: Extra keyword arguments passed to
                :func:`VectorQuantizer.change_config` for each stage.
        """
        logging.info(
            "Changing RVQ config with quantizer_dropout=%f, quantizer_grad_frac=%f",
            quantizer_dropout,
            quantizer_grad_frac,
        )
        self.quantizer_dropout = quantizer_dropout
        self.quantizer_grad_frac = float(quantizer_grad_frac)
        assert 0.0 <= quantizer_dropout <= 1.0, "quantizer_dropout must be in [0, 1]"
        assert (
            0.0 <= self.quantizer_grad_frac <= 1.0
        ), "quantizer_grad_frac must be in [0, 1]"

        kwargs = filter_func_args(self.quantizers[0].change_config, kwargs)
        for quantizer in self.quantizers:
            quantizer.change_config(**kwargs)

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
            "latent_dim": self.latent_dim,
            "codebook_sizes": self.codebook_sizes,
            "codebook_dims": self.codebook_dims,
            "quantizer_dropout": self.quantizer_dropout,
            "quantizer_grad_frac": self.quantizer_grad_frac,
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

    @torch.no_grad()
    def init_from_rvq(
        self, rvq: "ResidualVectorQuantizer"
    ) -> "ResidualVectorQuantizer":
        """Initialize this RVQ from another RVQ instance in-place.

        The shared prefix of quantizers is initialized by calling
        ``init_from_vq`` on each target quantizer:
        - if ``self`` has fewer levels than ``rvq``, only the first levels in
          ``self`` are initialized.
        - if ``self`` has more levels than ``rvq``, the extra tail levels in
          ``self`` are kept as originally initialized.

        Args:
            rvq: Source residual vector quantizer.

        Returns:
            self, initialized from ``rvq``.
        """
        if not isinstance(rvq, ResidualVectorQuantizer):
            raise TypeError(
                f"`rvq` must be a ResidualVectorQuantizer, got {type(rvq).__name__}"
            )

        if self.in_feats != rvq.in_feats:
            raise ValueError(
                "Cannot init_from_rvq with different in_feats: "
                f"target={self.in_feats}, source={rvq.in_feats}"
            )

        if self.latent_dim != rvq.latent_dim:
            raise ValueError(
                "Cannot init_from_rvq with different latent_dim: "
                f"target={self.latent_dim}, source={rvq.latent_dim}"
            )

        for proj_name in ("in_proj", "out_proj"):
            self_proj = getattr(self, proj_name, None)
            src_proj = getattr(rvq, proj_name, None)
            if (self_proj is None) != (src_proj is None):
                raise ValueError(
                    f"Cannot init_from_rvq with mismatched `{proj_name}` presence: "
                    f"target_is_none={self_proj is None}, source_is_none={src_proj is None}"
                )

        self.train(rvq.training)

        for proj_name in ("in_proj", "out_proj"):
            self_proj = getattr(self, proj_name, None)
            src_proj = getattr(rvq, proj_name, None)
            if self_proj is None:
                continue

            src_proj_state = src_proj.state_dict()
            tgt_proj_state = self_proj.state_dict()
            merged_proj_state = {}
            for key, tgt_tensor in tgt_proj_state.items():
                src_tensor = src_proj_state.get(key)
                if src_tensor is None or src_tensor.shape != tgt_tensor.shape:
                    continue
                merged_proj_state[key] = src_tensor.detach().to(
                    device=tgt_tensor.device, dtype=tgt_tensor.dtype
                )
            self_proj.load_state_dict(merged_proj_state, strict=False)

        if hasattr(self, "bypass_prob") and hasattr(rvq, "bypass_prob"):
            self.bypass_prob.copy_(rvq.bypass_prob.to(self.bypass_prob.device))

        num_shared = min(len(self.quantizers), len(rvq.quantizers))
        for i in range(num_shared):
            tgt_vq = self.quantizers[i]
            src_vq = rvq.quantizers[i]
            if not hasattr(tgt_vq, "init_from_vq"):
                raise TypeError(
                    f"Quantizer at level {i} does not implement `init_from_vq`"
                )
            tgt_vq.init_from_vq(src_vq)

        return self

    def scale_quantizer_grad(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Scale the gradient of the residual by a fixed fraction.

        Args:
            z_q (Tensor): The quantized tensor.

        Returns:
            Tensor: The quantized tensor with scaled gradient.
        """
        if self.quantizer_grad_frac == 0.0:
            return z_q.detach()
        else:
            return z_q * self.quantizer_grad_frac + z_q.detach() * (
                1.0 - self.quantizer_grad_frac
            )

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode residual VQ codes back to quantized latents.

        Args:
            codes: Code indices with shape (B, M, ...), where M is the number of
                residual quantizers represented in the tensor.

        Returns:
            Quantized latents with shape (B, ..., D_in).
        """
        if codes.dim() < 2:
            raise ValueError(
                f"`codes` must have shape (B, M, ...), got {tuple(codes.shape)}"
            )

        num_quantizers = codes.size(1)
        if num_quantizers > self.num_quantizers:
            raise ValueError(
                f"`codes` contains {num_quantizers} quantizers, "
                f"but this RVQ only has {self.num_quantizers}"
            )

        z_q = None
        for i in range(num_quantizers):
            quantizer = self.quantizers[i]
            z_q_i = quantizer.decode_codes(codes[:, i])
            if quantizer.out_proj is not None:
                z_q_i = quantizer.out_proj(z_q_i)

            if z_q is None:
                z_q = z_q_i
            else:
                z_q = z_q + z_q_i

        if self.out_proj is not None:
            z_q = self.out_proj(z_q)

        return z_q

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
        if self.in_proj is not None:
            z = self.in_proj(z)

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

            residual = residual - self.scale_quantizer_grad(z_q_i)
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

        if self.out_proj is not None:
            z_q = self.out_proj(z_q)

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
            --quantizer-grad-frac (float): Fraction of gradient passed through
                the residual update (0.0 detaches, 1.0 full).
            --base-vq-type (str): Quantizer type for each stage.
            --latent-dim (int): Optional latent dimension for the RVQ stack.
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
            default=2,
            help="Number of quantizers (M) to use in the residual vector quantizer",
        )
        parser.add_argument(
            "--quantizer-dropout",
            type=float,
            default=0.0,
            help="Probability of dropping each quantizer during training for regularization",
        )
        parser.add_argument(
            "--quantizer-grad-frac",
            type=float,
            default=0.0,
            help="Fraction of gradient passed through the residual update (0.0 detaches, 1.0 full)",
        )
        parser.add_argument(
            "--base-vq-type",
            choices=list(vq_dict.keys()),
            default="nn_vq",
            help="Type of vector quantizer to use in each stage",
        )
        parser.add_argument(
            "--latent-dim",
            type=int,
            default=None,
            help="Optional latent dimension for the RVQ stack (projects in_feats before quantization)",
        )
        parser.add_argument(
            "--codebook-sizes",
            type=int,
            nargs="+",
            default=[1024],
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

        skip = skip | {"vq_type", "codebook_size", "codebook_dim", "losses_reduction"}
        VectorQuantizerFactory.add_class_args(parser, skip=skip)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
