"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from ...utils import HypDataClass
from ...utils.misc import filter_func_args
from ..utils import seq_lengths_to_mask


class VQDistanceType(str, Enum):
    L2 = "l2"
    L1 = "l1"
    COSINE = "cosine"

    @staticmethod
    def choices():
        return [o.value for o in VQDistanceType]


@dataclass
class VectorQuantizerOutput(HypDataClass):
    """Output of vector quantization layers."""

    z_q: torch.Tensor = None  # Quantized vectors
    codebook_loss: torch.Tensor = None  # VQ loss
    commitment_loss: torch.Tensor = (
        None  # Commitment loss (shape depends on `losses_reduction`)
    )
    diversity_loss: Optional[torch.Tensor] = None  # Diversity regularizer (optional)
    orthogonality_loss: Optional[torch.Tensor] = (
        None  # Orthonormality regularizer (optional)
    )
    perplexity: Optional[torch.Tensor] = None  # Perplexity of the responsibilities
    codes: Optional[torch.Tensor] = None  # indices of the codebook vectors (optional)
    z_mask: Optional[torch.Tensor] = None  # mask used for quantization (optional)
    z_lengths: Optional[torch.Tensor] = None  # lengths used for quantization (optional)
    extras: Optional[Dict[str, Any]] = None  # any extra info


class VectorQuantizerBase(nn.Module):
    """
    Abstract vector quantization layer.

    This module maintains a discrete codebook of embedding vectors and quantizes
    continuous input features.
    It supports inputs of rank 2–5 (e.g., [B,D], [B,T,D], [B,C,H,W], …) and can
    handle variable-length sequences or masks.

    Attributes:
        in_feats (int): Input feature dimension.
        codebook_size (int): Number of embedding vectors (codebook size).
        codebook_dim (int, optional): Dimensionality of embedding vectors. Defaults
            to `in_feats`. If different, an input/output projection is learned.
        distance_metric (VQDistanceType): Distance metric for nearest-neighbor
            search (L2, L1, cosine).
        use_weight_norm (bool): If True, applies weight normalization to projection
            layers.
        channels_last (bool): If False, expects channel-first layout for >2D tensors
            (e.g., (B,C,H,W)) and internally transposes to (B,H*W,D).
        temp (Tensor): Current Gumbel-Softmax temperature buffer.
        temp_min (Tensor): Lower bound for temperature annealing.
        temp_anneal_rate (float): Exponential decay rate for temperature scheduling.
        commitment_weight (Tensor): Current scaling factor for the commitment loss.
        commitment_anneal_steps (int): Number of steps to linearly ramp the
            commitment weight from 0 to 1.
        compute_diversity_loss (bool): Whether to compute diversity loss terms.
        compute_orthogonality_loss (bool): Whether to compute orthogonality loss terms.
    """

    def __init__(
        self,
        in_feats: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        distance_metric: VQDistanceType = VQDistanceType.L2,
        use_weight_norm: bool = False,
        channels_last: bool = False,
        is_ema: bool = False,
        temp_init: float = 1.0,
        temp_min: Optional[float] = 0.05,
        temp_anneal_rate: float = 1e-5,
        temp_anneal_steps: Optional[int] = None,
        commitment_anneal_steps: int = 0,
        compute_diversity_loss: bool = False,
        compute_orthogonality_loss: bool = False,
        losses_reduction: str = "mean",
    ):
        super().__init__()
        self.in_feats = in_feats
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim if codebook_dim is not None else in_feats
        self.distance_metric = distance_metric
        self.use_weight_norm = use_weight_norm
        self.channels_last = channels_last
        if self.in_feats != self.codebook_dim:
            self.in_proj = nn.Linear(in_feats, self.codebook_dim)
            self.out_proj = nn.Linear(self.codebook_dim, in_feats)
            if use_weight_norm:
                self.in_proj = weight_norm(self.in_proj, name="weight")
                self.out_proj = weight_norm(self.out_proj, name="weight")
        else:
            self.in_proj = None
            self.out_proj = None

        W = torch.empty(codebook_size, self.codebook_dim)
        nn.init.uniform_(
            W, -1.0 / math.sqrt(self.codebook_dim), 1.0 / math.sqrt(self.codebook_dim)
        )
        # nn.init.normal_(W, 0.0, 1.0 / math.sqrt(self.codebook_dim))
        # nn.init.normal_(W, 0.0, 1.0)
        if is_ema:
            self.register_buffer("codebook", W)  # <- buffer, not Parameter
        else:
            self.codebook = nn.Parameter(W)  # <- Parameter

        if temp_min is None:
            temp_min = temp_init
        if temp_min is None:
            raise ValueError("`temp_min` must be provided for temperature scheduling.")
        self.temp_init = float(temp_init)
        self.register_buffer("temp", torch.tensor(float(temp_init)))
        self.register_buffer("temp_min", torch.tensor(float(temp_min)))
        self.temp_anneal_steps = (
            int(temp_anneal_steps) if temp_anneal_steps is not None else None
        )
        if self.temp_anneal_steps is not None and self.temp_anneal_steps > 0:
            target_ratio = float(self.temp_min) / max(self.temp_init, 1e-12)
            # Positive rate such that temp reaches temp_min after temp_anneal_steps.
            temp_anneal_rate = -math.log(target_ratio) / float(self.temp_anneal_steps)
        # Track last global step used for temperature scheduling so finetuning
        # resumes decay from the current temp instead of resetting to temp_init.
        self.register_buffer("temp_last_step", torch.tensor(0, dtype=torch.long))
        self.temp_anneal_rate = float(temp_anneal_rate)
        self.commitment_anneal_steps = max(0, int(commitment_anneal_steps))
        initial_commitment_weight = 0.0 if self.commitment_anneal_steps > 0 else 1.0
        self.register_buffer(
            "commitment_weight", torch.tensor(float(initial_commitment_weight))
        )
        self._compute_diversity_loss = bool(compute_diversity_loss)
        self._compute_orthogonality_loss = bool(compute_orthogonality_loss)
        if losses_reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"losses_reduction must be one of {{'none','mean','sum'}}, got {losses_reduction}"
            )
        self.losses_reduction = losses_reduction

        self.init_weights()

    def get_config(self):
        """Returns the configuration of the vector quantizer."""
        cfg = {
            "in_feats": self.in_feats,
            "codebook_size": self.codebook_size,
            "codebook_dim": self.codebook_dim,
            "distance_metric": self.distance_metric.value,
            "use_weight_norm": self.use_weight_norm,
            "channels_last": self.channels_last,
            "temp_init": float(self.temp_init),
            "temp_min": float(self.temp_min.item()),
            "temp_anneal_rate": float(self.temp_anneal_rate),
            "temp_anneal_steps": self.temp_anneal_steps,
            "commitment_anneal_steps": self.commitment_anneal_steps,
            "compute_diversity_loss": self._compute_diversity_loss,
            "compute_orthogonality_loss": self._compute_orthogonality_loss,
            "losses_reduction": self.losses_reduction,
        }
        return cfg

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"{self.__class__.__name__}(in_feats={self.in_feats}, "
            f"codebook_size={self.codebook_size}, codebook_dim={self.codebook_dim}, "
            f"distance_metric={self.distance_metric}, use_weight_norm={self.use_weight_norm}, "
            f"channels_last={self.channels_last}, temp={self.temp.item():.4f}, "
            f"temp_min={self.temp_min.item():.4f}, temp_anneal_rate={self.temp_anneal_rate}, "
            f"commitment_weight={self.commitment_weight.item():.4f}, "
            f"commitment_anneal_steps={self.commitment_anneal_steps}, "
            f"compute_diversity_loss={self._compute_diversity_loss}, "
            f"compute_orthogonality_loss={self._compute_orthogonality_loss}, "
            f"losses_reduction='{self.losses_reduction}')"
        )

    def init_weights(self) -> None:
        """Reset linear layers to N(0, 0.01) weights and zero bias.

        When weight normalization is active, initialize the underlying
        parametrized weight directly so the normalized weight has the
        desired scale.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # If parametrized (e.g., weight_norm), init the original weight
                if (
                    is_parametrized(m)
                    and hasattr(m, "parametrizations")
                    and "weight" in m.parametrizations
                ):
                    g = m.parametrizations.weight.original0
                    v = m.parametrizations.weight.original1
                    nn.init.normal_(v, 0.0, 0.01)
                    with torch.no_grad():
                        g.copy_(v.flatten(1).norm(dim=1, keepdim=True).view_as(g))
                else:
                    w = m.weight
                    nn.init.normal_(w, 0.0, 0.01)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def update_temp(self, global_step: int) -> None:
        """
        Anneal temperature exponentially for quantizers that enable it.

        Update rule:
            T_t = max(temp_min, T_{t-1} * exp(-temp_anneal_rate * delta_step))
        If temp_anneal_steps is set, temp_anneal_rate is chosen so that
        temp reaches temp_min after that many steps (from temp_init).

        Args:
            global_step (int): Number of elapsed training steps.
        """
        if not isinstance(self.temp, torch.Tensor) or not isinstance(
            self.temp_min, torch.Tensor
        ):
            return
        # Only decay by the steps elapsed since the last call so resumed/finetune
        # runs keep decreasing from the current temperature rather than jumping
        # back to temp_init.
        delta_step = max(0, int(global_step) - int(self.temp_last_step.item()))
        if delta_step == 0:
            return

        decay = math.exp(-float(self.temp_anneal_rate) * float(delta_step))
        new_temp = max(self.temp_min.item(), float(self.temp) * decay)
        if math.isclose(new_temp, self.temp.item()):
            return

        self.temp.fill_(new_temp)
        self.temp_last_step.fill_(global_step)
        if global_step % 1000 == 0:
            logging.info(
                f"VQ Temperature updated to {self.temp.item():.4f} at step={global_step}"
            )

    @torch.no_grad()
    def update_commitment_weight(self, global_step: int) -> None:
        """
        Linearly anneal the commitment loss weight from 0 to 1.

        Args:
            global_step (int): Current global training step.
        """
        if self.commitment_weight == 1.0:
            return

        if self.commitment_anneal_steps <= 0:
            self.commitment_weight.fill_(1.0)
            return

        progress = min(1.0, float(global_step) / float(self.commitment_anneal_steps))
        if progress == self.commitment_weight.item():
            return
        self.commitment_weight.fill_(progress)
        if global_step % 1000 == 0:
            logging.info(
                f"VQ Commitment weight updated to {self.commitment_weight.item():.4f}"
            )

    def update_sampling_mode(self, global_step: int) -> None:
        """
        Hook for subclasses that need to schedule sampling behaviour.

        Args:
            global_step (int): Current global training step.
        """
        return

    def update_params(self, global_step: int) -> None:
        """
        Update all scheduled parameters.

        Args:
            global_step (int): Current global training step.
        """
        self.update_temp(global_step)
        self.update_commitment_weight(global_step)
        self.update_sampling_mode(global_step)

    def compute_diversity_loss(
        self,
        posterior: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """
        Compute a diversity loss encouraging uniform code usage.

        Args:
            posterior (Tensor): Posterior code probabilities of shape (..., K).
            mask (Tensor, optional): Optional mask broadcastable to posterior[..., 0].
            eps (float): Numerical stability constant.

        Returns:
            Tensor: Scalar loss (KL(p || uniform)) in posterior.dtype on posterior.device.
        """
        if posterior.numel() == 0:
            return posterior.new_zeros(())

        probs = posterior.view(-1, posterior.shape[-1])
        if mask is not None:
            mask_flat = mask.view(-1).to(probs.dtype)
            weights = mask_flat.unsqueeze(-1)
            weighted = probs * weights
            denom = mask_flat.sum()
            if denom.item() <= 0:
                return posterior.new_zeros(())
            avg_prob = weighted.sum(dim=0) / denom.clamp_min(eps)
        else:
            if probs.shape[0] == 0:
                return posterior.new_zeros(())
            avg_prob = probs.mean(dim=0)

        total_mass = avg_prob.sum().clamp_min(eps)
        avg_prob = avg_prob / total_mass

        num_codes = avg_prob.shape[0]
        if num_codes == 0:
            return posterior.new_zeros(())
        log_k = posterior.new_tensor(math.log(num_codes))
        loss = (avg_prob * (avg_prob + eps).log()).sum()
        return loss + log_k

    def compute_orthogonal_loss(self) -> torch.Tensor:
        """
        Compute orthogonality loss for the current codebook.

        Returns:
            Tensor: Scalar loss measuring deviation from orthonormality.
        """
        normed = F.normalize(self.codebook, p=2, dim=-1)
        gram = torch.matmul(normed, normed.t())
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return F.mse_loss(gram, identity)

    @staticmethod
    def _maybe_allreduce_tensor(x: torch.Tensor) -> None:
        """
        Sum `x` across processes if torch.distributed is active.
        """
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM)

    def _reduce_loss(self, loss: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Reduce a loss tensor according to the configured reduction mode.

        Args:
            loss (Tensor or None): Loss values to reduce.

        Returns:
            Tensor or None: Reduced loss respecting ``losses_reduction``.
        """
        if loss is None:
            return None
        if self.losses_reduction == "none":
            return loss
        if self.losses_reduction == "mean":
            return loss.mean()
        if self.losses_reduction == "sum":
            return loss.sum()
        raise RuntimeError(
            f"Unexpected losses_reduction value: {self.losses_reduction}"
        )

    @torch.no_grad()
    def codebook_perplexity_hard(
        self,
        codes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-10,
    ):
        """
        Compute codebook perplexity from hard code indices.

        Args:
            codes (Tensor): Long tensor of shape (B, ...) containing integer code
                codes in [0, codebook_size-1].
            mask (Tensor, optional): Boolean or {0,1} mask of the same shape as
                `codes`. Only unmasked entries are counted.
            eps (float): Numerical stability constant.

        Returns:
            Tensor: Scalar perplexity (effective number of used codes).
        """
        # Flatten codes
        flat_codes = codes.view(-1)
        if mask is not None:
            flat_mask = mask.view(-1).bool()
            flat_codes = flat_codes[flat_mask]

        # Histogram over all codes
        counts = torch.bincount(flat_codes, minlength=self.codebook_size).to(
            device=self.codebook.device, dtype=torch.float32
        )
        self._maybe_allreduce_tensor(counts)
        total = counts.sum()
        if total <= 0:
            return torch.tensor(0.0, device=counts.device)
        probs = counts / total

        # Entropy
        entropy = -(probs * (probs + eps).log()).sum()

        # Perplexity
        ppl = entropy.exp()
        return ppl

    @torch.no_grad()
    def codebook_perplexity_soft(
        self,
        soft_one_hot: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """
        Compute perplexity from soft assignment probabilities.

        Args:
            soft_one_hot (Tensor): Soft assignments of shape (N, codebook_size).
            mask (Tensor, optional): Boolean mask selecting active positions.
            eps (float): Numerical stability constant.

        Returns:
            Tensor: Scalar perplexity.
        """
        # Flatten codes
        if mask is not None:
            flat_mask = mask.view(-1).bool()
            soft_one_hot = soft_one_hot[flat_mask]

        # Histogram over all codes
        counts = soft_one_hot.sum(dim=0).to(dtype=torch.float32)  # (num_embed,)
        self._maybe_allreduce_tensor(counts)
        total = counts.sum()
        if total <= 0:
            return torch.tensor(0.0, device=counts.device)

        # Convert to probabilities
        probs = counts / total

        # Entropy
        entropy = -(probs * (probs + eps).log()).sum()

        # Perplexity
        ppl = entropy.exp()
        return ppl

    def compute_codebook_distance(self, encodings: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between flattened encodings and codebook entries.

        Args:
            encodings (Tensor): Input of shape (N, D), where D = `codebook_dim`.

        Returns:
            Tensor: Distance matrix of shape (N, codebook_size).
        """
        codebook = self.codebook  # codebook: (N x D)
        if self.distance_metric == VQDistanceType.L2:
            return (
                torch.sum(encodings**2, dim=1, keepdim=True)
                + torch.sum(codebook**2, dim=1)
                - 2 * torch.matmul(encodings, codebook.t())
            )
        elif self.distance_metric == VQDistanceType.L1:
            return torch.cdist(encodings, codebook, p=1)
        elif self.distance_metric == VQDistanceType.COSINE:
            encodings = F.normalize(encodings, p=2, dim=-1)
            codebook = F.normalize(codebook, p=2, dim=-1)
            return 1 - torch.matmul(encodings, codebook.t())
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete codes back to embedding vectors.

        Args:
            codes (Tensor): Long tensor of shape (B, T) or (B, ...) with code indices.

        Returns:
            Tensor: Codebook embeddings of shape (B, T, D).
        """
        return F.embedding(codes, self.codebook)  # (B,T,D)

    def encode_latents(
        self, latents: torch.Tensor, return_probs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode continuous latents to nearest code indices.

        Args:
            latents (Tensor): Input tensor of shape (B, T, D).
            return_probs (bool): When ``True`` also returns soft assignment
                probabilities (``shape==(B*T, K)``) for diversity regularizers.

        Returns:
            Tensor | Tuple[Tensor, Tensor]:
                Either the code indices of shape ``(B, T)`` or a tuple
                ``(codes, probabilities)`` when ``return_probs`` is enabled.
        """
        latents_shape = latents.shape
        latents = latents.view(-1, latents.shape[-1])  # (B*T, D)
        distance = self.compute_codebook_distance(latents)
        # print("distance", distance, flush=True)
        codes = distance.min(dim=1)[1]  # (B*T)
        unique_codes = torch.unique(codes)
        print(f"[encode_latents] Unique codes={unique_codes.tolist()}")
        if unique_codes.numel() < 4:
            torch.set_printoptions(threshold=10_000)
            T = latents_shape[1] // 3
            print(f"[encode_latents] latents:\n", latents[T : T + 20, :20])
            print(f"[encode_latents] distance:\n", distance[T : T + 20])
            for uc in unique_codes.tolist():
                print(f"[encode_latents] distance[:,{uc}]:\n", distance[T : T + 20, uc])
            for uc in unique_codes.tolist():
                print(f"[encode_latents] codebook[{uc}]:\n", self.codebook[uc, :20])
        # print(
        #     "codes1",
        #     codes,
        #     latents[0],
        #     self.codebook[codes[0]],
        #     latents[-1],
        #     self.codebook[codes[-1]],
        #     flush=True,
        # )
        codes = codes.view(latents_shape[0], -1)  # (B, T)
        # print("codes2", codes, flush=True)
        if return_probs:
            logits = -distance / self.temp
            probs = F.softmax(logits, dim=-1)  # (B*T, K)
            return codes, probs

        return codes

    def quantize_latents(
        self, latents: torch.Tensor, return_probs: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Quantize continuous latents by nearest-neighbor lookup.

        Args:
            latents (Tensor): Input tensor of shape (B, T, D).
            return_probs (bool): When ``True`` also returns the soft assignments
                produced in :meth:`encode_latents`.

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor]]:
                Quantized tensor ``(B, T, D)``, code indices ``(B, T)``, and
                optionally the flattened soft responsibilities ``(B*T, K)``.
        """
        if return_probs:
            codes, probs = self.encode_latents(latents, return_probs=True)
        else:
            codes = self.encode_latents(latents)
            probs = None

        z_q = self.decode_codes(codes)  # (B, T, D)

        return z_q, codes, probs

    def reshape_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        """
        Flatten input into (B, T, D) while preserving original shape.

        Args:
            x (Tensor): Input of rank 2–5, e.g. (B,D), (B,T,D), (B,C,H,W).

        Returns:
            (Tensor, tuple):
                - Reshaped tensor (B, T, D).
                - Original shape tuple for restoration.
        """
        orig_shape = x.shape
        if x.dim() == 2:  # (B,D)
            x = x.unsqueeze(1)  # (B,1,D)
            return x, orig_shape  # keep the original

        if not self.channels_last:
            x = x.movedim(1, -1).contiguous()  # e.g., (B,C,H,W)->(B,H,W,C)

        x = x.contiguous().view(x.shape[0], -1, x.shape[-1])
        # x = (B, T, D) or (B, HW, D)
        return x, orig_shape

    def reshape_output(
        self,
        y: torch.Tensor,
        codes: Optional[torch.Tensor],
        orig_shape: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Restore quantized output and codes to original input shape.

        Args:
            y (Tensor): Quantized tensor of shape (B, T, D).
            codes (Tensor, optional): codes of shape (B, T).
            orig_shape (tuple): Original input shape.

        Returns:
            (Tensor, Tensor): Restored tensors with original shape.
        """
        # y is (B, T, D) or (B,1,D)
        if len(orig_shape) == 2:  # was (B,D)
            y = y.squeeze(1)  # (B,D)
            if codes is not None:
                codes = codes.squeeze(1)  # (B,)
            return y, codes

        # Restore T back to spatial dims first
        if self.channels_last:
            y = y.contiguous().view(
                *orig_shape[:-1], y.shape[-1]
            )  # (B, ..., D) already channels-last
        else:
            # We originally did movedim(1, -1): (B,C,...) -> (B,...,C)
            # So here we first rebuild (B, ..., C), then move C back to dim=1
            y = y.view(
                *orig_shape[0:1], *orig_shape[2:], orig_shape[1]
            )  # (B, H, W, ..., C)
            y = y.movedim(-1, 1).contiguous()  # (B, C, H, W, ...)

        if codes is not None:
            # drop trailing feature dim in codes
            if self.channels_last:
                codes = codes.view(*orig_shape[:-1])
            else:
                codes = codes.view(y.shape[0], *orig_shape[2:])

        return y, codes

    def remove_weight_norm(self) -> None:
        """
        Remove weight normalization from all layers that have it.
        """
        logging.info("Removing weight norm...")
        for m in self.modules():
            if is_parametrized(m):
                remove_parametrizations(m, "weight")


class _GumbelSoftSamplingMixin:
    """
    Mixin that provides soft-to-hard sampling scheduling for Gumbel quantizers.
    """

    soft_sampling_steps: int

    def _init_soft_sampling(self, soft_sampling_steps: int) -> None:
        steps = 0 if soft_sampling_steps is None else int(soft_sampling_steps)
        self.soft_sampling_steps = max(0, steps)
        initial_hard = self.soft_sampling_steps == 0
        self.register_buffer(
            "use_hard_sampling",
            torch.tensor(initial_hard, dtype=torch.bool),
        )

    def _resolve_sampling_mode(self, hard: Optional[bool]) -> bool:
        if hard is not None:
            return bool(hard)
        if not self.training:
            return True
        return bool(self.use_hard_sampling.item())

    @torch.no_grad()
    def update_sampling_mode(self, global_step: int) -> None:
        super().update_sampling_mode(global_step)
        if not hasattr(self, "soft_sampling_steps"):
            return

        if self.soft_sampling_steps == 0:
            target_hard = True
        else:
            target_hard = global_step >= self.soft_sampling_steps

        current_hard = bool(self.use_hard_sampling.item())
        if current_hard != target_hard:
            mode = "hard" if target_hard else "soft"
            logging.info(
                "%s switching to %s Gumbel sampling at step %d",
                self.__class__.__name__,
                mode,
                int(global_step),
            )
        self.use_hard_sampling.fill_(target_hard)


class _GDVectorQuantizer(VectorQuantizerBase):
    """Intermediate base class for gradient-descent vector quantizers."""

    def __init__(
        self,
        in_feats: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        distance_metric: VQDistanceType = VQDistanceType.L2,
        use_weight_norm: bool = False,
        channels_last: bool = False,
        reset_unused: bool = False,
        reset_unused_steps: int = 1,
        temp_init: float = 1.0,
        temp_min: Optional[float] = 0.5,
        temp_anneal_rate: float = 1e-5,
        commitment_anneal_steps: int = 0,
        compute_diversity_loss: bool = False,
        compute_orthogonality_loss: bool = False,
        losses_reduction: str = "mean",
    ):
        super().__init__(
            in_feats,
            codebook_size,
            codebook_dim,
            distance_metric,
            use_weight_norm,
            channels_last,
            is_ema=False,
            temp_init=temp_init,
            temp_min=temp_min,
            temp_anneal_rate=temp_anneal_rate,
            commitment_anneal_steps=commitment_anneal_steps,
            compute_diversity_loss=compute_diversity_loss,
            compute_orthogonality_loss=compute_orthogonality_loss,
            losses_reduction=losses_reduction,
        )
        self.reset_unused = reset_unused
        self.reset_unused_steps = max(1, int(reset_unused_steps)) if reset_unused else 0
        self.register_buffer(
            "unused_steps",
            torch.zeros(
                self.codebook_size,
                dtype=torch.long,
                device=self.codebook.device,
            ),
            persistent=reset_unused,
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "reset_unused": self.reset_unused,
                "reset_unused_steps": (
                    self.reset_unused_steps if self.reset_unused else None
                ),
            }
        )
        return cfg

    @torch.no_grad()
    def _reset_unused_codes(
        self,
        z: torch.Tensor,
        codes: torch.Tensor,
        z_mask: torch.Tensor | None,
    ) -> None:
        """Reset unused codebook entries after aggregating usage across processes."""
        flat_codes = codes.view(-1)
        flat_z = z.view(-1, z.shape[-1]).detach()
        if z_mask is not None:
            mask_flat = z_mask.view(-1)
            valid = (
                mask_flat.to(torch.bool) if mask_flat.dtype != torch.bool else mask_flat
            )
            flat_codes = flat_codes[valid]
            flat_z_valid = flat_z[valid]
        else:
            flat_z_valid = flat_z

        if flat_codes.numel() > 0:
            counts = torch.bincount(flat_codes, minlength=self.codebook_size)
        else:
            counts = torch.zeros(
                self.codebook_size,
                device=self.codebook.device,
                dtype=torch.long,
            )

        dist_ready = dist.is_available() and dist.is_initialized()
        if dist_ready:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        used = counts > 0
        self.unused_steps[used] = 0
        self.unused_steps[~used] += 1
        print("num unused", (counts == 0).sum().item(), flush=True)
        print("unused_steps", self.unused_steps[~used], flush=True)

        to_reset = self.unused_steps >= self.reset_unused_steps
        if not to_reset.any():
            return

        num_unused = int(to_reset.sum().item())
        D = flat_z.shape[-1]

        if dist_ready:
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            local_count = torch.tensor(
                [flat_z_valid.shape[0]],
                device=flat_z_valid.device,
                dtype=torch.long,
            )
            count_list = [torch.zeros_like(local_count) for _ in range(world_size)]
            dist.all_gather(count_list, local_count)
            counts_per_rank = torch.stack(count_list)
            max_valid = int(counts_per_rank.max().item())

            if max_valid > 0:
                padded = torch.zeros(
                    max_valid,
                    D,
                    device=flat_z_valid.device,
                    dtype=flat_z_valid.dtype,
                )
                if flat_z_valid.shape[0] > 0:
                    copy_len = min(max_valid, flat_z_valid.shape[0])
                    padded[:copy_len] = flat_z_valid[:copy_len]
                gathered = [torch.zeros_like(padded) for _ in range(world_size)]
                dist.all_gather(gathered, padded)
                if rank == 0:
                    candidates = torch.cat(
                        [g[: n.item()] for g, n in zip(gathered, counts_per_rank)],
                        dim=0,
                    )
                else:
                    candidates = torch.empty(
                        0,
                        D,
                        device=flat_z_valid.device,
                        dtype=flat_z_valid.dtype,
                    )
            else:
                candidates = torch.empty(
                    0,
                    D,
                    device=flat_z_valid.device,
                    dtype=flat_z_valid.dtype,
                )

            if rank == 0:
                if candidates.shape[0] > 0:
                    rand_idx = torch.randint(
                        0,
                        candidates.shape[0],
                        (num_unused,),
                        device=candidates.device,
                    )
                    new_vectors = candidates[rand_idx]
                else:
                    new_vectors = torch.randn(
                        num_unused,
                        D,
                        device=self.codebook.device,
                        dtype=self.codebook.dtype,
                    ) * (1.0 / math.sqrt(self.codebook_dim))
            else:
                new_vectors = torch.empty(
                    num_unused,
                    D,
                    device=self.codebook.device,
                    dtype=self.codebook.dtype,
                )

            dist.broadcast(new_vectors, src=0)
            if rank == 0:
                logging.info(f"Resetting {int(num_unused)} codebook entries.")
        else:
            if flat_z_valid.shape[0] > 0:
                rand_idx = torch.randint(
                    0,
                    flat_z_valid.shape[0],
                    (num_unused,),
                    device=flat_z_valid.device,
                )
                new_vectors = flat_z_valid[rand_idx]
            else:
                new_vectors = torch.randn(
                    num_unused,
                    D,
                    device=self.codebook.device,
                    dtype=self.codebook.dtype,
                ) * (1.0 / math.sqrt(self.codebook_dim))

            logging.info(f"Resetting {int(num_unused)} codebook entries.")

        new_vectors = new_vectors.to(self.codebook.dtype)
        self.codebook.data[to_reset] = new_vectors
        self.unused_steps[to_reset] = 0


class NNVectorQuantizer(_GDVectorQuantizer):
    """
    Standard (nearest-neighbor) vector quantization layer.

    This module maintains a discrete codebook of embedding vectors and quantizes
    continuous input features by nearest-neighbor lookup in the embedding space.
    It supports inputs of rank 2–5 (e.g., [B,D], [B,T,D], [B,C,H,W], …) and can
    handle variable-length sequences or masks.

    Attributes:
        in_feats (int): Input feature dimension.
        codebook_size (int): Number of embedding vectors (codebook size).
        codebook_dim (int, optional): Dimensionality of embedding vectors. Defaults
            to `in_feats`. If different, an input/output projection is learned.
        distance_metric (VQDistanceType): Distance metric for nearest-neighbor
            search (L2, L1, cosine).
        use_weight_norm (bool): If True, applies weight normalization to projection
            layers.
        channels_last (bool): If False, expects channel-first layout for >2D tensors
            (e.g., (B,C,H,W)) and internally transposes to (B,H*W,D).
        reset_unused (bool): If True (and in training mode), reinitializes codebook
            entries that remain unused for a configurable number of batches.
        reset_unused_steps (int): Number of consecutive forward passes a codeword can
            stay unused before it is reset. Only relevant if ``reset_unused`` is True.
    """

    def __init__(
        self,
        in_feats: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        distance_metric: VQDistanceType = VQDistanceType.L2,
        use_weight_norm: bool = False,
        channels_last: bool = False,
        reset_unused: bool = False,
        reset_unused_steps: int = 100,
        commitment_anneal_steps: int = 0,
        compute_diversity_loss: bool = False,
        compute_orthogonality_loss: bool = False,
        losses_reduction: str = "mean",
    ):
        super().__init__(
            in_feats,
            codebook_size,
            codebook_dim,
            distance_metric,
            use_weight_norm,
            channels_last,
            reset_unused=reset_unused,
            reset_unused_steps=reset_unused_steps,
            commitment_anneal_steps=commitment_anneal_steps,
            compute_diversity_loss=compute_diversity_loss,
            compute_orthogonality_loss=compute_orthogonality_loss,
            losses_reduction=losses_reduction,
        )

    def get_config(self):
        """Returns the configuration of the vector quantizer."""
        return super().get_config()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"{self.__class__.__name__}(in_feats={self.in_feats}, codebook_size={self.codebook_size}, "
            f"codebook_dim={self.codebook_dim}, distance_metric={self.distance_metric}, "
            f"use_weight_norm={self.use_weight_norm}, channels_last={self.channels_last}, "
            f"reset_unused={self.reset_unused}, "
            f"reset_unused_steps={self.reset_unused_steps if self.reset_unused else 'n/a'}, "
            f"temp={self.temp.item():.4f}, temp_min={self.temp_min.item():.4f}, "
            f"temp_anneal_rate={self.temp_anneal_rate}, "
            f"commitment_weight={self.commitment_weight.item():.4f}, "
            f"commitment_anneal_steps={self.commitment_anneal_steps}, "
            f"compute_diversity_loss={self._compute_diversity_loss}, "
            f"compute_orthogonality_loss={self._compute_orthogonality_loss}, "
            f"losses_reduction='{self.losses_reduction}')"
        )

    def forward(
        self,
        z: torch.Tensor,
        z_lengths: Optional[torch.Tensor] = None,
        z_mask: Optional[torch.Tensor] = None,
        return_codes: bool = False,
    ) -> VectorQuantizerOutput:
        """
        Forward pass of vector quantization.

        Args:
            z (Tensor): Input tensor, shape (B, ..., D).
            z_lengths (Tensor, optional): Sequence lengths (B,) if inputs are
                variable-length. Used to derive masks.
            z_mask (Tensor, optional): Mask of shape (B, T) with 1 for valid tokens.
            return_codes (bool): If True, return code indices in output.

        Returns:
            VectorQuantizerOutput: Named tuple with fields:
                - z_q (Tensor): Quantized tensor, same shape as input.
                - codebook_loss (Tensor): Loss term for codebook update (B,).
                - commitment_loss (Tensor): Loss term for commitment (B,).
                - perplexity (Tensor): Scalar perplexity.
                - codes (Tensor or None): codes of codes if requested.
        """
        z, z_shape = self.reshape_input(z)
        z_before_proj = z
        if self.in_proj is not None:
            z = self.in_proj(z)
        z_after_proj = z

        z_q, codes, probs = self.quantize_latents(
            z, return_probs=True if self._compute_diversity_loss else False
        )

        if self.training and codes.numel() > 0:
            unique_codes = torch.unique(codes)
            print(
                f"[NNVectorQuantizer] Unique codes={unique_codes.tolist()}", flush=True
            )
            if unique_codes.numel() < 4:
                torch.set_printoptions(threshold=10_000)
                T = z.shape[1] // 3
                print(
                    "[NNVectorQuantizer] z before in_proj:\n",
                    z_before_proj[0, T : T + 20, :20],
                )
                print(
                    "[NNVectorQuantizer] z after in_proj:\n",
                    z_after_proj[0, T : T + 20, :20],
                    flush=True,
                )
                print("[NNVectorQuantizer] z_q:\n", z_q[0, T : T + 20, :20], flush=True)

        if z_mask is not None:
            z_mask = z_mask.view(z_shape[0], -1)  # (B, T)
        elif z_lengths is not None:
            z_mask = seq_lengths_to_mask(
                z_lengths, z.size(1), time_dim=1, dtype=z.dtype
            )

        if z_mask is not None:
            m = z_mask.view(z_shape[0], -1).unsqueeze(-1)  # (B,T,1)
            z_q = z_q * m
            z = z * m
            den = z_mask.view(z_shape[0], -1).mean(dim=1).clamp_min(1e-8)
        else:
            den = 1

        commitment_loss = (
            F.mse_loss(z, z_q.detach(), reduction="none").mean([1, 2]) / den
        )
        commitment_loss = commitment_loss * self.commitment_weight
        codebook_loss = F.mse_loss(z_q, z.detach(), reduction="none").mean([1, 2]) / den
        if self._compute_diversity_loss:
            diversity_loss = self.compute_diversity_loss(probs, z_mask)
        else:
            diversity_loss = None

        if self._compute_orthogonality_loss:
            orth_loss = self.compute_orthogonal_loss()
        else:
            orth_loss = None

        ppl = self.codebook_perplexity_hard(codes, z_mask)

        if self.reset_unused and self.training:
            self._reset_unused_codes(z, codes, z_mask)

        commitment_loss = self._reduce_loss(commitment_loss)
        codebook_loss = self._reduce_loss(codebook_loss)

        # this allows to backprogate the gradients as if the output were equal to z_e
        z_q = z + (z_q - z).detach()

        if self.out_proj is not None:
            z_q = self.out_proj(z_q)

        if not return_codes:
            codes = None

        z_q, codes = self.reshape_output(z_q, codes, z_shape)

        output = VectorQuantizerOutput(
            z_q=z_q,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            diversity_loss=diversity_loss,
            orthogonality_loss=orth_loss,
            perplexity=ppl,
            codes=codes,
            z_mask=z_mask,
            z_lengths=z_lengths,
        )
        return output

    @staticmethod
    def filter_args(**kwargs):
        """
        Filters keyword arguments relevant to the class

        Returns:
            dict: Filtered kwargs.
        """
        return filter_func_args(NNVectorQuantizer.__init__, kwargs)


class GumbelVectorQuantizer(_GumbelSoftSamplingMixin, _GDVectorQuantizer):
    """
    Gumbel-Softmax vector quantizer.

    Unlike standard VQ, this class samples codes using the Gumbel-Softmax trick,
    which allows differentiable sampling of discrete codes.

    Attributes:
        in_feats (int): Input feature dimension.
        codebook_size (int): Number of embedding vectors (codebook size).
        codebook_dim (int, optional): Dimensionality of embedding vectors. Defaults
            to `in_feats`.
        distance_metric (VQDistanceType): Distance metric for computing logits.
        use_weight_norm (bool): If True, applies weight normalization to projections.
        channels_last (bool): If False, expects channel-first layout for >2D inputs.
        temp_init (float): Initial temperature for Gumbel-Softmax.
        temp_min (float): Minimum annealed temperature.
        temp_anneal_rate (float): Exponential decay rate for temperature scheduling.
        soft_sampling_steps (int): Number of training steps to run soft sampling
            before switching to hard sampling automatically.
        reset_unused (bool): If True (and in training mode), reinitializes codebook
            entries that remain unused for a configurable number of batches.
        reset_unused_steps (int): Consecutive forward passes a codeword can stay unused
            before being reset. Only relevant if ``reset_unused`` is True.
        commitment_anneal_steps (int): Steps to linearly ramp the commitment penalty.
        compute_diversity_loss (bool): Whether to compute diversity regularizer terms.
        compute_orthogonality_loss (bool): Whether to compute orthogonality penalty.
        losses_reduction (str): Reduction applied to returned losses ("none", "mean", "sum").
    """

    def __init__(
        self,
        in_feats,
        codebook_size,
        codebook_dim=None,
        distance_metric=VQDistanceType.L2,
        use_weight_norm=False,
        channels_last=False,
        temp_init: float = 1.0,
        temp_min: float = 0.5,
        temp_anneal_rate: float = 1e-5,
        soft_sampling_steps: int = 0,
        reset_unused: bool = False,
        reset_unused_steps: int = 1,
        commitment_anneal_steps: int = 0,
        compute_diversity_loss: bool = False,
        compute_orthogonality_loss: bool = False,
        losses_reduction: str = "mean",
    ):
        super().__init__(
            in_feats,
            codebook_size,
            codebook_dim,
            distance_metric,
            use_weight_norm,
            channels_last,
            reset_unused=reset_unused,
            reset_unused_steps=reset_unused_steps,
            temp_init=temp_init,
            temp_min=temp_min,
            temp_anneal_rate=temp_anneal_rate,
            commitment_anneal_steps=commitment_anneal_steps,
            compute_diversity_loss=compute_diversity_loss,
            compute_orthogonality_loss=compute_orthogonality_loss,
            losses_reduction=losses_reduction,
        )
        self._init_soft_sampling(soft_sampling_steps)

    def __str__(self):
        return (
            f"{self.__class__.__name__}(in_feats={self.in_feats}, "
            f"codebook_size={self.codebook_size}, codebook_dim={self.codebook_dim}, "
            f"distance_metric={self.distance_metric}, use_weight_norm={self.use_weight_norm}, "
            f"channels_last={self.channels_last}, temp={self.temp.item():.4f}, "
            f"temp_min={self.temp_min.item():.4f}, temp_anneal_rate={self.temp_anneal_rate}, "
            f"soft_sampling_steps={self.soft_sampling_steps}, "
            f"using_hard_sampling={bool(self.use_hard_sampling.item())}, "
            f"reset_unused={self.reset_unused}, "
            f"reset_unused_steps={self.reset_unused_steps if self.reset_unused else 'n/a'}, "
            f"commitment_weight={self.commitment_weight.item():.4f}, "
            f"commitment_anneal_steps={self.commitment_anneal_steps}, "
            f"compute_diversity_loss={self._compute_diversity_loss}, "
            f"compute_orthogonality_loss={self._compute_orthogonality_loss}, "
            f"losses_reduction='{self.losses_reduction}')"
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "temp_init": self.temp.item(),
                "temp_min": self.temp_min.item(),
                "temp_anneal_rate": self.temp_anneal_rate,
                "soft_sampling_steps": self.soft_sampling_steps,
            }
        )
        return cfg

    def encode_latents(
        self,
        latents: torch.Tensor,
        temp: float,
        hard: Optional[bool] = None,
        return_probs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode latents to soft Gumbel-Softmax assignments.

        Args:
            latents (Tensor): Input of shape (B, T, D).
            temp (float): Gumbel-Softmax temperature.
            hard (bool or None): If True/False override the sampling mode. When
                ``None`` the sampler follows the scheduled soft-to-hard transition.
            return_probs (bool): When ``True`` also returns the sampled soft
                assignments per latent.

        Returns:
            Tensor | Tuple[Tensor, Tensor]:
                Either the sampled codes ``(B, T)`` or a tuple ``(codes, soft_one_hot)``
                containing the corresponding soft one-hot assignments ``(B*T, K)``.
        """
        latents_shape = latents.shape
        latents = latents.view(-1, latents.shape[-1])  # (B*T, D)
        distance = self.compute_codebook_distance(latents)
        logits = -distance
        hard = self._resolve_sampling_mode(hard)
        soft_one_hot = F.gumbel_softmax(
            logits, tau=float(temp), hard=hard, dim=-1
        )  # (N, num_embed)
        codes = soft_one_hot.max(dim=1)[1]  # (B*T)
        codes = codes.view(latents_shape[0], -1)  # (B, T)
        if return_probs:
            return codes, soft_one_hot

        return codes

    def quantize_latents(
        self,
        latents: torch.Tensor,
        temp: float,
        hard: Optional[bool] = None,
        return_probs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Quantize latents using Gumbel-Softmax assignments.

        Args:
            latents (Tensor): Input of shape (B, T, D).
            temp (float): Gumbel-Softmax temperature.
            hard (bool or None): If provided, overrides the scheduled sampling mode.
            return_probs (bool): When ``True`` includes the sampled soft responsibilities.

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor]]:
                Quantized output ``(B, T, D)``, sampled codes ``(B, T)``, and optionally
                the soft Gumbel assignments ``(B*T, K)``.
        """
        encodings = self.encode_latents(latents, temp, hard, return_probs=True)
        assert isinstance(encodings, tuple)
        codes, soft_one_hot = encodings
        z_q = torch.matmul(soft_one_hot, self.codebook).view(
            latents.shape
        )  # (B*T, D) -> (B, T, D)
        if return_probs:
            return z_q, codes, soft_one_hot

        return z_q, codes, None

    def forward(
        self,
        z: torch.Tensor,
        z_lengths: Optional[torch.Tensor] = None,
        z_mask: Optional[torch.Tensor] = None,
        temp: float = None,
        hard: Optional[bool] = None,
        return_codes: bool = False,
    ) -> VectorQuantizerOutput:
        """
        Forward pass with Gumbel-Softmax quantization.

        Args:
            z (Tensor): Input tensor (B, ..., D).
            z_lengths (Tensor, optional): Sequence lengths (B,).
            z_mask (Tensor, optional): Mask (B, T).
            temp (float, optional): Gumbel-Softmax temperature. Defaults to current
                internal temperature buffer.
            hard (bool or None): Override for straight-through mode. When None,
                the layer selects soft sampling while training until
                ``soft_sampling_steps`` is reached, then switches to hard sampling.
                Evaluation mode always uses hard sampling.
            return_codes (bool): If True, include sampled codes in output.

        Returns:
            VectorQuantizerOutput: Named tuple with fields as in VectorQuantizer.
        """
        z, z_shape = self.reshape_input(z)
        if self.in_proj is not None:
            z = self.in_proj(z)

        if temp is None:
            temp = self.temp.item()

        z_q, codes, probs = self.quantize_latents(z, temp, hard, return_probs=True)

        if z_mask is not None:
            z_mask = z_mask.view(z_shape[0], -1)  # (B, T)
        elif z_lengths is not None:
            z_mask = seq_lengths_to_mask(
                z_lengths, z.size(1), time_dim=1, dtype=z.dtype
            )

        if z_mask is not None:
            m = z_mask.view(z_shape[0], -1).unsqueeze(-1)  # (B,T,1)
            z_q = z_q * m
            z = z * m
            den = z_mask.view(z_shape[0], -1).mean(dim=1).clamp_min(1e-8)
        else:
            den = 1

        commitment_loss = (
            F.mse_loss(z, z_q.detach(), reduction="none").mean([1, 2]) / den
        )
        codebook_loss = F.mse_loss(z_q, z.detach(), reduction="none").mean([1, 2]) / den
        commitment_loss = commitment_loss * self.commitment_weight
        if self._compute_diversity_loss:
            diversity_loss = self.compute_diversity_loss(probs, z_mask)
        else:
            diversity_loss = None

        if self._compute_orthogonality_loss:
            orth_loss = self.compute_orthogonal_loss()
        else:
            orth_loss = None

        ppl = self.codebook_perplexity_soft(probs, z_mask)

        if self.reset_unused and self.training:
            self._reset_unused_codes(z, codes, z_mask)

        commitment_loss = self._reduce_loss(commitment_loss)
        codebook_loss = self._reduce_loss(codebook_loss)

        if self.out_proj is not None:
            z_q = self.out_proj(z_q)

        if not return_codes:
            codes = None

        z_q, codes = self.reshape_output(z_q, codes, z_shape)

        output = VectorQuantizerOutput(
            z_q=z_q,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            diversity_loss=diversity_loss,
            orthogonality_loss=orth_loss,
            perplexity=ppl,
            codes=codes,
            z_mask=z_mask,
            z_lengths=z_lengths,
        )
        return output

    @staticmethod
    def filter_args(**kwargs):
        """
        Filters keyword arguments relevant to the class

        Returns:
            dict: Filtered kwargs.
        """
        return filter_func_args(GumbelVectorQuantizer.__init__, kwargs)


class EMANNVectorQuantizer(VectorQuantizerBase):
    """
    Vector quantizer with Exponential Moving Average (EMA) codebook updates.

    This class implements the "EMA" variant of vector quantization described
    in VQ-VAE. Instead of updating the codebook weights by gradient descent
    (as in nearest-neighbor VQ), it maintains running exponential moving
    averages of (1) cluster sizes (how often each code is used) and
    (2) the sum of encoder outputs assigned to each code. The codebook is
    updated from these statistics after each forward pass.

    Characteristics:
        • The codebook is stored as a buffer (`codebook`), not a Parameter.
        • Updates happen in `forward()` via `_ema_update` (no gradients).
        • `codebook_loss` is always zero, since EMA handles codebook updates.
        • `commitment_loss` is still included, to align encoder outputs with
          their assigned codes and is annealed via `commitment_weight`.
        • Supports L2, L1, and cosine distance metrics for nearest-neighbor
          assignment.
        • Sequence masks (`z_mask`) and lengths (`z_lengths`) are supported.
        • Optional code re-initialization for unused embeddings.
        • Temperature scheduling (``temp``/``temp_min``) and commitment loss
          annealing are inherited from :class:`VectorQuantizerBase`.

    Attributes:
        in_feats (int): Input feature dimension.
        codebook_size (int): Number of embedding vectors (codebook size).
        codebook_dim (int, optional): Dimension of embedding vectors. Defaults
            to `in_feats`. If different, input/output projection layers are added.
        distance_metric (VQDistanceType): Distance metric for nearest-neighbor
            search (L2, L1, cosine).
        use_weight_norm (bool): If True, applies weight normalization to the
            optional projection layers.
        channels_last (bool): If False, expects channel-first layout for >2D
            tensors (e.g., (B,C,H,W)) and transposes internally.
        decay (float): EMA decay rate (typically ~0.99).
        eps (float): Small constant to avoid division by zero in cluster counts.
        reset_unused (bool): If True, reinitializes codes that are effectively
            unused (cluster size < 1.0) from random encoder samples.
        temp_init (float): Initial temperature used by the base class.
        temp_min (float): Minimum value that temperature can attain.
        temp_anneal_rate (float): Exponential decay rate for the temperature.
        commitment_anneal_steps (int): Steps over which to ramp the commitment weight.
        compute_diversity_loss (bool): Whether to compute diversity regularizer terms.
        compute_orthogonality_loss (bool): Whether to compute orthogonality penalty.
        losses_reduction (str): Reduction applied to returned losses ("none", "mean", "sum").
    """

    def __init__(
        self,
        in_feats: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        distance_metric: VQDistanceType = VQDistanceType.L2,
        use_weight_norm: bool = False,
        channels_last: bool = False,
        decay: float = 0.99,
        eps: float = 1e-5,
        reset_unused: bool = False,
        temp_init: float = 1.0,
        temp_min: Optional[float] = 0.5,
        temp_anneal_rate: float = 1e-5,
        commitment_anneal_steps: int = 0,
        compute_diversity_loss: bool = False,
        compute_orthogonality_loss: bool = False,
        losses_reduction: str = "mean",
    ):
        super().__init__(
            in_feats,
            codebook_size,
            codebook_dim,
            distance_metric,
            use_weight_norm,
            channels_last,
            is_ema=True,
            temp_init=temp_init,
            temp_min=temp_min,
            temp_anneal_rate=temp_anneal_rate,
            commitment_anneal_steps=commitment_anneal_steps,
            compute_diversity_loss=compute_diversity_loss,
            compute_orthogonality_loss=compute_orthogonality_loss,
            losses_reduction=losses_reduction,
        )
        self.decay = decay
        self.eps = eps
        self.reset_unused = reset_unused

        # EMA buffers
        self.register_buffer(
            "ema_cluster_size", torch.zeros(self.codebook_size, dtype=torch.float32)
        )
        self.register_buffer(
            "ema_embed_avg", self.codebook.data.clone().to(torch.float32)
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__}(in_feats={self.in_feats}, codebook_size={self.codebook_size}, "
            f"codebook_dim={self.codebook_dim}, distance_metric={self.distance_metric}, "
            f"use_weight_norm={self.use_weight_norm}, channels_last={self.channels_last}, "
            f"decay={self.decay}, eps={self.eps}, reset_unused={self.reset_unused}, "
            f"temp={self.temp.item():.4f}, temp_min={self.temp_min.item():.4f}, "
            f"temp_anneal_rate={self.temp_anneal_rate}, "
            f"commitment_weight={self.commitment_weight.item():.4f}, "
            f"commitment_anneal_steps={self.commitment_anneal_steps}, "
            f"compute_diversity_loss={self._compute_diversity_loss}, "
            f"compute_orthogonality_loss={self._compute_orthogonality_loss}, "
            f"losses_reduction='{self.losses_reduction}')"
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "decay": self.decay,
                "eps": self.eps,
                "reset_unused": self.reset_unused,
            }
        )
        return cfg

    @torch.no_grad()
    def _ema_update(
        self,
        flat_z: torch.Tensor,
        flat_codes: torch.Tensor,
        mask_1d: torch.Tensor | None,
    ):
        """
        Update EMA cluster statistics and codebook weights.

        Args:
            flat_z (Tensor): Flattened encoder outputs of shape (N, D).
            flat_codes (Tensor): Integer code assignments of shape (N,).
            mask_1d (Tensor, optional): Boolean or float mask of shape (N,)
                indicating valid positions. If None, all entries are valid.

        Behavior:
            • Accumulates per-code counts (`ema_cluster_size`) and sums
              (`ema_embed_avg`) with exponential decay.
            • Normalizes to update `codebook = ema_embed_avg / cluster_size`.
            • Optionally reinitializes unused codes if `reset_unused=True`.
        """
        K = self.codebook_size
        D = self.codebook_dim
        device = flat_z.device
        dtype = flat_z.dtype

        if mask_1d is None:
            mask_1d = torch.ones(
                flat_codes.shape[0], device=device, dtype=torch.float32
            )
        else:
            # convert to float for weighting
            mask_1d = mask_1d.to(torch.float32)

        # One-hot (N,K) weighted by mask
        one_hot = F.one_hot(flat_codes, num_classes=K).to(torch.float32)  # (N,K)
        one_hot = one_hot * mask_1d.unsqueeze(-1)  # (N,K)

        # Per-code counts and sums
        batch_cluster_size = one_hot.sum(dim=0)  # (K,)
        batch_embed_sum = one_hot.T @ flat_z.to(torch.float32)  # (K,D)

        self._maybe_allreduce_tensor(batch_cluster_size)
        self._maybe_allreduce_tensor(batch_embed_sum)

        # EMA update
        self.ema_cluster_size.mul_(self.decay).add_(
            (1.0 - self.decay) * batch_cluster_size
        )
        self.ema_embed_avg.mul_(self.decay).add_((1.0 - self.decay) * batch_embed_sum)

        # Laplace smoothing of counts to avoid zeros
        n = self.ema_cluster_size + self.eps  # (K,)
        new_weight = self.ema_embed_avg / n.unsqueeze(-1)  # (K,D)

        # Optional: reinitialize codes that are effectively unused
        if self.reset_unused:
            unused = self.ema_cluster_size < 1.0
            if unused.any():
                new_weight = self._reset_unused_codes(
                    flat_z, mask_1d, new_weight, unused
                )

        self.codebook.data.copy_(new_weight.to(self.codebook.dtype))

    @torch.no_grad()
    def _reset_unused_codes(
        self,
        flat_z: torch.Tensor,
        mask_1d: torch.Tensor | None,
        new_weight: torch.Tensor,
        unused: torch.Tensor,
    ) -> torch.Tensor:
        """Reinitialize unused EMA codebook entries in a DDP-safe manner."""
        device = new_weight.device
        D = new_weight.shape[-1]

        if mask_1d is not None:
            valid_mask = mask_1d > 0
            flat_z_valid = flat_z[valid_mask]
        else:
            flat_z_valid = flat_z

        num_unused = int(unused.sum().item())
        dist_ready = dist.is_available() and dist.is_initialized()

        if dist_ready:
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            local_count = torch.tensor(
                [flat_z_valid.shape[0]], device=device, dtype=torch.long
            )
            count_list = [torch.zeros_like(local_count) for _ in range(world_size)]
            dist.all_gather(count_list, local_count)
            counts_per_rank = torch.stack(count_list)
            max_valid = int(counts_per_rank.max().item())

            if max_valid > 0:
                padded = torch.zeros(
                    max_valid,
                    D,
                    device=device,
                    dtype=flat_z_valid.dtype,
                )
                if flat_z_valid.shape[0] > 0:
                    copy_len = min(max_valid, flat_z_valid.shape[0])
                    padded[:copy_len] = flat_z_valid[:copy_len]
                gathered = [torch.zeros_like(padded) for _ in range(world_size)]
                dist.all_gather(gathered, padded)
                if rank == 0:
                    candidates = torch.cat(
                        [g[: n.item()] for g, n in zip(gathered, counts_per_rank)],
                        dim=0,
                    )
                else:
                    candidates = torch.empty(
                        0,
                        D,
                        device=device,
                        dtype=flat_z_valid.dtype,
                    )
            else:
                candidates = torch.empty(
                    0,
                    D,
                    device=device,
                    dtype=flat_z_valid.dtype,
                )

            if rank == 0:
                if candidates.shape[0] > 0:
                    rand_idx = torch.randint(
                        0, candidates.shape[0], (num_unused,), device=device
                    )
                    new_vectors = candidates[rand_idx]
                else:
                    new_vectors = torch.randn(
                        num_unused,
                        D,
                        device=device,
                        dtype=new_weight.dtype,
                    ) * (1.0 / math.sqrt(D))
            else:
                new_vectors = torch.empty(
                    num_unused,
                    D,
                    device=device,
                    dtype=new_weight.dtype,
                )

            dist.broadcast(new_vectors, src=0)
        else:
            if flat_z_valid.shape[0] > 0:
                rand_idx = torch.randint(
                    0, flat_z_valid.shape[0], (num_unused,), device=device
                )
                new_vectors = flat_z_valid[rand_idx]
            else:
                new_vectors = torch.randn(
                    num_unused,
                    D,
                    device=device,
                    dtype=new_weight.dtype,
                ) * (1.0 / math.sqrt(D))

        new_vectors = new_vectors.to(new_weight.dtype)
        new_weight = new_weight.clone()
        new_weight[unused] = new_vectors
        self.ema_embed_avg[unused] = new_vectors.to(self.ema_embed_avg.dtype)
        self.ema_cluster_size[unused] = torch.ones_like(self.ema_cluster_size[unused])

        return new_weight

    def forward(
        self,
        z: torch.Tensor,
        z_lengths: torch.Tensor | None = None,
        z_mask: torch.Tensor | None = None,
        return_codes: bool = False,
    ) -> VectorQuantizerOutput:
        """
        Forward pass of the EMA vector quantizer.

        Steps:
            1. Flattens and projects the input to shape (B,T,D).
            2. Assigns each vector to nearest code via hard NN lookup.
            3. Quantizes inputs (`z_q`) and applies optional masks.
            4. Computes losses:
                - `commitment_loss`: MSE between encoder outputs and detached codes.
                - `codebook_loss`: always zero (EMA update handles codebook).
            5. Calls `_ema_update` to update cluster statistics and codebook weights.
            6. Applies straight-through estimator to backprop encoder gradients.

        Args:
            z (Tensor): Input tensor of shape (B, ..., D).
            z_lengths (Tensor, optional): Sequence lengths for variable-length
                inputs, shape (B,).
            z_mask (Tensor, optional): Boolean or float mask of shape (B,T),
                indicating valid timesteps.
            return_codes (bool): If True, include code indices in the output.

        Returns:
            VectorQuantizerOutput:
                - z_q (Tensor): Quantized tensor, same shape as input.
                - codebook_loss (Tensor): Always zeros, same batch shape as commitment_loss.
                - commitment_loss (Tensor): Encoder commitment loss, shape (B,).
                - perplexity (Tensor): Scalar codebook perplexity.
                - codes (Tensor or None): Assigned code indices if requested.
        """
        # Reshape & (optional) input projection
        z, orig_shape = self.reshape_input(z)
        if self.in_proj is not None:
            z = self.in_proj(z)  # (B,T,D)

        z_q, codes, probs = self.quantize_latents(
            z, True if self._compute_diversity_loss else False
        )  # (B,T,D), (B,T) , (B,T,K) or None

        # Build mask (B,T) -> (B,T,1) for broadcasting, and den for normalization
        if z_mask is not None:
            z_mask_2d = z_mask.view(orig_shape[0], -1)
        elif z_lengths is not None:
            z_mask_2d = seq_lengths_to_mask(
                z_lengths, z.size(1), time_dim=1, dtype=z.dtype
            )
        else:
            z_mask_2d = None

        if z_mask_2d is not None:
            m = z_mask_2d.unsqueeze(-1)  # (B,T,1)
            z = z * m
            z_q = z_q * m
            den = z_mask_2d.mean(dim=1).clamp_min(1e-8)
        else:
            den = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)

        commitment_loss = (
            F.mse_loss(z, z_q.detach(), reduction="none").mean([1, 2]) / den
        )
        codebook_loss = (
            commitment_loss.detach()
        )  # codebook loss doesn't impact EMA updates
        commitment_loss = commitment_loss * self.commitment_weight
        if self._compute_diversity_loss:
            diversity_loss = self.compute_diversity_loss(probs, z_mask_2d)
        else:
            diversity_loss = None

        if self._compute_orthogonality_loss:
            orth_loss = self.compute_orthogonal_loss()
        else:
            orth_loss = None

        ppl = self.codebook_perplexity_hard(codes, z_mask_2d)

        # Straight-through estimator for the path to encoder
        z_q = z + (z_q - z).detach()

        # EMA update (no grad) on flattened views
        if self.training:
            with torch.no_grad():
                flat_z = z.view(-1, z.shape[-1])  # (N,D)
                flat_codes = codes.view(-1)  # (N,)
                flat_mask = z_mask_2d.view(-1) if z_mask_2d is not None else None
                self._ema_update(flat_z, flat_codes, flat_mask)

        # Output projection if needed
        if self.out_proj is not None:
            z_q = self.out_proj(z_q)

        # Optionally drop codes
        codes = codes if return_codes else None

        commitment_loss = self._reduce_loss(commitment_loss)
        codebook_loss = self._reduce_loss(codebook_loss)

        # Restore shapes/layout
        z_q, codes = self.reshape_output(z_q, codes, orig_shape)

        return VectorQuantizerOutput(
            z_q=z_q,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            diversity_loss=diversity_loss,
            orthogonality_loss=orth_loss,
            perplexity=ppl,
            codes=codes,
            z_mask=z_mask,
            z_lengths=z_lengths,
        )

    @staticmethod
    def filter_args(**kwargs):
        """
        Filters keyword arguments relevant to the class

        Returns:
            dict: Filtered kwargs.
        """
        return filter_func_args(EMANNVectorQuantizer.__init__, kwargs)


class EMAGumbelVectorQuantizer(_GumbelSoftSamplingMixin, EMANNVectorQuantizer):
    """
    Gumbel-Softmax vector quantizer with EMA codebook updates.

    This class combines the Gumbel-Softmax sampling approach with Exponential
    Moving Average (EMA) updates for the codebook, as described in VQ-VAE.
    It allows differentiable sampling of discrete codes while maintaining
    a stable codebook through EMA statistics.

    Attributes:
        in_feats (int): Input feature dimension.
        codebook_size (int): Number of embedding vectors (codebook size).
        codebook_dim (int, optional): Dimensionality of embedding vectors. Defaults
            to `in_feats`. If different, input/output projection layers are added.
        distance_metric (VQDistanceType): Distance metric for nearest-neighbor
            search (L2, L1, cosine).
        use_weight_norm (bool): If True, applies weight normalization to the
            optional projection layers.
        channels_last (bool): If False, expects channel-first layout for >2D
            tensors (e.g., (B,C,H,W)) and transposes internally.
        decay (float): EMA decay rate (typically ~0.99).
        eps (float): Small constant to avoid division by zero in cluster counts.
        reset_unused (bool): If True, reinitializes codes that are effectively
            unused (cluster size < 1.0) from random encoder samples.
        temp_init (float): Initial temperature for Gumbel-Softmax.
        temp_min (float): Minimum annealed temperature.
        temp_anneal_rate (float): Exponential decay rate for temperature scheduling.
        soft_sampling_steps (int): Number of training steps to run soft sampling
            before switching to hard sampling automatically.
        commitment_anneal_steps (int): Steps to linearly ramp the commitment penalty.
        compute_diversity_loss (bool): Whether to compute diversity regularizer terms.
        compute_orthogonality_loss (bool): Whether to compute orthogonality penalty.
    """

    def __init__(
        self,
        in_feats: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        distance_metric: VQDistanceType = VQDistanceType.L2,
        use_weight_norm: bool = False,
        channels_last: bool = False,
        decay: float = 0.99,
        eps: float = 1e-5,
        reset_unused: bool = False,
        temp_init: float = 1.0,
        temp_min: float = 0.5,
        temp_anneal_rate: float = 1e-5,
        soft_sampling_steps: int = 0,
        commitment_anneal_steps: int = 0,
        compute_diversity_loss: bool = False,
        compute_orthogonality_loss: bool = False,
        losses_reduction: str = "mean",
    ):
        super().__init__(
            in_feats,
            codebook_size,
            codebook_dim,
            distance_metric,
            use_weight_norm,
            channels_last,
            decay,
            eps,
            reset_unused,
            temp_init=temp_init,
            temp_min=temp_min,
            temp_anneal_rate=temp_anneal_rate,
            commitment_anneal_steps=commitment_anneal_steps,
            compute_diversity_loss=compute_diversity_loss,
            compute_orthogonality_loss=compute_orthogonality_loss,
            losses_reduction=losses_reduction,
        )
        self._init_soft_sampling(soft_sampling_steps)

    def __str__(self):
        return (
            f"{self.__class__.__name__}(in_feats={self.in_feats}, codebook_size={self.codebook_size}, "
            f"codebook_dim={self.codebook_dim}, distance_metric={self.distance_metric}, "
            f"use_weight_norm={self.use_weight_norm}, channels_last={self.channels_last}, "
            f"decay={self.decay}, eps={self.eps}, reset_unused={self.reset_unused}, "
            f"temp={self.temp.item():.4f}, temp_min={self.temp_min.item():.4f}, "
            f"temp_anneal_rate={self.temp_anneal_rate}, "
            f"soft_sampling_steps={self.soft_sampling_steps}, "
            f"using_hard_sampling={bool(self.use_hard_sampling.item())}, "
            f"commitment_weight={self.commitment_weight.item():.4f}, "
            f"commitment_anneal_steps={self.commitment_anneal_steps}, "
            f"compute_diversity_loss={self._compute_diversity_loss}, "
            f"compute_orthogonality_loss={self._compute_orthogonality_loss}, "
            f"losses_reduction='{self.losses_reduction}')"
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "temp_init": self.temp.item(),
                "temp_min": self.temp_min.item(),
                "temp_anneal_rate": self.temp_anneal_rate,
                "soft_sampling_steps": self.soft_sampling_steps,
            }
        )
        return cfg

    def encode_latents(
        self,
        latents: torch.Tensor,
        temp: float,
        hard: Optional[bool] = None,
        return_probs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode latents to soft Gumbel-Softmax assignments.

        Args:
            latents (Tensor): Input of shape (B, T, D).
            temp (float): Gumbel-Softmax temperature.
            hard (bool or None): If set, overrides the scheduled sampling mode.

        Returns:
            Tensor | Tuple[Tensor, Tensor]:
                Either the sampled codes ``(B, T)`` or a tuple ``(codes, soft_one_hot)``
                with the accompanying soft responsibilities ``(B*T, K)`` if
                ``return_probs`` is True.
        """
        latents_shape = latents.shape
        latents = latents.view(-1, latents.shape[-1])  # (B*T, D)
        distance = self.compute_codebook_distance(latents)
        logits = -distance
        hard = self._resolve_sampling_mode(hard)
        soft_one_hot = F.gumbel_softmax(
            logits, tau=float(temp), hard=hard, dim=-1
        )  # (N, num_embed)
        codes = soft_one_hot.max(dim=1)[1]  # (B*T)
        codes = codes.view(latents_shape[0], -1)  # (B, T)
        if return_probs:
            return codes, soft_one_hot

        return codes

    def quantize_latents(
        self,
        latents: torch.Tensor,
        temp: float,
        hard: Optional[bool] = None,
        return_probs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Quantize latents using Gumbel-Softmax assignments.

        Args:
            latents (Tensor): Input of shape (B, T, D).
            temp (float): Gumbel-Softmax temperature.
            hard (bool or None): When provided overrides the scheduled sampling mode.

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor]]:
                Quantized output ``(B, T, D)``, sampled codes ``(B, T)``, and
                optionally the soft Gumbel assignments ``(B*T, K)`` when
                ``return_probs`` is requested.
        """
        encodings = self.encode_latents(latents, temp, hard, return_probs=True)
        assert isinstance(encodings, tuple)
        codes, soft_one_hot = encodings
        z_q = torch.matmul(soft_one_hot, self.codebook).view(
            latents.shape
        )  # (B*T, D) -> (B, T, D)
        if return_probs:
            return z_q, codes, soft_one_hot

        return z_q, codes, None

    def forward(
        self,
        z: torch.Tensor,
        z_lengths: torch.Tensor | None = None,
        z_mask: torch.Tensor | None = None,
        temp: float = None,
        hard: Optional[bool] = None,
        return_codes: bool = False,
    ) -> VectorQuantizerOutput:
        """
        Forward pass of the EMA vector quantizer.

        Steps:
            1. Flattens and projects the input to shape (B,T,D).
            2. Samples (soft or hard) code assignments via Gumbel-Softmax.
            3. Quantizes inputs (`z_q`) and applies optional masks.
            4. Computes losses:
                - `commitment_loss`: MSE between encoder outputs and detached codes.
                - `codebook_loss`: always zero (EMA update handles codebook).
            5. Calls `_ema_update` to update cluster statistics and codebook weights.
            6. Applies straight-through estimator to backprop encoder gradients.

        Args:
            z (Tensor): Input tensor of shape (B, ..., D).
            z_lengths (Tensor, optional): Sequence lengths for variable-length
                inputs, shape (B,).
            z_mask (Tensor, optional): Boolean or float mask of shape (B,T),
                indicating valid timesteps.
            temp (float, optional): Gumbel-Softmax temperature. Defaults to current
                internal temperature buffer.
            hard (bool or None): Override for straight-through sampling. When None the
                layer transitions from soft to hard sampling based on
                ``soft_sampling_steps``; evaluation mode always uses hard sampling.
            return_codes (bool): If True, include code indices in the output.

        Returns:
            VectorQuantizerOutput:
                - z_q (Tensor): Quantized tensor, same shape as input.
                - codebook_loss (Tensor): Always zeros, same batch shape as commitment_loss.
                - commitment_loss (Tensor): Encoder commitment loss, shape (B,).
                - perplexity (Tensor): Scalar codebook perplexity.
                - codes (Tensor or None): Assigned code indices if requested.
        """
        # Reshape & (optional) input projection
        z, orig_shape = self.reshape_input(z)
        if self.in_proj is not None:
            z = self.in_proj(z)  # (B,T,D)

        if temp is None:
            temp = self.temp.item()
        z_q, codes, soft_one_hot = self.quantize_latents(z, temp, hard)

        # Build mask (B,T) -> (B,T,1) for broadcasting, and den for normalization
        if z_mask is not None:
            z_mask_2d = z_mask.view(orig_shape[0], -1)
        elif z_lengths is not None:
            z_mask_2d = seq_lengths_to_mask(
                z_lengths, z.size(1), time_dim=1, dtype=z.dtype
            )
        else:
            z_mask_2d = None

        if z_mask_2d is not None:
            m = z_mask_2d.unsqueeze(-1)  # (B,T,1)
            z = z * m
            z_q = z_q * m
            den = z_mask_2d.mean(dim=1).clamp_min(1e-8)
        else:
            den = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)

        # Losses (no codebook loss for EMA)
        commitment_loss = (
            F.mse_loss(z, z_q.detach(), reduction="none").mean([1, 2]) / den
        )
        codebook_loss = (
            commitment_loss.detach()
        )  # codebook loss doesn't impact EMA updates
        commitment_loss = commitment_loss * self.commitment_weight
        if self._compute_diversity_loss:
            diversity_loss = self.compute_diversity_loss(soft_one_hot, z_mask_2d)
        else:
            diversity_loss = None

        if self._compute_orthogonality_loss:
            orth_loss = self.compute_orthogonal_loss()
        else:
            orth_loss = None

        ppl = self.codebook_perplexity_soft(soft_one_hot, z_mask_2d)

        # EMA update (no grad) on flattened views
        if self.training:
            with torch.no_grad():
                flat_z = z.view(-1, z.shape[-1])  # (N,D)
                flat_codes = codes.view(-1)  # (N,)
                flat_mask = z_mask_2d.view(-1) if z_mask_2d is not None else None
                self._ema_update(flat_z, flat_codes, flat_mask)

        # Output projection if needed
        if self.out_proj is not None:
            z_q = self.out_proj(z_q)

        # Optionally drop codes
        codes = codes if return_codes else None

        commitment_loss = self._reduce_loss(commitment_loss)
        codebook_loss = self._reduce_loss(codebook_loss)

        # Restore shapes/layout
        z_q, codes = self.reshape_output(z_q, codes, orig_shape)

        return VectorQuantizerOutput(
            z_q=z_q,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            diversity_loss=diversity_loss,
            orthogonality_loss=orth_loss,
            perplexity=ppl,
            codes=codes,
            z_mask=z_mask,
            z_lengths=z_lengths,
        )

    @staticmethod
    def filter_args(**kwargs):
        """
        Filters keyword arguments relevant to the class

        Returns:
            dict: Filtered kwargs.
        """
        return filter_func_args(EMAGumbelVectorQuantizer.__init__, kwargs)
