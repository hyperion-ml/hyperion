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
    commitment_loss: torch.Tensor = None  # Commitment loss
    perplexity: torch.Tensor = None  # Perplexity of the responsibilities
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
        # nn.init.uniform_(
        #     W, -1.0 / math.sqrt(self.codebook_dim), 1.0 / math.sqrt(self.codebook_dim)
        # )
        # nn.init.normal_(W, 0.0, 1.0 / math.sqrt(self.codebook_dim))
        nn.init.normal_(W, 0.0, 1.0)
        if is_ema:
            self.register_buffer("codebook", W)  # <- buffer, not Parameter
        else:
            self.codebook = nn.Parameter(W)  # <- Parameter

        self.init_weights()

    def get_config(self):
        """Returns the configuration of the vector quantizer."""
        return {
            "in_feats": self.in_feats,
            "codebook_size": self.codebook_size,
            "codebook_dim": self.codebook_dim,
            "distance_metric": self.distance_metric.value,
            "use_weight_norm": self.use_weight_norm,
            "channels_last": self.channels_last,
        }

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}(in_feats={self.in_feats}, codebook_size={self.codebook_size}, codebook_dim={self.codebook_dim}, distance_metric={self.distance_metric}, use_weight_norm={self.use_weight_norm}, channels_last={self.channels_last})"

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
        counts = torch.bincount(flat_codes, minlength=self.codebook_size).float()
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
        counts = soft_one_hot.sum(dim=0)  # (num_embed,)
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
            # return 1 - F.cosine_similarity(
            #     encodings.unsqueeze(1), codebook.unsqueeze(0), dim=-1
            # )
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

    def encode_latents(self, latents):
        """
        Encode continuous latents to nearest code indices.

        Args:
            latents (Tensor): Input tensor of shape (B, T, D).

        Returns:
            Tensor: Code indices of shape (B, T).
        """
        latents_shape = latents.shape
        latents = latents.view(-1, latents.shape[-1])  # (B*T, D)
        distance = self.compute_codebook_distance(latents)
        # print("distance", distance, flush=True)
        codes = distance.min(dim=1)[1]  # (B*T)
        unique_codes = torch.unique(codes)
        if unique_codes.numel() < 4:
            torch.set_printoptions(threshold=10_000)
            print(f"[encode_latents] Unique codes={unique_codes.tolist()}")
            T = latents.shape[0] // 3
            print(f"[encode_latents] latents:\n", latents[T : T + 20, :20])
            print(f"[encode_latents] distance:\n", distance[T : T + 20, :20])
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
        return codes

    def decode_latents(self, latents):
        """
        Quantize continuous latents by nearest-neighbor lookup.

        Args:
            latents (Tensor): Input tensor of shape (B, T, D).

        Returns:
            (Tensor, Tensor):
                - Quantized tensor of shape (B, T, D).
                - codes of shape (B, T).
        """
        codes = self.encode_latents(latents)
        z_q = self.decode_codes(codes)  # (B, T, D)
        return z_q, codes

    def reshape_input(self, x):
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

        x = x.view(x.shape[0], -1, x.shape[-1])
        # x = (B, T, D) or (B, HW, D)
        return x, orig_shape

    def reshape_output(self, y, codes, orig_shape):
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
            y = y.view(
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
    ):
        super().__init__(
            in_feats,
            codebook_size,
            codebook_dim,
            distance_metric,
            use_weight_norm,
            channels_last,
            is_ema=False,
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
        )

    def get_config(self):
        """Returns the configuration of the vector quantizer."""
        return super().get_config()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}(in_feats={self.in_feats}, codebook_size={self.codebook_size}, codebook_dim={self.codebook_dim}, distance_metric={self.distance_metric}, use_weight_norm={self.use_weight_norm}, channels_last={self.channels_last}, reset_unused={self.reset_unused}, reset_unused_steps={self.reset_unused_steps if self.reset_unused else 'n/a'})"

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

        z_q, codes = self.decode_latents(z)

        if self.training and codes.numel() > 0:
            unique_codes = torch.unique(codes)
            if unique_codes.numel() < 4:
                torch.set_printoptions(threshold=10_000)
                T = z.shape[1] // 3
                print(f"[NNVectorQuantizer] Unique codes={unique_codes.tolist()}")
                print(
                    "[NNVectorQuantizer] z before in_proj:\n",
                    z_before_proj[0, T : T + 20, :20],
                )
                print(
                    "[NNVectorQuantizer] z after in_proj:\n",
                    z_after_proj[0, T : T + 20, :20],
                )

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
        ppl = self.codebook_perplexity_hard(codes, z_mask)

        if self.reset_unused and self.training:
            self._reset_unused_codes(z, codes, z_mask)

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


class GumbelVectorQuantizer(_GDVectorQuantizer):
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
        anneal_rate (float): Exponential decay rate for temperature scheduling.
        reset_unused (bool): If True (and in training mode), reinitializes codebook
            entries that remain unused for a configurable number of batches.
        reset_unused_steps (int): Consecutive forward passes a codeword can stay unused
            before being reset. Only relevant if ``reset_unused`` is True.
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
        anneal_rate: float = 1e-5,
        reset_unused: bool = False,
        reset_unused_steps: int = 1,
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
        )
        # Temperature parameters
        self.register_buffer("temp", torch.tensor(temp_init))
        self.register_buffer("temp_min", torch.tensor(temp_min))
        self.anneal_rate = anneal_rate

    def __str__(self):
        return (
            f"{self.__class__.__name__}(in_feats={self.in_feats}, "
            f"codebook_size={self.codebook_size}, codebook_dim={self.codebook_dim}, "
            f"distance_metric={self.distance_metric}, use_weight_norm={self.use_weight_norm}, "
            f"channels_last={self.channels_last}, temp={self.temp.item():.4f}, "
            f"temp_min={self.temp_min.item():.4f}, anneal_rate={self.anneal_rate}, "
            f"reset_unused={self.reset_unused}, reset_unused_steps={self.reset_unused_steps if self.reset_unused else 'n/a'})"
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "temp_init": self.temp.item(),
                "temp_min": self.temp_min.item(),
                "anneal_rate": self.anneal_rate,
            }
        )
        return cfg

    @torch.no_grad()
    def update_temp(self, global_step: int = 1) -> None:
        """
        Anneal temperature exponentially.

        Update rule:
            T = max(temp_min, temp * exp(-anneal_rate * step))

        Args:
            global_step (int): Number of elapsed training steps.
        """
        new_temp = torch.clamp(
            self.temp * torch.exp(-self.anneal_rate * global_step),
            min=self.temp_min.item(),
        )
        self.temp.copy_(new_temp)

    def encode_latents(
        self, latents: torch.Tensor, temp: float, hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode latents to soft Gumbel-Softmax assignments.

        Args:
            latents (Tensor): Input of shape (B, T, D).
            temp (float): Gumbel-Softmax temperature.
            hard (bool): If True, use straight-through (hard) sampling.

        Returns:
            (Tensor, Tensor):
                - codes (B, T): Argmax over sampled codes.
                - Soft assignments (N, codebook_size): One-hot or soft one-hot.
        """
        latents_shape = latents.shape
        latents = latents.view(-1, latents.shape[-1])  # (B*T, D)
        distance = self.compute_codebook_distance(latents)
        logits = -distance
        soft_one_hot = F.gumbel_softmax(
            logits, tau=float(temp), hard=hard, dim=-1
        )  # (N, num_embed)
        codes = soft_one_hot.max(dim=1)[1]  # (B*T)
        codes = codes.view(latents_shape[0], -1)  # (B, T)
        return codes, soft_one_hot

    def decode_latents(
        self, latents: torch.Tensor, temp: float, hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize latents using Gumbel-Softmax assignments.

        Args:
            latents (Tensor): Input of shape (B, T, D).
            temp (float): Gumbel-Softmax temperature.
            hard (bool): If True, apply straight-through discretization.

        Returns:
            (Tensor, Tensor, Tensor):
                - Quantized output (B, T, D).
                - codes (B, T).
                - Soft one-hot assignments (N, codebook_size).
        """
        codes, soft_one_hot = self.encode_latents(latents, temp, hard)
        z_q = torch.matmul(soft_one_hot, self.codebook).view(
            latents.shape
        )  # (B*T, D) -> (B, T, D)
        return z_q, codes, soft_one_hot

    def forward(
        self,
        z: torch.Tensor,
        z_lengths: Optional[torch.Tensor] = None,
        z_mask: Optional[torch.Tensor] = None,
        temp: float = None,
        hard: bool = False,
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
            hard (bool): If True, use straight-through (hard) discretization.
            return_codes (bool): If True, include sampled codes in output.

        Returns:
            VectorQuantizerOutput: Named tuple with fields as in VectorQuantizer.
        """
        z, z_shape = self.reshape_input(z)
        if self.in_proj is not None:
            z = self.in_proj(z)

        if temp is None:
            temp = self.temp.item()

        z_q, codes, soft_one_hot = self.decode_latents(z, temp, hard)

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
        ppl = self.codebook_perplexity_soft(soft_one_hot, z_mask)

        if self.reset_unused and self.training:
            self._reset_unused_codes(z, codes, z_mask)

        if self.out_proj is not None:
            z_q = self.out_proj(z_q)

        if not return_codes:
            codes = None

        z_q, codes = self.reshape_output(z_q, codes, z_shape)

        output = VectorQuantizerOutput(
            z_q=z_q,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
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
          their assigned codes.
        • Supports L2, L1, and cosine distance metrics for nearest-neighbor
          assignment.
        • Sequence masks (`z_mask`) and lengths (`z_lengths`) are supported.
        • Optional code re-initialization for unused embeddings.

    Args:
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
    ):
        super().__init__(
            in_feats,
            codebook_size,
            codebook_dim,
            distance_metric,
            use_weight_norm,
            channels_last,
            is_ema=True,
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
        return f"{self.__class__.__name__}(in_feats={self.in_feats}, codebook_size={self.codebook_size}, codebook_dim={self.codebook_dim}, distance_metric={self.distance_metric}, use_weight_norm={self.use_weight_norm}, channels_last={self.channels_last}, decay={self.decay}, eps={self.eps}, reset_unused={self.reset_unused})"

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

    @staticmethod
    def _maybe_allreduce(x: torch.Tensor) -> None:
        """
        All-reduce helper for multi-GPU training (DDP).

        If torch.distributed is initialized, sums the tensor `x` across all ranks
        in-place. Used to synchronize EMA cluster counts and embedding sums
        before updating the codebook.
        """
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM)

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

        self._maybe_allreduce(batch_cluster_size)
        self._maybe_allreduce(batch_embed_sum)

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

        z_q, codes = self.decode_latents(z)  # (B,T,D), (B,T)

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
        codebook_loss = torch.zeros_like(commitment_loss)
        ppl = self.codebook_perplexity_hard(codes, z_mask_2d)

        # Straight-through estimator for the path to encoder
        z_q = z + (z_q - z).detach()

        # EMA update (no grad) on flattened views
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

        # Restore shapes/layout
        z_q, codes = self.reshape_output(z_q, codes, orig_shape)

        return VectorQuantizerOutput(
            z_q=z_q,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
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


class EMAGumbelVectorQuantizer(EMANNVectorQuantizer):
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
        anneal_rate (float): Exponential decay rate for temperature scheduling.
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
        anneal_rate: float = 1e-5,
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
        )
        # Temperature parameters
        self.register_buffer("temp", torch.tensor(temp_init))
        self.register_buffer("temp_min", torch.tensor(temp_min))
        self.anneal_rate = anneal_rate

    def __str__(self):
        return f"{self.__class__.__name__}(in_feats={self.in_feats}, codebook_size={self.codebook_size}, codebook_dim={self.codebook_dim}, distance_metric={self.distance_metric}, use_weight_norm={self.use_weight_norm}, channels_last={self.channels_last}, decay={self.decay}, eps={self.eps}, reset_unused={self.reset_unused}, temp={self.temp.item():.4f}, temp_min={self.temp_min.item():.4f}, anneal_rate={self.anneal_rate})"

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "temp_init": self.temp.item(),
                "temp_min": self.temp_min.item(),
                "anneal_rate": self.anneal_rate,
            }
        )
        return cfg

    @torch.no_grad()
    def update_temp(self, global_step: int = 1) -> None:
        """
        Anneal temperature exponentially.

        Update rule:
            T = max(temp_min, temp * exp(-anneal_rate * step))

        Args:
            global_step (int): Number of elapsed training steps.
        """
        new_temp = torch.clamp(
            self.temp * torch.exp(-self.anneal_rate * global_step),
            min=self.temp_min.item(),
        )
        self.temp.copy_(new_temp)

    def encode_latents(
        self, latents: torch.Tensor, temp: float, hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode latents to soft Gumbel-Softmax assignments.

        Args:
            latents (Tensor): Input of shape (B, T, D).
            temp (float): Gumbel-Softmax temperature.
            hard (bool): If True, use straight-through (hard) sampling.

        Returns:
            (Tensor, Tensor):
                - codes (B, T): Argmax over sampled codes.
                - Soft assignments (N, codebook_size): One-hot or soft one-hot.
        """
        latents_shape = latents.shape
        latents = latents.view(-1, latents.shape[-1])  # (B*T, D)
        distance = self.compute_codebook_distance(latents)
        logits = -distance
        soft_one_hot = F.gumbel_softmax(
            logits, tau=float(temp), hard=hard, dim=-1
        )  # (N, num_embed)
        codes = soft_one_hot.max(dim=1)[1]  # (B*T)
        codes = codes.view(latents_shape[0], -1)  # (B, T)
        return codes, soft_one_hot

    def decode_latents(
        self, latents: torch.Tensor, temp: float, hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize latents using Gumbel-Softmax assignments.

        Args:
            latents (Tensor): Input of shape (B, T, D).
            temp (float): Gumbel-Softmax temperature.
            hard (bool): If True, apply straight-through discretization.

        Returns:
            (Tensor, Tensor, Tensor):
                - Quantized output (B, T, D).
                - codes (B, T).
                - Soft one-hot assignments (N, codebook_size).
        """
        codes, soft_one_hot = self.encode_latents(latents, temp, hard)
        z_q = torch.matmul(soft_one_hot, self.codebook).view(
            latents.shape
        )  # (B*T, D) -> (B, T, D)
        return z_q, codes, soft_one_hot

    def forward(
        self,
        z: torch.Tensor,
        z_lengths: torch.Tensor | None = None,
        z_mask: torch.Tensor | None = None,
        temp: float = None,
        hard: bool = False,
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
            hard (bool): If True, use straight-through (hard) discretization.
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
        z_q, codes, soft_one_hot = self.decode_latents(z, temp, hard)

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
        codebook_loss = torch.zeros_like(commitment_loss)
        ppl = self.codebook_perplexity_soft(soft_one_hot, z_mask_2d)

        # EMA update (no grad) on flattened views
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

        # Restore shapes/layout
        z_q, codes = self.reshape_output(z_q, codes, orig_shape)

        return VectorQuantizerOutput(
            z_q=z_q,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
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
