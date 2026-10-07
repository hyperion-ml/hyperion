"""Sketched isotropic Gaussian regularization from LeJEPA, Algorithm 1."""

import math
from typing import Callable, Literal

import torch
import torch.distributed as dist
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch import Tensor, nn
from torch.distributed.nn.functional import all_reduce
from torch.utils.checkpoint import checkpoint

from ...utils.misc import filter_func_args
from ..utils.masking import seq_lengths_to_mask


class SIGReg(nn.Module):
    """Match random projections of raw embeddings to a standard Gaussian.

    Implements the Epps--Pulley statistic in LeJEPA Algorithm 1
    (https://arxiv.org/abs/2511.08544). Each view is tested separately and
    sequence frames are pooled over valid time steps and samples within a view.
    View losses are reduced according to reduction; directions are always averaged.
    Inputs are never centered, standardized, whitened, or length-normalized.

    In global-data mode, all ranks must use the same configuration, device
    type, counter, and forward/backward call order, including the view count.
    The differentiable SUM collective sums backward contributions from the
    replicated global losses. Standard DDP parameter-gradient averaging cancels
    that world-size factor, matching the concatenated single-process objective.
    Local embedding gradients consequently have a world-size factor in this
    mode; no extra loss scaling should be applied with standard DDP.

    Attributes:
        training: Module training flag (inherited); sampling advances in both modes.
        num_slices: Number of independently sampled unit projection directions.
        num_points: Number of equally spaced frequency nodes.
        t_max: Positive endpoint of the symmetric frequency interval.
        seed: Base seed for the dedicated device generator.
        multi_view: Whether the leading input axis indexes views.
        projection_chunk_size: Maximum directions per checkpointed chunk, or None.
        distributed_mode: Local or globally aggregated statistics.
        reduction: Reduction across views: mean, sum, or none.
        t: Float32 frequency-node buffer.
        phi: Float32 standard Gaussian characteristic-function buffer.
        weights: Float32 trapezoidal weights including the Gaussian window.
        counter: Checkpointed number of completed forward calls.
    """

    def __init__(
        self,
        num_slices: int = 256,
        num_points: int = 17,
        t_max: float = 5.0,
        seed: int = 0,
        multi_view: bool = False,
        projection_chunk_size: int | None = None,
        distributed_mode: Literal["local", "global_data"] = "global_data",
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        """Initialize quadrature buffers and sampling configuration.

        Args:
            num_slices: Projection count, e.g. 256 or 1024.
            num_points: Frequency count, at least two.
            t_max: Finite positive frequency endpoint; nodes span [-t_max, t_max].
            seed: Base seed; each forward uses seed + counter.
            multi_view: Require inputs with a leading view axis when True.
            projection_chunk_size: Optional positive projection chunk size.
                Checkpointing recomputes local statistics during backward.
            distributed_mode: Global-data reduces sums and counts across the
                default process group. Local suits replicated class prototypes.
            reduction: Reduction across views. None returns one loss per view;
                single-view inputs always return a scalar.
        """
        super().__init__()
        for name, value, minimum in (
            ("num_slices", num_slices, 1),
            ("num_points", num_points, 2),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(t_max) or t_max <= 0:
            raise ValueError("t_max must be finite and positive")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if not isinstance(multi_view, bool):
            raise ValueError("multi_view must be boolean")
        if projection_chunk_size is not None and (
            isinstance(projection_chunk_size, bool)
            or not isinstance(projection_chunk_size, int)
            or projection_chunk_size <= 0
        ):
            raise ValueError("projection_chunk_size must be a positive integer or None")
        if distributed_mode not in ("local", "global_data"):
            raise ValueError("distributed_mode must be 'local' or 'global_data'")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.num_slices = num_slices
        self.num_points = num_points
        self.t_max = t_max
        self.seed = seed
        self.multi_view = multi_view
        self.projection_chunk_size = projection_chunk_size
        self.distributed_mode = distributed_mode
        self.reduction = reduction
        t = torch.linspace(-t_max, t_max, num_points, dtype=torch.float32)
        phi = torch.exp(-0.5 * t.square())
        weights = torch.full_like(t, 2 * t_max / (num_points - 1))
        weights[[0, -1]] *= 0.5
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)
        self.register_buffer("counter", torch.zeros((), dtype=torch.long))
        self._generator: torch.Generator | None = None

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> "SIGReg":
        """Move buffers while retaining original float32 quadrature precision.

        Args:
            fn: Tensor conversion supplied by Module.to and related methods.
            recurse: Whether to convert child modules.

        Returns:
            This module.
        """
        quadrature = {name: getattr(self, name) for name in ("t", "phi", "weights")}
        super()._apply(fn, recurse=recurse)
        for name, value in quadrature.items():
            self._buffers[name] = value.to(device=self.t.device, dtype=torch.float32)
        return self

    def _prepare_views(self, z: Tensor, z_lengths: Tensor | None) -> tuple[Tensor, ...]:
        """Validate shapes and select valid frames for each view.

        Args:
            z: Embeddings with the configured leading axes.
            z_lengths: Optional torch.long valid-frame counts.

        Returns:
            Rank-two tensors, one per view.
        """
        ranks = (3, 4) if self.multi_view else (2, 3)
        if z.ndim not in ranks or z.shape[-1] < 1 or not z.is_floating_point():
            raise ValueError(
                f"z must be floating point with rank in {ranks} and D >= 1"
            )
        if self.multi_view and z.shape[0] == 0:
            raise ValueError("z must contain at least one view")
        sequence = z.ndim == ranks[1]
        if z_lengths is not None:
            if not sequence:
                raise ValueError("z_lengths is only accepted for sequence inputs")
            if z_lengths.shape != z.shape[:-2]:
                raise ValueError("z_lengths must have shape z.shape[:-2]")
            if z_lengths.dtype != torch.long:
                raise ValueError("z_lengths must have dtype torch.long")
            if z_lengths.device != z.device:
                raise ValueError("z_lengths must be on the same device as z")
        views = z.unbind(0) if self.multi_view else (z,)
        if not sequence:
            return views
        if z_lengths is None:
            return tuple(view.reshape(-1, z.shape[-1]) for view in views)
        lengths = z_lengths.unbind(0) if self.multi_view else (z_lengths,)
        return tuple(
            view[seq_lengths_to_mask(length, max_length=z.shape[-2])]
            for view, length in zip(views, lengths)
        )

    def _use_distributed_global_data(self) -> bool:
        """Check whether the default process group should aggregate statistics.

        Returns:
            True when global aggregation across multiple ranks is active.
        """
        return (
            self.distributed_mode == "global_data"
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

    def _sample_directions(self, embedding_dim: int, device: torch.device) -> Tensor:
        """Sample directions without changing application RNG state.

        Args:
            embedding_dim: Number of embedding coordinates.
            device: Embedding device, also used for the cached generator.

        Returns:
            Float32 unit directions with shape [D, M].
        """
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
        self._generator.manual_seed((self.seed + int(self.counter.item())) % (2**63))
        directions = torch.randn(
            embedding_dim,
            self.num_slices,
            device=device,
            dtype=torch.float32,
            generator=self._generator,
        )
        return nn.functional.normalize(directions, dim=0)

    def _characteristic_sums(self, z: Tensor, directions: Tensor, t: Tensor) -> Tensor:
        """Compute local cosine/sine sums without collecting embeddings.

        Args:
            z: Float32 valid embeddings [N_local, D].
            directions: Unit projection chunk [D, M_chunk].
            t: Float32 frequency nodes [T].

        Returns:
            Stacked real and imaginary sums [2, M_chunk, T].
        """
        with torch.autocast(device_type=z.device.type, enabled=False):
            # For h[n, m] = z[n] @ u[m], compute the unnormalized ECF parts:
            # C[m, j] = sum_n cos(t[j] * h[n, m]),
            # S[m, j] = sum_n sin(t[j] * h[n, m]).
            phase = (z @ directions).unsqueeze(-1) * t
            return torch.stack((phase.cos().sum(0), phase.sin().sum(0)))

    def _compute_view_loss(self, z: Tensor, directions: Tensor) -> Tensor:
        """Compute the batch-scaled projection mean for one view.

        Args:
            z: Float32 valid embeddings [N_local, D].
            directions: Shared unit directions [D, M].

        Returns:
            Scalar float32 Epps--Pulley loss.
        """
        distributed = self._use_distributed_global_data()
        count = torch.tensor(z.shape[0], device=z.device, dtype=torch.long)
        if distributed:
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        if count.item() == 0:
            raise ValueError("each view must contain at least one global valid sample")
        t = self.t.to(z.device)
        phi = self.phi.to(z.device)
        weights = self.weights.to(z.device)
        total = z.new_zeros(())
        chunk_size = self.projection_chunk_size or self.num_slices
        for chunk in directions.split(chunk_size, dim=1):
            if (
                self.projection_chunk_size is not None
                and z.requires_grad
                and torch.is_grad_enabled()
            ):
                sums = checkpoint(
                    self._characteristic_sums,
                    z,
                    chunk,
                    t,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                sums = self._characteristic_sums(z, chunk, t)
            if distributed:
                # SUM in both forward and backward. Replicated loss backward
                # contributes W times the local gradient, canceled by DDP's /W.
                sums = all_reduce(sums, op=dist.ReduceOp.SUM)
            # With N=sum_r N_r, global ECF is (C/N) + i(S/N), where C and S
            # are the global cosine and sine sums. For each projection m:
            # L_m = N * sum_j q[j] * ((C[m,j]/N - phi[j])**2
            #                           + (S[m,j]/N)**2),
            # where q[j] includes the trapezoidal coefficient and Gaussian
            # weight w(t[j])=phi[j]=exp(-t[j]**2/2). Average L_m over M.
            means = sums / count
            error = (means[0] - phi).square() + means[1].square()
            total = total + (error * weights).sum()
        return total * count / self.num_slices

    def apply_reduction(self, view_losses: Tensor) -> Tensor:
        """Reduce per-view losses according to the configured reduction.

        Args:
            view_losses: Scalar loss for each view, shape [V].

        Returns:
            Reduced scalar, or the original [V] tensor for ``reduction='none'``.
                Single-view inputs always return a scalar.
        """
        if not self.multi_view or self.reduction == "mean":
            return view_losses.mean()
        if self.reduction == "sum":
            return view_losses.sum()
        return view_losses

    def forward(self, z: Tensor, z_lengths: Tensor | None = None) -> Tensor:
        """Evaluate separate view losses with one shared projection draw.

        Args:
            z: [B, D] or [B, T, D] when multi_view=False; [V, B, D]
                or [V, B, T, D] otherwise. Embeddings are unnormalized.
            z_lengths: torch.long counts [B] or [V, B] for sequence inputs.
                Each sequence contributes its first length frames. None means
                all time steps are valid. Longer sequences have greater weight.

        Returns:
            Float32 scalar after view reduction, or [V] for multi-view inputs
            with reduction='none'. Single-view inputs always return a scalar.
            Each view retains its own
            global valid-sample multiplier. Empty local views are allowed in
            global-data mode if their global count is positive.
        """
        views = self._prepare_views(z, z_lengths)
        with torch.autocast(device_type=z.device.type, enabled=False):
            directions = self._sample_directions(z.shape[-1], z.device)
            view_losses = torch.stack(
                [self._compute_view_loss(view.float(), directions) for view in views]
            )
            loss = self.apply_reduction(view_losses)
        self.counter.add_(1)
        return loss

    def get_config(self) -> dict:
        """Return constructor configuration; sampling progress is in state_dict.

        Returns:
            JSON-friendly constructor arguments.
        """
        return {
            name: getattr(self, name)
            for name in (
                "num_slices",
                "num_points",
                "t_max",
                "seed",
                "multi_view",
                "projection_chunk_size",
                "distributed_mode",
                "reduction",
            )
        }

    @staticmethod
    def filter_args(**kwargs: object) -> dict:
        """Filter constructor keyword arguments.

        Args:
            **kwargs: Candidate constructor arguments.

        Returns:
            Accepted constructor arguments.
        """
        return filter_func_args(SIGReg.__init__, kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: str | None = None,
        skip: set[str] | None = None,
    ) -> None:
        """Add CLI arguments using the existing loss parser convention.

        Args:
            parser: Parser to extend.
            prefix: Optional nested configuration section.
            skip: Constructor argument names to omit from the parser.
        """
        if skip is None:
            skip = set()
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")
        if "num_slices" not in skip:
            parser.add_argument(
                "--num-slices", type=int, default=256, help="Random projection count"
            )
        if "num_points" not in skip:
            parser.add_argument(
                "--num-points", type=int, default=17, help="Frequency-node count"
            )
        if "t_max" not in skip:
            parser.add_argument(
                "--t-max", type=float, default=5.0, help="Symmetric frequency endpoint"
            )
        if "seed" not in skip:
            parser.add_argument(
                "--seed", type=int, default=0, help="Dedicated generator base seed"
            )
        if "multi_view" not in skip:
            parser.add_argument(
                "--multi-view",
                action=ActionYesNo,
                default=False,
                help="Leading axis indexes views",
            )
        if "projection_chunk_size" not in skip:
            parser.add_argument(
                "--projection-chunk-size",
                type=int,
                default=None,
                help="Directions per checkpointed chunk",
            )
        if "distributed_mode" not in skip:
            parser.add_argument(
                "--distributed-mode",
                choices=["local", "global_data"],
                default="global_data",
                help="Statistics aggregation mode",
            )
        if "reduction" not in skip:
            parser.add_argument(
                "--reduction",
                choices=["mean", "sum", "none"],
                default="mean",
                help="Reduction across views; directions are always averaged",
            )
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
