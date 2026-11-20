"""Utilities for FSDP2 with CPU-resident parameter and gradient shards."""

from collections import defaultdict
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.utils._pytree import tree_map_with_path


class CPUOffloadGradScaler(ShardedGradScaler):
    """Unscale local CPU shards and let the scaler reduce overflow on CUDA."""

    @torch.no_grad()
    def _unscale_grads_(
        self,
        optimizer: torch.optim.Optimizer,
        inv_scale: torch.Tensor,
        found_inf: torch.Tensor,
        allow_fp16: bool = True,
    ) -> dict[torch.device, torch.Tensor]:
        # DTensor unscaling reduces through its CUDA mesh. Run the kernel on
        # the CPU shards instead, then ShardedGradScaler reduces found_inf.
        per_dtype = defaultdict(list)
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    grad = param.grad.to_local()
                    per_dtype[grad.dtype].append(grad)
        cpu_scale = inv_scale.cpu()
        cpu_found_inf = found_inf.cpu()
        for grads in per_dtype.values():
            torch._amp_foreach_non_finite_check_and_unscale_(
                grads, cpu_found_inf, cpu_scale
            )
        return {cpu_found_inf.device: cpu_found_inf}


@torch.no_grad()
def clip_cpu_gradient_shards_(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    norm_type: float,
    device: torch.device,
) -> torch.Tensor:
    """Clip CPU shards using the global norm, reducing only their norm data."""
    grads = [p.grad.to_local() for p in parameters if p.grad is not None]
    if not grads:
        return torch.zeros(())

    norm_type = float(norm_type)
    if norm_type == 0:
        # A parameter contributes once, even when multiple shards are nonzero.
        nonzero = torch.stack([torch.count_nonzero(g) > 0 for g in grads]).to(device)
        dist.all_reduce(nonzero, op=dist.ReduceOp.MAX)
        total_norm = nonzero.sum().float().cpu()
    else:
        norms = torch.stack(
            [
                torch.linalg.vector_norm(grad, norm_type)
                for grad in grads
                if grad.numel()
            ]
            or [torch.tensor(float("inf") if norm_type < 0 else 0.0)]
        )
        if norm_type == float("inf"):
            norm = norms.max().to(device)
            dist.all_reduce(norm, op=dist.ReduceOp.MAX)
        elif norm_type == -float("inf"):
            norm = norms.min().to(device)
            dist.all_reduce(norm, op=dist.ReduceOp.MIN)
        else:
            norm = norms.pow(norm_type).sum().to(device)
            dist.all_reduce(norm, op=dist.ReduceOp.SUM)
            norm = norm.pow(1 / norm_type)
        total_norm = norm.cpu()

    torch._foreach_mul_(grads, (max_norm / (total_norm + 1e-6)).clamp(max=1.0))
    return total_norm


def _shard_state_tensor(value: Any, local: Any) -> Any:
    if isinstance(local, DTensor):
        # Use NCCL on the CUDA mesh and put the result back on the shard's
        # original CPU device before the loader copies it into the destination.
        return distribute_tensor(value, local.device_mesh, local.placements).to(
            local.device
        )
    return value


def load_cpu_offloaded_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    checkpoint: dict[str, Any],
) -> None:
    """Restore full checkpoint tensors into CPU-offloaded DTensor state.

    Args:
        model: Sharded model with its CPU offload policy already installed.
        optimizer: Optimizer for the sharded parameters, if being restored.
        checkpoint: Full CPU checkpoint, read by each rank.
    """
    local_model = get_model_state_dict(model)
    model_state = {
        name: _shard_state_tensor(value, local_model.get(name))
        for name, value in checkpoint["model_state_dict"].items()
    }
    set_model_state_dict(model, model_state)
    if optimizer is not None:
        local_optim = get_optimizer_state_dict(model, optimizer)

        def shard_optimizer_value(path, value):
            local = local_optim
            for key in path:
                try:
                    local = key.get(local)
                except (KeyError, IndexError, TypeError):
                    return value
            return _shard_state_tensor(value, local)

        optim_state = tree_map_with_path(
            shard_optimizer_value, checkpoint["optimizer_state_dict"]
        )
        set_optimizer_state_dict(model, optimizer, optim_state)
