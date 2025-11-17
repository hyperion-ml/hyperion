"""
Lightweight tensor-parallel linear layers built on top of PyTorch collectives.

These implementations mimic the FairScale Column/Row parallel layers but only rely on
PyTorch's distributed primitives so we can avoid the FairScale dependency.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn


def _dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _resolve_process_group(
    process_group: Optional[dist.ProcessGroup],
) -> Optional[dist.ProcessGroup]:
    if process_group is not None:
        return process_group
    if _dist_initialized():
        return dist.group.WORLD
    return None


def get_tensor_parallel_world_size(
    process_group: Optional[dist.ProcessGroup] = None,
) -> int:
    group = _resolve_process_group(process_group)
    if group is None:
        return 1
    return dist.get_world_size(group)


def get_tensor_parallel_rank(
    process_group: Optional[dist.ProcessGroup] = None,
) -> int:
    group = _resolve_process_group(process_group)
    if group is None:
        return 0
    return dist.get_rank(group)


def _gather_from_parallel_region(
    tensor: torch.Tensor,
    process_group: Optional[dist.ProcessGroup],
    world_size: int,
) -> torch.Tensor:
    if process_group is None or world_size == 1:
        return tensor
    gathered = dist_nn.all_gather(tensor, group=process_group)
    return torch.cat(gathered, dim=-1)


def _split_to_parallel_region(
    tensor: torch.Tensor,
    world_size: int,
    rank: int,
) -> torch.Tensor:
    if world_size == 1:
        return tensor
    if tensor.size(-1) % world_size != 0:
        raise ValueError(
            f"Cannot shard dimension {tensor.size(-1)} into {world_size} ranks."
        )
    chunks = tensor.chunk(world_size, dim=-1)
    return chunks[rank].contiguous()


class ColumnParallelLinear(nn.Module):
    """Linear layer sharded across the output (column) dimension."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        input_is_parallel: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.process_group = _resolve_process_group(process_group)
        self.world_size = get_tensor_parallel_world_size(self.process_group)
        self.in_features = in_features
        if out_features % self.world_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by tensor parallel world size ({self.world_size})."
            )
        self.output_size_per_partition = out_features // self.world_size
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                in_features,
                **factory_kwargs,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.output_size_per_partition, **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.gather_output = gather_output
        self.input_is_parallel = input_is_parallel
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel and self.world_size > 1:
            expected = self.in_features // self.world_size
            if input.size(-1) != expected:
                raise ValueError(
                    f"Expected parallel input with {expected} features, "
                    f"but received {input.size(-1)}."
                )
            input = _gather_from_parallel_region(
                input, self.process_group, self.world_size
            )

        output_parallel = nn.functional.linear(input, self.weight, self.bias)

        if self.gather_output and self.world_size > 1:
            output_parallel = _gather_from_parallel_region(
                output_parallel, self.process_group, self.world_size
            )
        return output_parallel


class RowParallelLinear(nn.Module):
    """Linear layer sharded across the input (row) dimension."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.process_group = _resolve_process_group(process_group)
        self.world_size = get_tensor_parallel_world_size(self.process_group)
        self.rank = get_tensor_parallel_rank(self.process_group)
        if in_features % self.world_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by tensor parallel world size ({self.world_size})."
            )
        self.input_size_per_partition = in_features // self.world_size
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                self.input_size_per_partition,
                **factory_kwargs,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.input_is_parallel = input_is_parallel
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel and input.size(-1) != self.input_size_per_partition:
            raise ValueError(
                f"Expected shard width {self.input_size_per_partition}, "
                f"but received {input.size(-1)} features."
            )
        if not self.input_is_parallel and self.world_size > 1:
            input = _split_to_parallel_region(input, self.world_size, self.rank)

        output_parallel = nn.functional.linear(input, self.weight, bias=None)
        if self.world_size > 1 and self.process_group is not None:
            dist.all_reduce(output_parallel, group=self.process_group)

        if self.bias is not None:
            output_parallel = output_parallel + self.bias
        return output_parallel
