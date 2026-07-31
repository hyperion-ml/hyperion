"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn


class GatherDistributedFunction(torch.autograd.Function):
    """
    Autograd-capable all-gather operation for distributed contrastive learning.

    This function gathers tensors from all distributed processes and allows
    gradients to flow back only to the local process's inputs.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        Gathers tensors from all ranks into a single tensor.

        Args:
            x (Tensor): Input tensor from the local rank.

        Returns:
            Tensor: Concatenated tensor from all ranks.
        """
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return x  # single GPU fallback

        ctx.batch_size = x.shape[0]
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Distributes gradient only to the local batch.

        Args:
            grad_output (Tensor): Gradient of the output tensor.

        Returns:
            Tensor: Gradient of the input tensor for this rank only.
        """
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return grad_output  # no slicing needed for single GPU

        # Only return the gradient for the local batch
        rank = torch.distributed.get_rank()
        start = rank * ctx.batch_size
        end = (rank + 1) * ctx.batch_size
        return grad_output[start:end]


class GatherDistributed(nn.Module):
    def __init__():
        """
        Wrapper module for GatherDistributedFunction to use in nn.Module.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method to apply the gather operation.

        Args:
            x (Tensor): Input tensor from the local rank.

        Returns:
            Tensor: Concatenated tensor from all ranks.
        """
        return GatherDistributedFunction.apply(x)
