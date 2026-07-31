"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.distributed as dist


class FeatureMemoryBank:
    """Circular FIFO memory bank for feature vectors.

    The bank stores a fixed number of feature rows and overwrites the oldest rows
    when full. In distributed mode, :meth:`get` returns the concatenation of local
    banks from all ranks (no gradient tracking).

    Attributes:
        queue: Tensor of shape ``(size, dim)`` storing the local memory bank.
        ptr: Circular write pointer indicating the next insertion position.
        size: Maximum number of feature rows stored in the local bank.
    """

    def __init__(self, size: int, dim: int, device: torch.device):
        """Create a memory bank.

        Args:
            size: Number of feature rows stored per rank.
            dim: Feature dimension ``D`` for each row.
            device: Device where the queue tensor is allocated.
        """
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")

        self.queue = torch.randn(size, dim, device=device)
        self.ptr = 0
        self.size = size
        self.dim = dim

    def update(self, new_feats: torch.Tensor) -> None:
        """Enqueue a batch of features into the circular buffer.

        Args:
            new_feats: Feature batch with shape ``(B, D)``. ``B`` must be
                ``<= self.size``.

        Raises:
            ValueError: If feature shape/device is invalid or ``B > self.size``.
        """
        if new_feats.dim() != 2:
            raise ValueError(
                f"new_feats must be a 2D tensor of shape (B, D), got {new_feats.dim()}D"
            )
        if new_feats.size(1) != self.dim:
            raise ValueError(
                f"Feature dim mismatch: expected D={self.dim}, got D={new_feats.size(1)}"
            )
        if new_feats.device != self.queue.device:
            raise ValueError(
                f"Device mismatch: expected {self.queue.device}, got {new_feats.device}"
            )

        batch_size = new_feats.size(0)
        if batch_size > self.size:
            raise ValueError(
                f"Batch size {batch_size} exceeds memory bank size {self.size}"
            )

        end_ptr = self.ptr + batch_size
        if end_ptr <= self.size:
            # No wrap-around
            self.queue[self.ptr : end_ptr] = new_feats.detach()
        else:
            # Wrap-around needed
            first_part = self.size - self.ptr
            second_part = end_ptr - self.size
            self.queue[self.ptr :] = new_feats[:first_part].detach()
            self.queue[:second_part] = new_feats[first_part:].detach()
        self.ptr = (self.ptr + batch_size) % self.size

    @staticmethod
    def gather_without_grad(tensor: torch.Tensor) -> torch.Tensor:
        """Gather a tensor from all distributed ranks without gradients.

        Args:
            tensor: Local tensor to gather. Assumed to have the same shape on all
                ranks.

        Returns:
            Concatenated tensor across ranks along dim 0. If distributed is not
            initialized, returns a clone of the input tensor.
        """
        if not dist.is_available() or not dist.is_initialized():
            return tensor.clone()

        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor.detach())
        return torch.cat(gathered, dim=0)

    def get(self) -> torch.Tensor:
        """Return a detached snapshot of the current memory bank.

        Returns:
            Local queue of shape ``(size, dim)`` when not distributed, otherwise
            gathered queue of shape ``(world_size * size, dim)``.
        """
        return self.gather_without_grad(self.queue).detach()
