"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.distributed as dist


class FeatureMemoryBank:
    """
    A circular FIFO memory bank for contrastive learning, storing feature vectors.
    """

    def __init__(self, size: int, dim: int, device: torch.device):
        self.queue = torch.randn(size, dim, device=device)
        self.ptr = 0
        self.size = size

    def update(self, new_feats: torch.Tensor) -> None:
        """
        Add new features to the memory bank (with wrap-around if necessary).

        Args:
            new_feats (Tensor): New features to enqueue (shape: B x D)
        """
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
        """
        Gathers tensor across all distributed ranks (no gradient).
        """
        if not dist.is_available() or not dist.is_initialized():
            return tensor.clone()

        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor.detach())
        return torch.cat(gathered, dim=0)

    def get(self) -> torch.Tensor:
        """
        Get a copy of the current memory bank.

        Returns:
            Tensor: Memory bank of shape (size, dim)
        """
        return self.gather_without_grad(self.queue).detach()
