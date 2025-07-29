"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class HypSampler(Sampler):
    """
    Base class for custom PyTorch samplers supporting distributed training,
    reproducible shuffling, and batch-level progress tracking.

    This sampler does not implement a specific sampling strategy itself,
    but provides a consistent foundation for:
      - distributed training using torch.distributed
      - reproducible shuffling with seeding
      - checkpointable epoch and batch counters

    Attributes:
        max_batches_per_epoch (Optional[int]): Optional limit on the number of batches per epoch.
        shuffle (bool): Whether to shuffle samples at the beginning of each epoch.
        seed (int): Random seed for reproducibility.
        rank (int): Local rank of the process in distributed training.
        world_size (int): Total number of distributed processes.
        epoch (int): Current epoch number.
        batch (int): Current batch index.
        init_batch (int): Starting batch index for the current epoch.
        rng (torch.Generator): Random generator for sampling with reproducible seed.
    """

    def __init__(
        self,
        max_batches_per_epoch: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 1234,
    ):
        super().__init__(None)
        self.epoch = 0
        self.batch = 0
        self.init_batch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.max_batches_per_epoch = max_batches_per_epoch

        # Detect distributed environment
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            # If torch.distributed not initialized, assume single process
            rank = 0
            world_size = 1

        self.rank = rank
        self.world_size = world_size
        # Random number generator (used if shuffling)
        self.rng = torch.Generator()

    def set_epoch(self, epoch: int, batch: int = 0):
        """
        Sets the current epoch and initial batch index.

        Args:
            epoch (int): Epoch number to resume from.
            batch (int): Optional starting batch index within the epoch.
        """
        self.epoch = epoch
        self.init_batch = batch

    def _set_seed(self):
        """
        Sets the random seed for the sampler's RNG.

        If shuffling is enabled, the seed is offset by epoch and batch index
        to ensure reproducibility across training sessions.
        """
        if self.shuffle:
            self.rng.manual_seed(self.seed + 10 * self.epoch + 100 * self.init_batch)
        else:
            self.rng.manual_seed(self.seed)

    def __iter__(self):
        """
        Entry point for iterating over the sampler. Must be overridden by subclasses.

        Returns:
            Iterator object (typically `self`, for use in subclasses).
        """
        self.batch = self.init_batch
        self.init_batch = 0
        self._set_seed()
        return self
