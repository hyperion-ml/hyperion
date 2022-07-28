import math
from jsonargparse import ArgumentParser, ActionParser
import logging

import numpy as np

import torch
from torch.utils.data import Sampler
import torch.distributed as dist


class HypSampler(Sampler):
    def __init__(self, shuffle=False, seed=1234):
        super().__init__(None)
        self.epoch = 0
        self.batch = 0
        self.shuffle = shuffle
        self.seed = seed

        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1

        self.rank = rank
        self.world_size = world_size
        self.rng = torch.Generator()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _set_seed(self):
        if self.shuffle:
            self.rng.manual_seed(self.seed + 10 * self.epoch)
        else:
            self.rng.manual_seed(self.seed)

    def __iter__(self):
        self.batch = 0
        self._set_seed()
        return self
