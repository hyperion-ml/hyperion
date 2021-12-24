"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from collections import OrderedDict as ODict
import numpy as np

import torch
import torch.distributed as dist


class MetricAcc(object):
    """Class to accumulate metrics during an epoch."""

    def __init__(self, device=None):
        self.keys = None
        self.acc = None
        self.count = 0
        self.device = device
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1
        self.rank = rank
        self.world_size = world_size

    def reset(self):
        """Resets the accumulators."""
        self.count = 0
        if self.acc is not None:
            self.acc[:] = 0

    def _reduce(self, metrics):
        if self.world_size == 1:
            return

        metrics_list = [v for k, v in metrics.items()]
        metrics_tensor = torch.tensor(metrics_list, device=self.device)
        dist.reduce(metrics_tensor, 0, op=dist.ReduceOp.SUM)
        metrics_tensor /= self.world_size
        for i, k in enumerate(metrics.keys()):
            metrics[k] = metrics_tensor[i]

    def update(self, metrics, num_samples=1):
        """Updates the values of the metric

        It uses recursive formula, it may be more numerically stable

            m^(i) = m^(i-1) + n^(i)/sum(n^(i)) (x^(i) - m^(i-1))

           where i is the batch number,
           m^(i) is the accumulated average of the metric at batch i,
           x^(i) is the average of the metric at batch i,
           n^(i) is the batch_size at batch i.

        Args:
            metrics: dictionary with metrics for current batch
            num_samples: number of samples in current batch (batch_size)
        """
        self._reduce(metrics)
        if self.rank != 0:
            return

        if self.keys is None:
            self.keys = metrics.keys()
            self.acc = np.zeros((len(self.keys),))

        self.count += num_samples
        r = num_samples / self.count
        for i, k in enumerate(self.keys):
            self.acc[i] += r * (metrics[k] - self.acc[i])

    @property
    def metrics(self):
        """Returns metrics dictionary"""
        if self.rank != 0:
            return {}
        logs = ODict()
        for i, k in enumerate(self.keys):
            logs[k] = self.acc[i]

        return logs
