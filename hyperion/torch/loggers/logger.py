"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import torch.distributed as dist


class Logger(object):
    """Base class for logger objects

    Attributes:
       params: training params dictionary
    """

    def __init__(self):
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1
        self.cur_epoch = 0
        self.cur_batch = 0
        self.params = None
        self.rank = rank
        self.world_size = world_size

    def on_epoch_begin(self, epoch, logs, **kwargs):
        """At the start of an epoch

        Args:
           epoch: index of the epoch
           logs: dictionary of logs
        """
        self.cur_epoch = epoch

    def on_epoch_end(self, logs, **kwargs):
        """At the end of an epoch

        Args:
           logs: dictionary of logs
        """
        pass

    def on_batch_begin(self, batch, logs, **kwargs):
        """At the start of a batch

        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        self.cur_batch = batch

    def on_batch_end(self, logs, **kwargs):
        """At the end of a batch

        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        pass

    def on_train_begin(self, logs, **kwargs):
        """At the start of training

        Args:
           logs: dictionary of logs
        """
        pass

    def on_train_end(self, logs, **kwargs):
        """At the end of training

        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        pass
