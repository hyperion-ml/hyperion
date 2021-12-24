"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import torch.distributed as dist

from .tensorboard_logger import TensorBoardLogger as TBL


class LoggerList(object):
    """Container for a list of logger callbacks

    Attributes:
       loggers: list of Logger objects
    """

    def __init__(self, loggers=None):
        self.loggers = loggers or []

    def append(self, logger):
        self.loggers.append(logger)

    @property
    def tensorboard_logger(self):
        for l in self.loggers:
            if isinstance(l, TBL):
                return l

    @property
    def tensorboard_writer(self):
        for l in self.loggers:
            if isinstance(l, TBL):
                return l.writer

    def on_epoch_begin(self, epoch, logs=None, **kwargs):
        """At the start of an epoch

        Args:
           epoch: index of the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_epoch_begin(epoch, logs, **kwargs)

    def on_epoch_end(self, logs=None, **kwargs):
        """At the end of an epoch

        Args:
           epoch: index of the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_epoch_end(logs, **kwargs)

    def on_batch_begin(self, batch, logs=None, **kwargs):
        """At the start of a batch

        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_batch_begin(batch, logs, **kwargs)

    def on_batch_end(self, logs=None, **kwargs):
        """At the end of a batch

        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_batch_end(logs, **kwargs)

    def on_train_begin(self, logs=None, **kwargs):
        """At the start of training

        Args:
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_train_begin(logs, **kwargs)

    def on_train_end(self, logs=None, **kwargs):
        """At the end of training

        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_train_end(logs, **kwargs)

    def __iter__(self):
        return iter(self.loggers)
