"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional
import torch.distributed as dist


class Logger:
    """Base class for logger objects

    Attributes:
       params: training params dictionary
    """

    def __init__(self) -> None:
        """Initializes logger state and distributed rank information."""
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        self.cur_epoch = 0
        self.cur_batch = 0
        self.cur_step = 0
        self.params = None
        self.rank = rank
        self.world_size = world_size

    def on_epoch_begin(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """At the start of an epoch

        Args:
           epoch: index of the epoch
           logs: dictionary of logs
        """
        self.cur_epoch = epoch
        self.cur_batch = 0

    def on_epoch_end(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """At the end of an epoch

        Args:
           logs: dictionary of logs
        """
        pass

    def on_val_end(self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """At the end of validation

        Args:
           logs: dictionary of logs
        """
        pass

    def on_batch_begin(
        self, batch: int, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """At the start of a batch

        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        self.cur_batch = batch

    def on_batch_end(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """At the end of a batch

        Args:
           logs: dictionary of logs
        """
        pass

    def on_model_update(
        self, step: int, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """At the end of a model update

        Args:
           step: index of the step
           logs: dictionary of logs
        """
        self.cur_step = step

    def on_train_begin(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """At the start of training

        Args:
           logs: dictionary of logs
        """
        if "epoch" in kwargs:
            self.cur_epoch = kwargs["epoch"]
        if "step" in kwargs:
            self.cur_step = kwargs["step"]

    def on_train_end(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """At the end of training

        Args:
           logs: dictionary of logs
        """
        pass
