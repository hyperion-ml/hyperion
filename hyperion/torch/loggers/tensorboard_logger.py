"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import re
from torch.utils.tensorboard import SummaryWriter

from .logger import Logger


class TensorBoardLogger(Logger):
    """Logger that sends training progress to tensorboard

    Attributes:
       tb_path: tensorboard output directory

    """

    def __init__(self, tb_path, interval=10):
        super().__init__()
        self.tb_path = tb_path
        self.writer = None
        self.interval = interval
        self.batches = 0
        self.cur_epoch = 0
        self.cur_batch = 0

    def on_train_begin(self, logs=None, **kwargs):
        if self.rank != 0:
            return

        self.writer = SummaryWriter(self.tb_path)

    def on_epoch_begin(self, epoch, logs=None, **kwargs):
        if self.rank != 0:
            return

        self.cur_epoch = epoch
        if "batches" in kwargs:
            self.batches = kwargs["batches"]
        else:
            self.batches = 0

        self.cur_batch = 0

    def on_batch_end(self, logs=None, **kwargs):
        if self.rank != 0:
            return

        self.cur_batch += 1
        if (self.cur_batch % self.interval) == 0:
            step = self.cur_epoch * self.batches + self.cur_batch
            for k, v in logs.items():
                self.writer.add_scalar(k + "/global_steps", v, step)

    def on_epoch_end(self, logs=None, **kwargs):
        if self.rank != 0:
            return

        for k, v in logs.items():
            k = re.sub(r"^(train|val)_(.*)$", r"\2/\1", k)
            self.writer.add_scalar(k, v, self.cur_epoch + 1)

    def on_train_end(self, logs=None, **kwargs):
        if self.rank != 0:
            return

        self.writer.close()
        self.writer = None
