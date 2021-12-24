"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import re
import os

try:
    import wandb
except:
    pass

from .logger import Logger


class WAndBLogger(Logger):
    """Logger that sends training progress to weights and biases (wandb)

    Attributes:
       tb_path: tensorboard output directory

    """

    def __init__(
        self, project=None, group=None, name=None, path=None, mode="online", interval=10
    ):
        super().__init__()
        self.project = project
        self.path = path
        self.name = name
        self.group = group
        self.mode = mode
        self.interval = interval
        self.batches = 0
        self.cur_epoch = 0
        self.cur_batch = 0

    def on_train_begin(self, logs=None, **kwargs):
        if self.rank != 0:
            return

        if self.path is not None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)

        wandb.init(
            project=self.project,
            group=self.group,
            name=self.name,
            dir=self.path,
            mode=self.mode,
            reinit=True,
        )

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
            logs = {k + "/global_steps": v for k, v in logs.items()}
            logs["batch"] = step
            wandb.log(logs)
            # for k,v in logs.items():
            #     self.writer.add_scalar(k+'/global_steps', v, step)

    def on_epoch_end(self, logs=None, **kwargs):
        if self.rank != 0:
            return

        logs = {re.sub(r"^(train|val)_(.*)$", r"\2/\1", k): v for k, v in logs.items()}
        logs["epoch"] = self.cur_epoch + 1
        wandb.log(logs)
        # for k,v in logs.items():
        #     k = re.sub(r'^(train|val)_(.*)$', r'\2/\1', k)
        #     self.writer.add_scalar(k, v, self.cur_epoch+1)

    def on_train_end(self, logs=None, **kwargs):
        if self.rank != 0:
            return

        wandb.finish()
