"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import time
import logging
from collections import OrderedDict

import numpy as np

from .logger import Logger


class ProgLogger(Logger):
    """Logger that prints training progress to stdout

    Attributes:
       metrics: list of metrics
       interval: number of batches between prints
    """

    def __init__(self, metrics=None, interval=10):
        super().__init__()

        self.metrics = None if metrics is None else set(metrics)

        self.interval = interval
        self.epochs = 0
        self.batches = 0
        self.samples = 0
        self.cur_epoch = 0
        self.cur_batch = 0
        self.cur_sample = 0
        self.t0 = 0

    def on_train_begin(self, logs=None, **kwargs):
        self.epochs = kwargs["epochs"]

    def on_epoch_begin(self, epoch, logs=None, **kwargs):
        if self.rank != 0:
            return

        self.cur_epoch = epoch
        logging.info("epoch: %d/%d starts" % (epoch + 1, self.epochs))
        if "samples" in kwargs:
            self.samples = kwargs["samples"] * self.world_size
        else:
            self.samples = 0

        if "batches" in kwargs:
            self.batches = kwargs["batches"]
        else:
            self.batches = 0

        self.cur_batch = 0
        self.cur_sample = 0
        self.t0 = time.time()

    def on_batch_begin(self, batch, logs=None, **kwargs):
        self.cur_batch = batch

    def on_batch_end(self, logs=None, **kwargs):
        if self.rank != 0:
            return

        batch_size = 0
        if "batch_size" in kwargs:
            batch_size = kwargs["batch_size"] * self.world_size
            self.cur_sample += batch_size

        self.cur_batch += 1

        if (self.cur_batch % self.interval) == 0:
            info = "epoch: %d/%d " % (self.cur_epoch + 1, self.epochs)
            etime, eta = self.estimate_epoch_time()
            if eta == None:
                info += " et: %s" % (etime)
            else:
                info += " et: %s eta: %s" % (etime, eta)

            if self.batches > 0:
                info += " batches: %d/%d(%d%%)" % (
                    self.cur_batch,
                    self.batches,
                    int(100 * self.cur_batch / self.batches),
                )
            else:
                info += " batches: %d" % (self.cur_batch)

            if self.cur_sample > 0:
                if self.samples > 0:
                    info += " samples: %d/%d(%d%%)" % (
                        self.cur_sample,
                        self.samples,
                        int(100 * self.cur_sample / self.samples),
                    )
                else:
                    info += " samples: %d" % (self.cur_sample)

            for k, v in logs.items():
                if self.metrics is None or k in self.metrics:
                    info += " %s: %.6f" % (k, v)

            logging.info(info)

    def on_epoch_end(self, logs=None, **kwargs):
        if self.rank != 0:
            return

        info = "epoch: %d/%d " % (self.cur_epoch + 1, self.epochs)
        for k, v in logs.items():
            if self.metrics is None or k in self.metrics:
                info += " %s: %.6f" % (k, v)

    def estimate_epoch_time(self):
        t1 = time.time()
        et = t1 - self.t0
        if self.batches > 0:
            total_t = et / self.cur_batch * self.batches
        elif self.samples > 0 and self:
            total_t = et / self.cur_sample * self.samples
        else:
            total_t = -1

        etime = self.sec2str(et)
        if total_t == -1:
            eta = None
        else:
            eta = self.sec2str(total_t - et)

        return etime, eta

    @staticmethod
    def sec2str(t):
        t = time.gmtime(t)
        if t.tm_mday > 1:
            st = "%d:%02d:%02d:%02d" % (t.tm_mday - 1, t.tm_hour, t.tm_min, t.tm_sec)
        elif t.tm_hour > 0:
            st = "%d:%02d:%02d" % (t.tm_hour, t.tm_min, t.tm_sec)
        elif t.tm_min > 0:
            st = "%d:%02d" % (t.tm_min, t.tm_sec)
        else:
            st = "%ds" % (t.tm_sec)

        return st
