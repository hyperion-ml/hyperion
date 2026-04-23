"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import time
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from .logger import Logger


class ProgLogger(Logger):
    """Logger that prints training progress to stdout

    Attributes:
       metrics: list of metrics
       interval: number of batches between prints
    """

    def __init__(
        self, metrics: Optional[Iterable[str]] = None, interval: int = 10
    ) -> None:
        """Initializes the progress logger.

        Args:
            metrics: Optional subset of metric names to display.
            interval: Number of batches between progress prints.
        """
        super().__init__()

        self.metrics: Optional[Set[str]] = None if metrics is None else set(metrics)

        if interval <= 0:
            raise ValueError("ProgLogger requires interval > 0")
        self.interval = interval
        self.epochs = 0
        self.batches = 0
        self.samples = 0
        self.cur_sample = 0
        self.t0 = 0

    def on_train_begin(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Validates training metadata and stores total epoch count.

        Args:
            logs: optional training logs.
            kwargs: expects ``epochs``.
        """
        super().on_train_begin(logs, **kwargs)
        if "epochs" not in kwargs:
            raise ValueError("ProgLogger requires 'epochs' in kwargs")
        self.epochs = kwargs["epochs"]

    def on_epoch_begin(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Resets counters and starts epoch timing.

        Args:
            epoch: Zero-based epoch index.
            logs: Optional logs dictionary.
            kwargs: Additional callback arguments such as ``samples`` and ``batches``.
        """
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

    def on_batch_begin(
        self, batch: int, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Tracks the current batch index.

        Args:
            batch: Zero-based batch index within the epoch.
            logs: Optional logs dictionary.
            kwargs: Additional callback arguments.
        """
        self.cur_batch = batch

    def on_batch_end(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Prints periodic batch progress information.

        Args:
            logs: Optional metric dictionary for the current batch.
            kwargs: Additional callback arguments such as ``batch_size``.
        """
        if self.rank != 0:
            return

        logs = logs or {}
        logs = {k.replace("/", "_"): v for k, v in logs.items()}
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

            info += " global_step: %d" % (self.cur_step)

            for k, v in logs.items():
                if self.metrics is None or k in self.metrics:
                    info += " %s: %.6f" % (k, v)

            logging.info(info)

    def on_epoch_end(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Prints final epoch metrics.

        Args:
            logs: Optional metric dictionary for the epoch.
            kwargs: Additional callback arguments.
        """
        if self.rank != 0:
            return

        logs = logs or {}
        logs = {k.replace("/", "_"): v for k, v in logs.items()}
        info = "epoch: %d/%d " % (self.cur_epoch + 1, self.epochs)
        if self.cur_step > 0:
            info += " global_step: %d" % (self.cur_step)

        for k, v in logs.items():
            if self.metrics is None or k in self.metrics:
                info += " %s: %.6f" % (k, v)

        logging.info(info)

    def on_val_end(self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Prints validation metrics using the epoch-end formatter.

        Args:
            logs: Optional validation metrics.
            kwargs: Additional callback arguments.
        """
        self.on_epoch_end(logs, **kwargs)

    def on_model_update(
        self, step: int, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Prints model-update metrics at the provided global step.

        Args:
            step: Global optimizer step.
            logs: Optional update-level metrics.
            kwargs: Additional callback arguments.
        """
        super().on_model_update(step, logs, **kwargs)
        if self.rank != 0 or not logs:
            return

        logs = {k.replace("/", "_"): v for k, v in logs.items()}
        info = "global_step: %d" % (step)

        for k, v in logs.items():
            if self.metrics is None or k in self.metrics:
                info += " %s: %.6f" % (k, v)

        logging.info(info)

    def estimate_epoch_time(self) -> Tuple[str, Optional[str]]:
        """Estimates elapsed time and ETA for the current epoch."""
        t1 = time.time()
        et = t1 - self.t0
        if self.batches > 0 and self.cur_batch > 0:
            total_t = et / self.cur_batch * self.batches
        elif self.samples > 0 and self.cur_sample > 0:
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
    def sec2str(t: float) -> str:
        """Formats seconds into a compact human-readable duration string.

        Args:
            t: Duration in seconds.
        """
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
