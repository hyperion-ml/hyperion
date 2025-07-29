"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import csv
import os
from collections import OrderedDict as ODict

import numpy as np

from .logger import Logger


class CSVLogger(Logger):
    """Logger that prints metrics to csv file
       at the end of each epoch

    Attributes:
       file_path: filenane of csv file.
       sep: column separator for csv file
       append: False, overwrite existing file, True, appends.
    """

    def __init__(self, file_path, sep=",", append=False):
        super().__init__()
        self.file_path = file_path
        self.sep = sep
        self.append = append
        self.csv_writer = None
        self.csv_file = None
        self.append_header = True
        self.log_keys = None

    def on_train_begin(self, logs=None, **kwargs):
        super().on_train_begin(logs, **kwargs)
        if self.rank != 0:
            return

        file_dir = os.path.dirname(self.file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        if self.append:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r") as f:
                    self.append_header = len(f.readline()) == 0

        if self.append_header:
            self.csv_file = open(self.file_path, "w")
        else:
            self.csv_file = open(self.file_path, "a")

    def on_epoch_end(self, logs=None, **kwargs):
        if self.rank != 0:
            return
        logs = logs or {}

        logs = {k.replace("/", "_"): v for k, v in logs.items()}
        if self.log_keys is None:
            self.log_keys = list(logs.keys())

        if not self.csv_writer:

            class MyDialect(csv.excel):
                delimiter = self.sep

            if self.cur_step == 0:
                # legacy support for old versions
                fieldnames = ["epoch"] + self.log_keys
            else:
                fieldnames = ["epoch", "batch", "global_step"] + self.log_keys
            self.csv_writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=MyDialect
            )
            if self.append_header:
                self.csv_writer.writeheader()

        if self.cur_step == 0:
            # legacy support for old versions
            row = ODict([("epoch", self.cur_epoch + 1)])
        else:
            row = ODict(
                [
                    ("epoch", self.cur_epoch + 1),
                    ("batch", self.cur_batch),
                    ("global_step", self.cur_step),
                ]
            )
        row.update((k, logs[k] if k in logs else "NA") for k in self.log_keys)
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def on_val_end(self, logs=None, **kwargs):
        """At the end of validation

        Args:
           logs: dictionary of logs
        """
        self.on_epoch_end(logs, **kwargs)

    def on_train_end(self, logs=None, **kwargs):
        if self.rank != 0:
            return

        self.csv_file.close()
        self.writer = None
