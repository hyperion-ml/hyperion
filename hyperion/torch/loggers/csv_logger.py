"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import csv
from collections import OrderedDict as ODict
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from ...utils import PathLike
from .logger import Logger


class CSVLogger(Logger):
    """Logger that prints metrics to csv file
       at the end of each epoch

    Attributes:
       file_path: filename of csv file.
       sep: column separator for csv file
       append: False, overwrite existing file, True, appends.
    """

    def __init__(
        self, file_path: PathLike, sep: str = ",", append: bool = False
    ) -> None:
        """Initializes the CSV logger.

        Args:
            file_path: Path to the output CSV file.
            sep: Delimiter used between columns.
            append: If True, append to an existing file when possible.
        """
        super().__init__()
        self.file_path = Path(file_path)
        self.sep = sep
        self.append = append
        self.csv_writer: Optional[csv.DictWriter] = None
        self.csv_file: Optional[TextIO] = None
        self.log_keys: Optional[List[str]] = None
        self.fieldnames: Optional[List[str]] = None
        self._header_written = False

    @staticmethod
    def _has_step_columns(fieldnames: Optional[List[str]]) -> bool:
        """Checks whether a CSV header already includes batch/step columns."""
        return (
            fieldnames is not None
            and "batch" in fieldnames
            and "global_step" in fieldnames
        )

    def _rewrite_legacy_file(self, fieldnames: List[str]) -> None:
        """Rewrites an existing legacy CSV file with the modern schema."""
        if not self.file_path.exists():
            return

        with self.file_path.open("r", newline="") as f:
            reader = csv.DictReader(f, delimiter=self.sep)
            rows = list(reader)

        modern_fieldnames = ["epoch", "batch", "global_step"] + [
            k for k in fieldnames if k not in {"epoch", "batch", "global_step"}
        ]
        with self.file_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=modern_fieldnames, delimiter=self.sep)
            writer.writeheader()
            for row in rows:
                row = dict(row)
                row.setdefault("batch", "NA")
                row.setdefault("global_step", "NA")
                writer.writerow(row)
            f.flush()

    def on_train_begin(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Opens the CSV file before training starts.

        Args:
            logs: Optional training logs.
            kwargs: Additional callback arguments.
        """
        super().on_train_begin(logs, **kwargs)
        if self.rank != 0:
            return

        file_dir = self.file_path.parent
        file_dir.mkdir(parents=True, exist_ok=True)

        if self.append:
            if self.file_path.exists():
                with self.file_path.open("r", newline="") as f:
                    reader = csv.reader(f, delimiter=self.sep)
                    existing_fieldnames = next(reader, None)
                if existing_fieldnames is not None and not self._has_step_columns(
                    existing_fieldnames
                ):
                    self._rewrite_legacy_file(existing_fieldnames)
            self._header_written = (
                self.file_path.exists() and self.file_path.stat().st_size > 0
            )
            self.csv_file = self.file_path.open("a", newline="")
        else:
            self._header_written = False
            self.csv_file = self.file_path.open("w", newline="")

    def on_epoch_end(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Writes one row of epoch metrics to the CSV file.

        Args:
            logs: Metric dictionary for the current epoch.
            kwargs: Additional callback arguments.
        """
        if self.rank != 0:
            return
        logs = logs or {}

        logs = {k.replace("/", "_"): v for k, v in logs.items()}
        if self.log_keys is None:
            self.log_keys = list(logs.keys())

        if not self.csv_writer:

            class MyDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ["epoch", "batch", "global_step"] + self.log_keys
            self.fieldnames = fieldnames
            self.csv_writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=MyDialect
            )
            if not self._header_written:
                self.csv_writer.writeheader()
                self._header_written = True

        batch = kwargs.get("batch", self.cur_batch)
        step = kwargs.get("step", kwargs.get("global_step", self.cur_step))
        row = ODict(
            [("epoch", self.cur_epoch + 1), ("batch", batch), ("global_step", step)]
        )
        row.update((k, logs[k] if k in logs else "NA") for k in self.log_keys)
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def on_val_end(self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """At the end of validation

        Args:
           logs: dictionary of logs
        """
        self.on_epoch_end(logs, **kwargs)

    def on_train_end(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Closes the CSV file when training ends.

        Args:
            logs: Optional final training logs.
            kwargs: Additional callback arguments.
        """
        if self.rank != 0:
            return

        if self.csv_file is not None:
            self.csv_file.close()
        self.csv_file = None
        self.csv_writer = None
        self._header_written = False
