"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from enum import Enum
from typing import Optional

import numpy as np
import torch
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils import SegmentSet
from ...utils.misc import filter_func_args
from .hyp_sampler import HypSampler


class LengthSamplingMethod(str, Enum):
    """
    Enum for length sampling methods used in segment-based sampling.
    """

    UNIFORM = "uniform"
    MAXIMUM = "maximum"

    def __str__(self):
        return self.value

    @staticmethod
    def choices():
        """
        Returns a list of valid choices for length sampling methods.
        """
        return [method.value for method in LengthSamplingMethod]


class SegSampler(HypSampler):
    """
    Segment-based sampler for PyTorch DataLoaders with support for:

    - Fixed or variable batch sizes.
    - Maximum total length per batch (e.g., max total duration).
    - Optional shuffling.
    - Distributed data loading (multi-GPU training).
    - Support for segment-chunk indexing (if "chunk_start" present in `segments`).

    Attributes:
        segments (SegmentSet): The set of segments to sample from.
        min_batch_size (int): Minimum number of samples per batch.
        max_batch_size (Optional[int]): Maximum number of samples per batch.
        max_batch_length (Optional[int]): Max accumulated value for `length_name` per batch.
        length_name (str): Name of the column in `segments` that defines length (e.g., "duration").
        max_batches_per_epoch (Optional[int]): Optional limit on the number of batches per epoch.
        shuffle (bool): Whether to shuffle segment order each epoch.
        drop_last (bool): Whether to drop the last partial batch.
        skip_long_segs (bool): If True, skips segments longer than `max_batch_length`.
        sort_by_length (bool): Whether to sort batch items by length descending.
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        segments: SegmentSet,
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        max_batch_length: Optional[int] = None,
        length_name: str = "duration",
        max_batches_per_epoch: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        skip_long_segs: bool = False,
        sort_by_length: bool = True,
        seed: int = 1234,
    ):
        super().__init__(
            max_batches_per_epoch=max_batches_per_epoch, shuffle=shuffle, seed=seed
        )
        self.segments = segments
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_batch_length = max_batch_length
        self.var_batch_size = max_batch_length is not None
        self.length_name = length_name
        self.drop_last = drop_last
        self.sort_by_length = sort_by_length
        self.skip_long_segs = skip_long_segs
        if max_batch_size is not None:
            assert min_batch_size <= max_batch_size

        num_segs = len(segments)
        logging.info(f"Initializing SegSampler with {num_segs} segments.")

        if self.var_batch_size:
            logging.info("Variable batch size mode enabled.")
            if self.skip_long_segs:
                logging.info(
                    f"Filtering segments longer than max_batch_length={max_batch_length}."
                )
                lengths = segments.loc[
                    segments[self.length_name] <= max_batch_length, self.length_name
                ]
                num_segs = len(lengths)
                if num_segs == 0:
                    raise ValueError(
                        f"No segments fit within max_batch_length={max_batch_length}."
                    )
                avg_batch_size = max_batch_length / np.mean(lengths)
                logging.info(
                    f"Average batch size estimated from filtered segments: {avg_batch_size:.2f}"
                )
            else:
                lengths = segments[self.length_name].values
                avg_batch_size = max_batch_length / np.mean(
                    self.segments[self.length_name]
                )
                logging.info(
                    f"Average batch size estimated from all segments: {avg_batch_size:.2f}"
                )
            if max_batch_size is not None and avg_batch_size > max_batch_size:
                logging.warning(
                    f"Average batch size {avg_batch_size:.2f} exceeds max_batch_size={max_batch_size}. "
                    "Adjusting to max_batch_size."
                )
                avg_batch_size = max_batch_size
        else:
            logging.info("Fixed batch size mode enabled.")
            avg_batch_size = min_batch_size
            logging.info(f"Using fixed min_batch_size = {min_batch_size}")
            lengths = segments[self.length_name].values

        self.avg_batch_size = avg_batch_size

        if drop_last:
            self._len = int(num_segs / (avg_batch_size * self.world_size))
        else:
            self._len = int(math.ceil((num_segs // self.world_size) / avg_batch_size))

        logging.info(
            f"Batches per epoch (before max_batches_per_epoch limit): {self._len}"
        )

        if self.max_batches_per_epoch is not None:
            self._len = min(self._len, self.max_batches_per_epoch)
            logging.info(
                f"Limiting batches per epoch to max_batches_per_epoch = {self.max_batches_per_epoch}"
            )

        logging.info(
            f"Sampler final configuration: "
            f"min_batch_size={self.min_batch_size}, max_batch_size={self.max_batch_size}, "
            f"min_length={lengths.min():.2f}, max_length={lengths.max():.2f}, "
            f"avg_batch_size={self.avg_batch_size:.2f}, shuffle={shuffle}, "
            f"drop_last={drop_last}, sort_by_length={sort_by_length}, seed={seed}"
        )

        self._permutation = None
        self.length_col_idx = self.segments.columns.get_loc(self.length_name)

    def __len__(self):
        """Returns the number of batches per epoch."""
        return self._len

    def _shuffle_segs(self):
        """Shuffles the segment indices using internal RNG."""
        self._permutation = torch.randperm(
            len(self.segments), generator=self.rng
        ).numpy()

    def __iter__(self):
        """Initializes the iterator for batching."""
        super().__iter__()
        if self.shuffle:
            self._shuffle_segs()

        self.start = self.rank
        return self

    def __next__(self):
        """Generates the next batch of segment IDs."""
        # Stop if we've reached the predefined number of batches
        if self.batch == self._len:
            raise StopIteration

        if self.var_batch_size:
            idxs = []  # List to hold selected segment indices
            max_length = 0  # Longest segment seen in this batch
            batch_size = 0  # Number of segments in the batch
            attempts = 0  # How many segments we've tried to include
            max_attempts = len(self.segments)  # Limit to prevent infinite loop

            while attempts < max_attempts and (
                batch_size < self.max_batch_size if self.max_batch_size else True
            ):
                # Get the candidate index (shuffled or sequential)
                if self.shuffle:
                    idx = self._permutation[self.start]
                else:
                    idx = self.start

                seg_length = self.segments.iloc[idx, self.length_col_idx]
                max_length = max(max_length, seg_length)

                if max_length * (batch_size + 1) > self.max_batch_length:
                    self.start = (self.start + self.world_size) % len(self.segments)
                    if seg_length > self.max_batch_length and self.skip_long_segs:
                        # Skip this segment and try the next one
                        attempts += 1
                        continue
                    elif batch_size == 0:
                        # This segment alone doesn't fit — raise
                        raise ValueError(
                            f"No segments fit into max_batch_length={self.max_batch_length}. "
                            f"First segment length: {seg_length:.2f}."
                        )
                    break  # stop adding more segments

                idxs.append(idx)
                # Move to next candidate for this process (handles multi-GPU)
                self.start = (self.start + self.world_size) % len(self.segments)
                batch_size += 1
                attempts += 1

                # Respect an upper limit on batch size if specified
                if (
                    self.max_batch_size is not None
                    and batch_size >= self.max_batch_size
                ):
                    break

            if self.drop_last and (len(idxs) < self.min_batch_size):
                # If we’re near the end and can’t meet the minimum size, skip this batch
                raise StopIteration

            # If we tried all possible segments and couldn't add even one, raise an error
            if len(idxs) < 1:
                raise ValueError(
                    f"No segments fit into max_batch_length={self.max_batch_length}. "
                    f"Longest seen segment: {max_length:.2f}."
                )

        else:
            # Fixed-size batching: take a block of consecutive samples
            stop = min(
                self.start + self.world_size * self.min_batch_size, len(self.segments)
            )
            if self.shuffle:
                idxs = self._permutation[self.start : stop : self.world_size]
            else:
                idxs = slice(self.start, stop, self.world_size)

            # Advance the starting index for the next batch
            self.start += self.world_size * self.min_batch_size

        # Extract segment IDs from selected indices
        ids = self.segments.iloc[idxs].id.values

        # Optionally sort the batch by segment length (descending)
        if self.sort_by_length:
            lengths = self.segments.loc[ids, self.length_name].values
            sort_idx = np.argsort(lengths)[::-1]
            ids = ids[sort_idx]

        # If this is chunked data, return (seg_id, chunk_start, length) triplets
        if "chunk_start" in self.segments:
            chunks = self.segments.loc[ids]
            seg_ids = [
                (id, s, d)
                for id, s, d in zip(
                    chunks.seg_id, chunks.chunk_start, chunks[self.length_name]
                )
            ]
            # print(
            #     "train_segids", seg_ids, flush=True
            # )  # Debug print to inspect segment IDs
        else:
            seg_ids = ids  # Just return segment IDs
            # print("val_segids", seg_ids, flush=True)

        # Log a sample batch for inspection
        if self.batch == 0:
            logging.info("batch 0 seg_ids=%s", str(seg_ids[:10]))

        self.batch += 1
        return seg_ids

    # def __next__(self):
    #     """Generates the next batch of segment IDs."""
    #     if self.batch == self._len:
    #         raise StopIteration

    #     if self.var_batch_size:
    #         # Dynamically batch up to max_batch_length
    #         # column_idx = self.segments.columns.get_loc(self.length_name)
    #         idxs = []
    #         max_length = 0
    #         batch_size = 0
    #         while True:
    #             if self.shuffle:
    #                 idx = self._permutation[self.start]
    #             else:
    #                 idx = self.start

    #             max_length = max(max_length, self.segments.iloc[idx, self.length_col_idx])
    #             if max_length * (batch_size + 1) > self.max_batch_length:
    #                 break

    #             idxs.append(idx)
    #             self.start = (self.start + self.world_size) % len(self.segments)
    #             batch_size += 1
    #             if (
    #                 self.max_batch_size is not None
    #                 and batch_size >= self.max_batch_size
    #             ):
    #                 break

    #         if len(idxs) < 1:
    #             raise ValueError(f"No segments fit within max_batch_length={self.max_batch_length}. "
    #                  f"Longest segment seen: {max_length:.2f}")
    #     else:
    #         # Fixed-size batching
    #         stop = min(
    #             self.start + self.world_size * self.min_batch_size, len(self.segments)
    #         )
    #         if self.shuffle:
    #             idxs = self._permutation[self.start : stop : self.world_size]
    #         else:
    #             idxs = slice(self.start, stop, self.world_size)

    #         self.start += self.world_size * self.min_batch_size

    #     # Get segment IDs
    #     # ids = self.segments.iloc[idxs].id.values
    #     ids = self.segments.iloc[idxs, self.id_col_idx].values
    #     if self.sort_by_length:
    #         lengths = self.segments.loc[ids, self.length_name].values
    #         sort_idx = np.argsort(lengths)[::-1]
    #         ids = ids[sort_idx]

    #     # If chunked segments (chunk_start column exists), yield (seg_id, start, duration)
    #     if "chunk_start" in self.segments:
    #         chunks = self.segments.loc[ids]
    #         seg_ids = [
    #             (id, s, d)
    #             for id, s, d in zip(
    #                 chunks.seg_id, chunks.chunk_start, chunks[self.length_name]
    #             )
    #         ]
    #     else:
    #         seg_ids = ids

    #     if self.batch == 0:
    #         logging.info("batch 0 seg_ids=%s", str(seg_ids[:10]))

    #     self.batch += 1
    #     return seg_ids

    @staticmethod
    def filter_args(**kwargs):
        """
        Filters keyword arguments relevant to the SegSampler constructor.

        Returns:
            dict: Filtered kwargs.
        """
        return filter_func_args(SegSampler.__init__, kwargs, skip={"segments"})

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None):
        """
        Adds SegSampler-specific arguments to a parser.

        Args:
            parser (ArgumentParser): The parser to extend.
            prefix (Optional[str]): If provided, adds a subparser group under this prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--min-batch-size",
            type=int,
            default=1,
            help="Minimum number of samples (segments) per batch on each GPU. "
            "Required for both fixed and variable batch sizing.",
        )

        parser.add_argument(
            "--max-batch-size",
            type=int,
            default=None,
            help="Optional cap on the number of segments per batch. "
            "Useful when using variable-length batching. If not set, batching is only constrained by max-batch-duration.",
        )

        parser.add_argument(
            "--max-batch-length",
            type=float,
            default=None,
            help="Maximum total duration (sum of segment lengths) allowed per batch, in seconds. "
            "If set, enables variable-length batching where batch size adapts based on segment duration.",
        )

        parser.add_argument(
            "--drop-last",
            action=ActionYesNo,
            help="If enabled, drops the last batch of the epoch if it contains fewer than the minimum number of samples. "
            "This is useful when consistent batch sizes are desired across all iterations.",
        )
        parser.add_argument(
            "--skip-long-segs",
            action=ActionYesNo,
            help="If enabled, skips segments longer than max-batch-length. "
            "This is useful when you want to avoid batches with very long segments that could skew training.",
        )
        parser.add_argument(
            "--max-batches-per-epoch",
            type=int,
            default=None,
            help="Optional cap on the total number of batches per epoch. "
            "Useful for debugging or limiting training duration.",
        )

        parser.add_argument(
            "--shuffle",
            action=ActionYesNo,
            help="Shuffle the dataset at the start of each epoch. "
            "Recommended for training to ensure randomness.",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=1234,
            help="Random seed used for shuffling. Ensures reproducibility across runs.",
        )

        parser.add_argument(
            "--length-name",
            default="duration",
            help="Name of the column in the segment table that represents segment length (e.g., 'duration'). "
            "This is used to compute variable-length batches.",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
