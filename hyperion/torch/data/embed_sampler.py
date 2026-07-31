"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from typing import Any, Dict, Optional

import numpy as np
import torch
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils import InfoTable
from ...utils.misc import filter_func_args
from .hyper_sampler import HyperSampler


class EmbedSampler(HyperSampler):
    """
    Fixed-size sampler for embedding metadata stored in an ``InfoTable``.

    Embeddings have a fixed dimension, so batches are constrained only by the
    number of embeddings they contain. Distributed workers receive rank-strided
    rows from a shared epoch index. When ``drop_last`` is ``False``, the index is
    padded by repeating leading rows so that every rank receives the same number
    of full batches.

    Attributes:
        embed_set (InfoTable): Metadata for the embeddings to sample.
        batch_size (int): Number of embeddings in each batch per rank.
        avg_batch_size (int): Average number of embeddings per batch.
        drop_last (bool): Whether to drop embeddings that cannot fill a complete
            distributed batch instead of repeating leading embeddings as padding.
        _len (int): Number of batches yielded by this rank per epoch.
        _epoch_indices (Optional[np.ndarray]): Padded, optionally shuffled row
            indices for the current epoch.
        start (int): Next rank-strided row position in ``_epoch_indices``.
    """

    def __init__(
        self,
        embed_set: InfoTable,
        batch_size: int = 1,
        max_batches_per_epoch: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 1234,
    ) -> None:
        """
        Initialize the embedding sampler.

        Args:
            embed_set: Metadata table containing an ``id`` column.
            batch_size: Number of embeddings in each batch per rank.
            max_batches_per_epoch: Optional maximum number of batches per epoch.
            shuffle: Whether to shuffle embedding rows at the start of each epoch.
            drop_last: Whether to drop embeddings that cannot fill a complete
                distributed batch. When ``False``, leading embeddings repeat to
                pad the final distributed batch.
            seed: Base random seed.
        """
        super().__init__(
            max_batches_per_epoch=max_batches_per_epoch, shuffle=shuffle, seed=seed
        )
        if len(embed_set) == 0:
            raise ValueError("embed_set must contain at least one row.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        self.embed_set = embed_set
        self.batch_size = batch_size
        self.avg_batch_size = batch_size
        self.drop_last = drop_last

        num_embeds = len(self.embed_set)
        num_batches = num_embeds / (self.world_size * batch_size)
        if drop_last:
            self._len = int(num_batches)
        else:
            self._len = int(math.ceil(num_batches))

        if self.max_batches_per_epoch is not None:
            self._len = min(self._len, self.max_batches_per_epoch)

        if self._len == 0:
            raise ValueError(
                "EmbedSampler would yield zero batches per rank. Increase the "
                "number of embeddings, reduce batch_size, or reduce world_size."
            )

        self._epoch_indices: Optional[np.ndarray] = None
        self.start = 0

    def __len__(self) -> int:
        """
        Return the number of batches per epoch.

        Returns:
            Number of batches yielded by this rank.
        """
        return self._len

    def _create_epoch_indices(self) -> None:
        """Create the shared, padded row-index sequence for the current epoch."""
        num_embeds = len(self.embed_set)
        if self.shuffle:
            indices = torch.randperm(num_embeds, generator=self.rng).numpy()
        else:
            indices = np.arange(num_embeds)

        total_size = self._len * self.world_size * self.batch_size
        if total_size <= num_embeds:
            self._epoch_indices = indices[:total_size]
            return

        padding_size = total_size - num_embeds
        padding = np.resize(indices, padding_size)
        self._epoch_indices = np.concatenate((indices, padding))

    def __iter__(self) -> "EmbedSampler":
        """
        Initialize iteration and replay a requested resume offset.

        Returns:
            This sampler instance.
        """
        resume_batch = self.init_batch
        self.init_batch = 0
        super().__iter__()
        self._create_epoch_indices()

        self.start = self.rank
        if resume_batch != 0:
            logging.info("Replaying %d batches to resume EmbedSampler.", resume_batch)
            for _ in range(min(resume_batch, self._len)):
                try:
                    next(self)
                except StopIteration:
                    break
        return self

    def __next__(self) -> np.ndarray:
        """
        Generate the next batch of embedding IDs.

        Returns:
            Embedding IDs selected for this rank's next batch.

        Raises:
            StopIteration: When all batches for the epoch have been yielded.
        """
        if self.batch >= self._len:
            raise StopIteration
        if self._epoch_indices is None:
            raise RuntimeError("EmbedSampler must be iterated before calling next().")

        stop = self.start + self.world_size * self.batch_size
        idx = self._epoch_indices[self.start : stop : self.world_size]

        self.start += self.world_size * self.batch_size

        embed_ids = self.embed_set.iloc[idx].id.values

        if self.batch == 0:
            logging.info("batch 0 embed_ids=%s", str(embed_ids[:10]))

        self.batch += 1
        return embed_ids

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """
        Filter keyword arguments relevant to the sampler constructor.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Keyword arguments accepted by :meth:`__init__`.
        """
        return filter_func_args(EmbedSampler.__init__, kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """
        Add ``EmbedSampler`` arguments to a command-line parser.

        Args:
            parser: Parser to extend.
            prefix: Optional nested parser prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help="batch size per GPU",
        )

        parser.add_argument(
            "--drop-last",
            action=ActionYesNo,
            help="drops the last batch of the epoch",
        )

        parser.add_argument(
            "--max-batches-per-epoch",
            type=int,
            default=None,
            help=("Max. batches per epoch"),
        )

        parser.add_argument(
            "--shuffle",
            action=ActionYesNo,
            help="shuffles the embeddings at the beginning of the epoch",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=1234,
            help=("seed for sampler random number generator"),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
