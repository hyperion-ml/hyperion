"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
# import os
import math

import numpy as np
from jsonargparse import ActionParser, ArgumentParser

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class ClassWeightedSeqSampler(Sampler):
    """Samples utterances following:
        1. It samples a class with a given probability.
        2. It samples an random utterance from the class.

    Attributes:
      dataset: dataset containing audio or feature sequences.
      batch_size: batch size per gpu for the largest chunk-size.
      num_egs_per_utt_epoch: number of samples per utterance and epoch.
      num_egs_per_class: number of samples per class in each batch.
      num_egs_per_utt: number of samples per utterance in each batch.
      var_batch_size: whether to use variable batch size when using
        variable utterance length.
      num_hard_prototypes: number of hard prototype classes per random class
        in a batch.
      num_egs_per_hard_prototype: number of utterances per each hard
        prototype in a batch.
      iters_per_epoch: deprecated, if not None, will overwrite "num_egs_per_utt_epoch".
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        num_egs_per_utt_epoch="auto",
        num_egs_per_class=1,
        num_egs_per_utt=1,
        var_batch_size=False,
        num_hard_prototypes=0,
        affinity_matrix=None,
        iters_per_epoch=None,
    ):

        super().__init__(None)

        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1

        if iters_per_epoch is not None:
            num_egs_per_utt_epoch = iters_per_epoch

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_egs_per_class = num_egs_per_class
        self.num_egs_per_utt = num_egs_per_utt
        self.var_batch_size = var_batch_size
        self.num_hard_prototypes = num_hard_prototypes
        self.batch = 0

        self.rank = rank
        self.world_size = world_size
        if rank > 0:
            # this will make sure that each process produces different data
            # when using ddp
            dummy = torch.rand(1000 * rank)
            del dummy

        self.has_short_seqs = self.dataset.short_seq_exist
        self.set_num_egs_per_utt_epoch(num_egs_per_utt_epoch)
        self._compute_avg_batch_size()
        self._compute_len(world_size)
        self._compute_num_classes_per_batch()
        self.set_hard_prototypes(affinity_matrix)
        logging.info(
            "batches/epoch=%d classes/batch=%d avg-batch-size/gpu=%d samples/(utt*epoch)=%d",
            self._len,
            self._num_classes_per_batch,
            self.avg_batch_size,
            self.num_egs_per_utt_epoch,
        )

    def _compute_avg_batch_size(self):
        if not self.var_batch_size:
            self.avg_batch_size = self.batch_size
            return

        dataset = self.dataset
        avg_chunk_length = int(
            (dataset.max_chunk_length + dataset.min_chunk_length) / 2
        )
        batch_mult = dataset.max_chunk_length / avg_chunk_length
        self.avg_batch_size = int(self.batch_size * batch_mult)

    def set_num_egs_per_utt_epoch(self, num_egs_per_utt_epoch):
        if num_egs_per_utt_epoch == "auto":
            self._compute_num_egs_per_utt_epoch_auto()
        else:
            self.num_egs_per_utt_epoch = num_egs_per_utt_epoch

    def _compute_num_egs_per_utt_epoch_auto(self):
        dataset = self.dataset
        avg_seq_length = np.mean(dataset.seq_lengths)
        avg_chunk_length = int(
            (dataset.max_chunk_length + dataset.min_chunk_length) / 2
        )
        self.num_egs_per_utt_epoch = math.ceil(avg_seq_length / avg_chunk_length)
        logging.debug("num iters per epoch: %d", self.num_egs_per_utt_epoch)

    def _compute_len(self, world_size):
        self._len = int(
            math.ceil(
                self.num_egs_per_utt_epoch
                * self.dataset.num_seqs
                / self.avg_batch_size
                / world_size
            )
        )

    def _compute_num_classes_per_batch(self):
        self._num_classes_per_batch = int(
            math.ceil(
                self.avg_batch_size / self.num_egs_per_class / self.num_egs_per_utt
            )
        )

    def _get_class_weights(self, chunk_length):
        if not self.has_short_seqs:
            return self.dataset.class_weights

        # get classes with utt shorter than chunk length and put weight to 0
        zero_idx = self.dataset.class2max_length < chunk_length
        if not np.any(zero_idx):
            return self.dataset.class_weights

        class_weights = self.dataset.class_weights.clone()
        class_weights[zero_idx] = 0
        # renormalize weights
        class_weights /= class_weights.sum()
        return class_weights

    def _get_seq_weights(self, chunk_length):
        pass

    def __len__(self):
        return self._len

    def __iter__(self):
        self.batch = 0
        return self

    @property
    def hard_prototype_mining(self):
        return self.num_hard_prototypes > 0

    def set_hard_prototypes(self, affinity_matrix):
        if affinity_matrix is None:
            self.hard_prototypes = None
            return

        # affinity_matrix[np.diag(affinity_matrix.shape[0])] = -1.0
        # hard prototypes for a class are itself and k-1 closest to it.
        self.hard_prototypes = torch.topk(
            affinity_matrix, self.num_hard_prototypes, dim=-1
        ).indices

    def get_hard_prototypes(self, class_idx):
        return self.hard_prototypes[class_idx].flatten()

    def _get_utt_idx_basic(self, batch_mult=1):
        dataset = self.dataset
        num_classes_per_batch = batch_mult * self._num_classes_per_batch
        if self.hard_prototype_mining:
            num_classes_per_batch = int(
                math.ceil(num_classes_per_batch / self.num_hard_prototypes)
            )

        if dataset.class_weights is None:
            class_idx = torch.randint(
                low=0, high=dataset.num_classes, size=(num_classes_per_batch,)
            )
        else:
            class_idx = torch.multinomial(
                dataset.class_weights,
                num_samples=num_classes_per_batch,
                replacement=True,
            )

        if self.hard_prototype_mining:
            class_idx = self.get_hard_prototypes(class_idx)

        if self.num_egs_per_class > 1:
            class_idx = class_idx.repeat(self.num_egs_per_class)

        utt_idx = torch.as_tensor(
            [
                dataset.class2utt_idx[c][
                    torch.randint(low=0, high=int(dataset.class2num_utt[c]), size=(1,))
                ]
                for c in class_idx.tolist()
            ]
        )

        return utt_idx

    def _get_utt_idx_seq_st_max_length(self, chunk_length, batch_mult=1):
        dataset = self.dataset

        num_classes_per_batch = batch_mult * self._num_classes_per_batch
        if self.hard_prototype_mining:
            num_classes_per_batch = int(
                math.ceil(num_classes_per_batch / self.num_hard_prototypes)
            )

        # first we sample the batch classes
        class_weights = dataset.class_weights.clone()
        # get classes with utt shorter than chunk lenght
        class_weights[dataset.class2max_length < chunk_length] = 0

        # renormalize weights and sample
        class_weights /= class_weights.sum()
        # logging.info(str(class_weights))
        class_idx = torch.multinomial(
            class_weights, num_samples=num_classes_per_batch, replacement=True
        )

        if self.hard_prototype_mining:
            class_idx = self.get_hard_prototypes(class_idx)

        utt_idx = torch.zeros(
            (len(class_idx) * self.num_egs_per_class,), dtype=torch.long
        )
        k = 0
        for c in class_idx.tolist():
            # for each class we sample an utt between the utt longer than chunk length

            # get utts for class c
            utt_idx_c = torch.as_tensor(dataset.class2utt_idx[c])

            # find utts longer than chunk length for class c
            seq_lengths_c = torch.as_tensor(dataset.seq_lengths[utt_idx_c])
            utt_weights = torch.ones((int(dataset.class2num_utt[c]),))
            utt_weights[seq_lengths_c < chunk_length] = 0
            utt_weights /= utt_weights.sum()

            # sample utt idx
            try:
                utt_idx[k : k + self.num_egs_per_class] = utt_idx_c[
                    torch.multinomial(
                        utt_weights,
                        num_samples=self.num_egs_per_class,
                        replacement=True,
                    )
                ]
            except:
                logging.info("{} {}".format(seq_lengths_c, utt_weights))

            k += self.num_egs_per_class

        return utt_idx

    def __next__(self):

        if self.batch == self._len:
            raise StopIteration

        chunk_length = self.dataset.get_random_chunk_length()

        if self.var_batch_size:
            batch_mult = int(self.dataset.max_chunk_length // chunk_length)
        else:
            batch_mult = 1

        if self.dataset.short_seq_exist:
            utt_idx = self._get_utt_idx_seq_st_max_length(chunk_length, batch_mult)
        else:
            utt_idx = self._get_utt_idx_basic(batch_mult)

        if self.num_egs_per_utt > 1:
            utt_idx = utt_idx.repeat(self.num_egs_per_utt)

        utt_idx = utt_idx.tolist()[: self.batch_size * batch_mult]
        if self.batch == 0:
            logging.info("batch 0 uttidx=%s", str(utt_idx[:10]))

        self.batch += 1
        index = [(i, chunk_length) for i in utt_idx]
        return index

    @staticmethod
    def filter_args(**kwargs):

        if "no_shuffle_seqs" in kwargs:
            kwargs["shuffle_seqs"] = not kwargs["no_shuffle_seqs"]

        valid_args = (
            "batch_size",
            "var_batch_size",
            "iters_per_epoch",
            "num_egs_per_utt_epoch",
            "num_egs_per_class",
            "num_egs_per_utt",
            "num_hard_prototypes",
        )
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--batch-size", default=128, type=int, help=("batch size per gpu")
        )

        parser.add_argument(
            "--var-batch-size",
            default=False,
            action="store_true",
            help=(
                "use variable batch-size, "
                "then batch-size is the minimum batch size, "
                "which is used when the batch chunk length is "
                "equal to max-chunk-length"
            ),
        )

        parser.add_argument(
            "--iters-per-epoch",
            default=None,
            type=lambda x: x if (x == "auto" or x is None) else float(x),
            help=("number of times we sample an utterance in each epoch"),
        )

        parser.add_argument(
            "--num-egs-per-utt-epoch",
            default="auto",
            type=lambda x: x if x == "auto" else float(x),
            help=("number of times we sample an utterance in each epoch"),
        )

        parser.add_argument(
            "--num-egs-per-class",
            type=int,
            default=1,
            help=("number of samples per class in batch"),
        )
        parser.add_argument(
            "--num-egs-per-utt",
            type=int,
            default=1,
            help=("number of samples per utterance in batch"),
        )
        parser.add_argument(
            "--num-hard-prototypes",
            type=int,
            default=0,
            help=("number of hard prototype classes per batch"),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args
