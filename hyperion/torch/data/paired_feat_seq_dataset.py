"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import numpy as np

import torch

from ..torch_defs import floatstr_torch

from ...utils.utt2info import Utt2Info
from .feat_seq_dataset import FeatSeqDataset


class PairedFeatSeqDataset(FeatSeqDataset):
    def __init__(
        self,
        rspecifier,
        key_file,
        pairs_file,
        class_file=None,
        num_frames_file=None,
        path_prefix=None,
        min_chunk_length=1,
        max_chunk_length=None,
        return_fullseqs=False,
        return_class=True,
        transpose_input=True,
        is_val=False,
    ):

        super().__init__(
            rspecifier,
            key_file,
            class_file=class_file,
            num_frames_file=num_frames_file,
            path_prefix=path_prefix,
            min_chunk_length=min_chunk_length,
            max_chunk_length=max_chunk_length,
            return_fullseqs=return_fullseqs,
            return_class=return_class,
            transpose_input=transpose_input,
            is_val=is_val,
        )

        logging.info("loading utt pairs file %s" % key_file)
        u2pair = Utt2Info.load(pairs_file, sep=" ")
        u2pair_dict = {}
        for u, p in u2pair:
            u2pair_dict[u] = p
        self.u2pair = u2pair_dict

    def _get_fullseq(self, index):
        key = self.u2c.key[index]
        x = self.r.read([key])[0].astype(floatstr_torch(), copy=False)
        key_pair = self.u2pair[key]
        x_pair = self.r.read([key_pair])[0].astype(floatstr_torch(), copy=False)
        if self.transpose_input:
            x = x.T
            x_pair = x_pair.T
        if not self.return_class:
            return x, x_pair

        class_idx = self.utt_idx2class[index]
        return x, x_pair, class_idx

    def _get_random_chunk(self, index):

        if len(index) == 2:
            index, chunk_length = index
        else:
            chunk_length = self.max_chunk_length

        key = self.u2c.key[index]
        full_seq_length = int(self.seq_lengths[index])
        assert (
            chunk_length <= full_seq_length
        ), "chunk_length(%d) <= full_seq_length(%d)" % (chunk_length, full_seq_length)
        first_frame = torch.randint(
            low=0, high=full_seq_length - chunk_length + 1, size=(1,)
        ).item()

        x = self.r.read([key], row_offset=first_frame, num_rows=chunk_length)[0]
        key_pair = self.u2pair[key]
        x_pair = self.r.read([key_pair], row_offset=first_frame, num_rows=chunk_length)[
            0
        ]

        x = x.astype(floatstr_torch(), copy=False)
        x_pair = x_pair.astype(floatstr_torch(), copy=False)
        if self.transpose_input:
            x = x.T
            x_pair = x_pair.T

        if not self.return_class:
            return x, x_pair

        class_idx = self.utt_idx2class[index]
        return x, x_pair, class_idx
