"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import os
import logging
from jsonargparse import ArgumentParser, ActionParser
import time
import copy
import threading

import numpy as np
import pandas as pd

import torch

from ..torch_defs import floatstr_torch
from ...io import RandomAccessDataReaderFactory as RF
from ...utils.utt2info import Utt2Info

from torch.utils.data import Dataset


class FeatSeqDataset(Dataset):
    def __init__(
        self,
        rspecifier,
        key_file,
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

        logging.info("opening dataset %s" % rspecifier)
        self.r = RF.create(rspecifier, path_prefix=path_prefix, scp_sep=" ")
        logging.info("loading utt2info file %s" % key_file)
        self.u2c = Utt2Info.load(key_file, sep=" ")
        logging.info("dataset contains %d seqs" % self.num_seqs)

        self.is_val = is_val
        self._seq_lengths = None
        if num_frames_file is not None:
            self._read_num_frames_file(num_frames_file)
        self._prune_short_seqs(min_chunk_length)

        self.short_seq_exist = self._seq_shorter_than_max_length_exists(
            max_chunk_length
        )

        self._prepare_class_info(class_file)

        if max_chunk_length is None:
            max_chunk_length = min_chunk_length
        self._min_chunk_length = min_chunk_length
        self._max_chunk_length = max_chunk_length

        self.return_fullseqs = return_fullseqs
        self.return_class = return_class

        self.transpose_input = transpose_input

    def _read_num_frames_file(self, file_path):
        logging.info("reading num_frames file %s" % file_path)
        nf_df = pd.read_csv(file_path, header=None, sep=" ")
        nf_df.index = nf_df[0]
        self._seq_lengths = nf_df.loc[self.u2c.key, 1].values

    @property
    def num_seqs(self):
        return len(self.u2c)

    def __len__(self):
        return self.num_seqs

    @property
    def seq_lengths(self):
        if self._seq_lengths is None:
            self._seq_lengths = self.r.read_num_rows(self.u2c.key)

        return self._seq_lengths

    @property
    def total_length(self):
        return np.sum(self.seq_lengths)

    @property
    def min_chunk_length(self):
        if self.return_fullseqs:
            self._min_chunk_length = np.min(self.seq_lengths)
        return self._min_chunk_length

    @property
    def max_chunk_length(self):
        if self._max_chunk_length is None:
            self._max_chunk_length = np.max(self.seq_lengths)
        return self._max_chunk_length

    @property
    def min_seq_length(self):
        return np.min(self.seq_lengths)

    @property
    def max_seq_length(self):
        return np.max(self.seq_lengths)

    def _prune_short_seqs(self, min_length):
        logging.info("pruning short seqs")
        keep_idx = self.seq_lengths >= min_length
        self.u2c = self.u2c.filter_index(keep_idx)
        self._seq_lengths = self.seq_lengths[keep_idx]
        logging.info(
            "pruned seqs with min_length < %d,"
            "keep %d/%d seqs" % (min_length, self.num_seqs, len(keep_idx))
        )

    def _prepare_class_info(self, class_file):
        class_weights = None
        if class_file is None:
            classes, class_idx = np.unique(self.u2c.info, return_inverse=True)
            class2idx = {k: i for i, k in enumerate(classes)}
        else:
            logging.info("reading class-file %s" % (class_file))
            class_info = pd.read_csv(class_file, header=None, sep=" ")
            class2idx = {str(k): i for i, k in enumerate(class_info[0])}
            class_idx = np.array([class2idx[k] for k in self.u2c.info], dtype=int)
            if class_info.shape[1] == 2:
                class_weights = np.array(class_info[1]).astype(
                    floatstr_torch(), copy=False
                )

        self.num_classes = len(class2idx)

        class2utt_idx = {}
        class2num_utt = np.zeros((self.num_classes,), dtype=int)

        for k in range(self.num_classes):
            idx = (class_idx == k).nonzero()[0]
            class2utt_idx[k] = idx
            class2num_utt[k] = len(idx)
            if class2num_utt[k] == 0:
                if not self.is_val:
                    logging.warning("class %d doesn't have any samples" % (k))
                if class_weights is None:
                    class_weights = np.ones((self.num_classes,), dtype=floatstr_torch())
                class_weights[k] = 0

        count_empty = np.sum(class2num_utt == 0)
        if count_empty > 0:
            logging.warning("%d classes have 0 samples" % (count_empty))

        self.utt_idx2class = class_idx
        self.class2utt_idx = class2utt_idx
        self.class2num_utt = class2num_utt
        if class_weights is not None:
            class_weights /= np.sum(class_weights)
            class_weights = torch.Tensor(class_weights)
        self.class_weights = class_weights

        if self.short_seq_exist:
            # if there are seq shorter than max_chunk_lenght we need some extra variables
            # we will need class_weights to put to 0 classes that have all utts shorter than the batch chunk length
            if self.class_weights is None:
                self.class_weights = torch.ones((self.num_classes,))

            # we need the max length of the utterances of each class
            class2max_length = torch.zeros((self.num_classes,), dtype=torch.int)
            for c in range(self.num_classes):
                if class2num_utt[c] > 0:
                    class2max_length[c] = int(
                        np.max(self.seq_lengths[self.class2utt_idx[c]])
                    )

            self.class2max_length = class2max_length

    def _seq_shorter_than_max_length_exists(self, max_length):
        return np.any(self.seq_lengths < max_length)

    @property
    def var_chunk_length(self):
        return self.min_chunk_length < self.max_chunk_length

    def get_random_chunk_length(self):

        if self.var_chunk_length:
            return torch.randint(
                low=self.min_chunk_length, high=self.max_chunk_length + 1, size=(1,)
            ).item()

        return self.max_chunk_length

    # def get_random_chunk_length(self, index):

    #     if self.min_chunk_length < self.max_chunk_length:
    #         if self.short_seq_exist:
    #             max_chunk_length = min(int(np.min(self.seq_lengths[index])),
    #                                    self.max_chunk_length)
    #         else:
    #             max_chunk_length = self.max_chunk_length

    #         chunk_length = torch.randint(
    #             low=self.min_chunk_length, high=max_chunk_length+1, size=(1,)).item()

    #         # logging.info('{} {} {} set_random_chunk_length={}'.format(
    #         #     self,os.getpid(), threading.get_ident(), chunk_length))
    #         return chunk_length

    #     return self.max_chunk_length

    def __getitem__(self, index):
        # logging.info('{} {} {} get item {}'.format(
        #     self, os.getpid(), threading.get_ident(), index))
        if self.return_fullseqs:
            return self._get_fullseq(index)
        else:
            return self._get_random_chunk(index)

    def _get_fullseq(self, index):
        key = self.u2c.key[index]
        x = self.r.read([key])[0].astype(floatstr_torch(), copy=False)
        if self.transpose_input:
            x = x.T
        if not self.return_class:
            return x

        class_idx = self.utt_idx2class[index]
        return x, class_idx

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

        x = self.r.read([key], row_offset=first_frame, num_rows=chunk_length)[0].astype(
            floatstr_torch(), copy=False
        )
        if self.transpose_input:
            x = x.T

        if not self.return_class:
            return x

        class_idx = self.utt_idx2class[index]
        return x, class_idx

    @staticmethod
    def filter_args(**kwargs):
        valid_args = (
            "path_prefix",
            "class_file",
            "num_frames_file",
            "min_chunk_length",
            "max_chunk_length",
            "return_fullseqs",
            "part_idx",
            "num_parts",
        )
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--path-prefix", default="", help=("path prefix for rspecifier scp file")
        )

        parser.add_argument(
            "--class-file",
            default=None,
            help=("ordered list of classes keys, it can contain class weights"),
        )

        parser.add_argument(
            "--num-frames-file",
            default=None,
            help=(
                "utt to num_frames file, if None it reads from the dataset "
                "but it is slow"
            ),
        )

        parser.add_argument(
            "--min-chunk-length",
            type=int,
            default=None,
            help=("minimum length of sequence chunks"),
        )
        parser.add_argument(
            "--max-chunk-length",
            type=int,
            default=None,
            help=("maximum length of sequence chunks"),
        )

        parser.add_argument(
            "--return-fullseqs",
            default=False,
            action="store_true",
            help=("returns full sequences instead of chunks"),
        )

        # parser.add_argument('--part-idx',
        #                     type=int, default=1,
        #                     help=('splits the list of files in num-parts and process part_idx'))
        # parser.add_argument('--num-parts',
        #                     type=int, default=1,
        #                     help=('splits the list of files in num-parts and process part_idx'))
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='feature sequence dataset options')

    add_argparse_args = add_class_args
