"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from jsonargparse import ArgumentParser, ActionParser
import time
import math

import numpy as np
import pandas as pd

import torch

from ..torch_defs import floatstr_torch
from ...io import RandomAccessAudioReader as AR
from ...utils.utt2info import Utt2Info
from ...augment import SpeechAugment

from torch.utils.data import Dataset
import torch.distributed as dist


class AudioDataset(Dataset):
    def __init__(
        self,
        audio_path,
        key_file,
        class_file=None,
        time_durs_file=None,
        min_chunk_length=1,
        max_chunk_length=None,
        aug_cfg=None,
        return_fullseqs=False,
        return_class=True,
        return_clean_aug_pair=False,
        transpose_input=False,
        wav_scale=2 ** 15 - 1,
        is_val=False,
    ):

        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1

        self.rank = rank
        self.world_size = world_size

        if rank == 0:
            logging.info("opening dataset %s" % audio_path)
        self.r = AR(audio_path, wav_scale=wav_scale)
        if rank == 0:
            logging.info("loading utt2info file %s" % key_file)
        self.u2c = Utt2Info.load(key_file, sep=" ")
        if rank == 0:
            logging.info("dataset contains %d seqs" % self.num_seqs)

        self.is_val = is_val
        self._read_time_durs_file(time_durs_file)

        # self._seq_lengths = self.r.read_time_duration(self.u2c.key)
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
        self.return_clean_aug_pair = return_clean_aug_pair

        self.transpose_input = transpose_input

        self.augmenter = None
        self.reverb_context = 0
        if aug_cfg is not None:
            self.augmenter = SpeechAugment.create(
                aug_cfg, random_seed=112358 + 1000 * rank
            )
            self.reverb_context = self.augmenter.max_reverb_context

    def _read_time_durs_file(self, file_path):
        if self.rank == 0:
            logging.info("reading time_durs file %s" % file_path)
        nf_df = pd.read_csv(file_path, header=None, sep=" ")
        nf_df.index = nf_df[0]
        self._seq_lengths = nf_df.loc[self.u2c.key, 1].values

    @property
    def wav_scale(self):
        return self.r.wav_scale

    @property
    def num_seqs(self):
        return len(self.u2c)

    def __len__(self):
        return self.num_seqs

    @property
    def seq_lengths(self):
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
        if self.rank == 0:
            logging.info("pruning short seqs")
        keep_idx = self.seq_lengths >= min_length
        self.u2c = self.u2c.filter_index(keep_idx)
        self._seq_lengths = self.seq_lengths[keep_idx]
        if self.rank == 0:
            logging.info(
                "pruned seqs with min_length < %f,"
                "keep %d/%d seqs" % (min_length, self.num_seqs, len(keep_idx))
            )

    def _prepare_class_info(self, class_file):
        class_weights = None
        if class_file is None:
            classes, class_idx = np.unique(self.u2c.info, return_inverse=True)
            class2idx = {k: i for i, k in enumerate(classes)}
        else:
            if self.rank == 0:
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
            class2max_length = torch.zeros((self.num_classes,), dtype=torch.float)
            for c in range(self.num_classes):
                if class2num_utt[c] > 0:
                    class2max_length[c] = np.max(
                        self.seq_lengths[self.class2utt_idx[c]]
                    )

            self.class2max_length = class2max_length

    def _seq_shorter_than_max_length_exists(self, max_length):
        return np.any(self.seq_lengths < max_length)

    @property
    def var_chunk_length(self):
        return self.min_chunk_length < self.max_chunk_length

    def get_random_chunk_length(self):

        if self.var_chunk_length:
            return (
                torch.rand(size=(1,)).item()
                * (self.max_chunk_length - self.min_chunk_length)
                + self.min_chunk_length
            )

        return self.max_chunk_length

    def __getitem__(self, index):
        # logging.info('{} {} {} get item {}'.format(
        #     self, os.getpid(), threading.get_ident(), index))
        if self.return_fullseqs:
            return self._get_fullseq(index)
        else:
            return self._get_random_chunk(index)

    def _get_fullseq(self, index):
        key = self.u2c.key[index]
        x, fs = self.r.read([key])
        x = x[0].astype(floatstr_torch(), copy=False)
        x_clean = x
        if self.augmenter is not None:
            x, aug_info = self.augmenter(x)

        if self.transpose_input:
            x = x[None, :]
            if self.return_clean_aug_pair:
                x_clean = x_clean[None, :]

        if self.return_clean_aug_pair:
            r = x, x_clean

        if not self.return_class:
            return r

        class_idx = self.utt_idx2class[index]
        r = *r, class_idx
        return r

    def _get_random_chunk(self, index):

        if len(index) == 2:
            index, chunk_length = index
        else:
            chunk_length = self.max_chunk_length

        key = self.u2c.key[index]

        full_seq_length = self.seq_lengths[index]
        assert (
            chunk_length <= full_seq_length
        ), "chunk_length(%d) <= full_seq_length(%d)" % (chunk_length, full_seq_length)

        time_offset = torch.rand(size=(1,)).item() * (full_seq_length - chunk_length)
        reverb_context = min(self.reverb_context, time_offset)
        time_offset -= reverb_context
        read_chunk_length = chunk_length + reverb_context

        # logging.info('get-random-chunk {} {} {} {} {}'.format(index, key, time_offset, chunk_length, full_seq_length ))
        x, fs = self.r.read([key], time_offset=time_offset, time_durs=read_chunk_length)

        # try:
        #     x, fs = self.r.read([key], time_offset=time_offset,
        #                     time_durs=read_chunk_length)
        # except:
        #     # some files produce error in the fseek after reading the data,
        #     # this seems an issue from pysoundfile or soundfile lib itself
        #     # reading from a sligthly different starting position seems to solve the problem in most cases
        #     try:
        #         logging.info('error-1 reading at key={} totol_dur={} offset={} read_chunk_length={}, retrying...'.format(
        #             key, full_seq_length, time_offset, read_chunk_length))
        #         time_offset = math.floor(time_offset)
        #         x, fs = self.r.read([key], time_offset=time_offset,
        #                             time_durs=read_chunk_length)
        #     except:
        #         try:
        #             # if changing the value of time-offset doesn't solve the issue, we try to read from
        #             # from time-offset to the end of the file, and remove the extra frames later
        #             logging.info('error-2 reading at key={} totol_dur={} offset={} retrying reading until end-of-file ...'.format(
        #                 key, full_seq_length, time_offset))
        #             x, fs = self.r.read([key], time_offset=time_offset)
        #             x = [x[0][:int(read_chunk_length * fs[0])]]
        #         except:
        #             # try to read the full file
        #             logging.info('error-3 reading at key={} totol_dur={} retrying reading full file ...'.format(
        #                 key, full_seq_length))
        #             x, fs = self.r.read([key])
        #             x = [x[0][:int(read_chunk_length * fs[0])]]

        x = x[0]
        fs = fs[0]

        x_clean = x
        logging.info("hola1")
        if self.augmenter is not None:
            logging.info("hola2")
            chunk_length_samples = int(chunk_length * fs)
            end_idx = len(x)
            reverb_context_samples = end_idx - chunk_length_samples
            assert reverb_context_samples >= 0, (
                "key={} time-offset={}, read-chunk={} "
                "read-x-samples={}, chunk_samples={}, reverb_context_samples={}"
            ).format(
                key,
                time_offset,
                read_chunk_length,
                end_idx,
                chunk_length_samples,
                reverb_context_samples,
            )
            # end_idx = reverb_context_samples + chunk_length_samples
            x, aug_info = self.augmenter(x)
            x = x[reverb_context_samples:end_idx]
            if self.return_clean_aug_pair:
                x_clean = x_clean[reverb_context_samples:end_idx]
                x_clean = x_clean.astype(floatstr_torch(), copy=False)
            # x_clean = x_clean[reverb_context_samples:]
            # logging.info('augmentation x-clean={}, x={}, aug_info={}'.format(
            #    x_clean.shape, x.shape, aug_info))
        #     if len(x) != 64000:
        #         logging.info('x!=4s, {} {} {} {} {} {} {} {}'.format(len(x),reverb_context, reverb_context_samples, chunk_length, chunk_length_samples, end_idx, fs, read_chunk_length))

        # if len(x) != 64000:
        #         logging.info('x!=4s-2, {} {} {} {}'.format(len(x), chunk_length, fs, read_chunk_length))

        if self.transpose_input:
            x = x[None, :]
            if self.return_clean_aug_pair:
                x_clean = x_clean[None, :]

        x = x.astype(floatstr_torch(), copy=False)
        if self.return_clean_aug_pair:
            r = x, x_clean
        else:
            r = (x,)

        if not self.return_class:
            return r

        class_idx = self.utt_idx2class[index]
        r = *r, class_idx
        return r

    @staticmethod
    def filter_args(**kwargs):

        ar_args = AR.filter_args(**kwargs)
        valid_args = (
            "path_prefix",
            "class_file",
            "time_durs_file",
            "min_chunk_length",
            "max_chunk_length",
            "return_fullseqs",
            "part_idx",
            "num_parts",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        args.update(ar_args)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        # parser.add_argument('--path-prefix',
        #                     default='',
        #                     help=('path prefix for rspecifier scp file'))

        parser.add_argument(
            "--class-file",
            default=None,
            help=("ordered list of classes keys, it can contain class weights"),
        )

        parser.add_argument(
            "--time-durs-file", default=None, help=("utt to duration in secs file")
        )

        parser.add_argument(
            "--min-chunk-length",
            type=float,
            default=None,
            help=("minimum length of sequence chunks"),
        )
        parser.add_argument(
            "--max-chunk-length",
            type=float,
            default=None,
            help=("maximum length of sequence chunks"),
        )

        parser.add_argument(
            "--return-fullseqs",
            default=False,
            action="store_true",
            help=("returns full sequences instead of chunks"),
        )

        AR.add_class_args(parser)
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='audio dataset options')

    add_argparse_args = add_class_args
