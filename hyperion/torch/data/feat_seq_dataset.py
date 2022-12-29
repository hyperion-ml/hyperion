"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import copy
import logging
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from jsonargparse import ActionParser, ArgumentParser, ActionYesNo
from torch.utils.data import Dataset

from ...io import RandomAccessDataReaderFactory as RF
from ...utils.misc import filter_func_args
from ...utils.class_info import ClassInfo
from ...utils.segment_set import SegmentSet
from ..torch_defs import floatstr_torch


class FeatSeqDataset(Dataset):
    def __init__(
        self,
        feat_file,
        segments_file,
        class_names=None,
        class_files=None,
        num_frames_file=None,
        return_segment_info=None,
        path_prefix=None,
        transpose_input=True,
        is_val=False,
    ):

        super().__init__()
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1

        if rank == 0:
            logging.info("opening feature reader %s", feat_file)

        self.r = RF.create(feat_file, path_prefix=path_prefix, scp_sep=" ")

        if rank == 0:
            logging.info("loading segments file %s" % segments_file)

        self.seg_set = SegmentSet.load(segments_file)
        if rank == 0:
            logging.info("dataset contains %d seqs", len(self.seg_set))

        self.is_val = is_val
        if num_frames_file is not None:
            if rank == 0:
                logging.info("loading durations file %s", num_frames_file)

            time_durs = SegmentSet.load(num_frames_file)
            self.seg_set["num_frames"] = time_durs.loc[
                self.seg_set["id"]
            ].class_id.values.astype(int, copy=False)
        else:
            assert "num_frames" in self.seg_set

        logging.info("loading class-info files")
        self._load_class_infos(class_names, class_files, is_val)

        self.return_segment_info = (
            [] if return_segment_info is None else return_segment_info
        )

        self.transpose_input = transpose_input

    def _load_class_infos(self, class_names, class_files, is_val):
        self.class_info = {}
        if class_names is None:
            assert class_files is None
            return

        assert len(class_names) == len(class_files)
        for name, file in zip(class_names, class_files):
            assert (
                name in self.seg_set
            ), f"class_name {name} not present in the segment set"
            if self.rank == 0:
                logging.info("loading class-info file %s" % file)
            table = ClassInfo.load(file)
            self.class_info[name] = table
            if not is_val:
                # check that all classes are present in the training segments
                class_ids = table["id"]
                segment_class_ids = self.seg_set[name].unique()
                for c_id in class_ids:
                    if c_id not in segment_class_ids:
                        logging.warning(
                            "%s class: %s not present in dataset", name, c_id
                        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    @property
    def num_seqs(self):
        return len(self.seg_set)

    def __len__(self):
        return self.num_seqs

    @property
    def seq_lengths(self):
        return self.seg_set["num_frames"]

    @property
    def total_length(self):
        return np.sum(self.seq_lengths)

    @property
    def min_seq_length(self):
        return np.min(self.seq_lengths)

    @property
    def max_seq_length(self):
        return np.max(self.seq_lengths)

    @property
    def num_classes(self):
        return {k: t.num_classes for k, t in self.class_info.items()}

    def _parse_segment_item(self, segment):
        if isinstance(segment, (tuple, list)):
            seg_id, start, num_frames = segment
            assert num_frames <= self.seg_set.loc[seg_id].num_frames, (
                f"{seg_id} with start={start} num_frames "
                f"({self.seg_set.loc[seg_id].num_frames}) < "
                f"chunk duration ({num_frames})"
            )
        else:
            seg_id, start, num_frames = segment, 0, 0

        if "start" in self.seg_set:
            start += self.seg_set.loc[seg_id].start

        return seg_id, int(start), int(num_frames)

    def _read_feats(self, seg_id, start, num_frames):
        x = self.r.read(seg_id, row_offset=start, num_rows=num_frames)[0].astype(
            floatstr_torch(), copy=False
        )
        return x

    def _get_segment_info(self, seg_id):
        seg_info = {}
        # converts the class_ids to integers
        for info_name in self.return_segment_info:
            seg_info_i = self.seg_set.loc[seg_id, info_name]
            if info_name in self.class_info:
                # if the type of information is a class-id
                # we use the class information table to
                # convert from id to integer
                class_info = self.class_info[info_name]
                seg_info_i = class_info.loc[seg_info_i, "class_idx"]

            seg_info[info_name] = seg_info_i

        return seg_info

    def __getitem__(self, segment):

        seg_id, start, num_frames = self._parse_segment_item(segment)
        x = self._read_feats(seg_id, start, num_frames)
        num_frames = x.shape[0]
        if self.transpose_input:
            x = x.T

        data = {"seg_id": seg_id, "x": x, "x_lengths": num_frames}

        # adds the segment labels
        seg_info = self._get_segment_info(seg_id)
        data.update(seg_info)
        return data

    @staticmethod
    def filter_args(**kwargs):
        return filter_func_args(FeatSeqDataset.__init__, kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "feat_file" not in skip:
            parser.add_argument(
                "--audio-file", required=True, help=("feature manifest file"),
            )

        if "segments_file" not in skip:
            parser.add_argument(
                "--segments-file", required=True, help=("segments manifest file"),
            )

        parser.add_argument(
            "--class-names",
            default=None,
            nargs="+",
            help=(
                "list with the names of the types of classes in the datasets, e.g., speaker, language"
            ),
        )

        parser.add_argument(
            "--class-files", default=None, nargs="+", help=("list of class info files"),
        )

        parser.add_argument(
            "--num-frames-file",
            default=None,
            help=("segment to num-frames file, if durations are not in segments_file"),
        )

        parser.add_argument(
            "--return-segment-info",
            default=None,
            nargs="+",
            help=(
                "list of columns of the segment file which should be returned as supervisions"
            ),
        )

        parser.add_argument(
            "--path-prefix", default="", help=("path prefix for rspecifier scp file")
        )
        RF.add_class_args(parser)
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
