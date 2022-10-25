#!/usr/bin/env python
# Copyright 2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#

import sys
import os
import argparse
import time
import logging

import numpy as np
import pandas as pd

from hyperion.hyp_defs import config_logger
from hyperion.utils import SegmentList, RTTM
from hyperion.io import DataWriterFactory as DWF


def rttm_to_bin_vad(
    rttm_file,
    num_frames_file,
    frame_shift,
    output_path,
    fix_empy_files,
    part_idx,
    num_parts,
):

    num_frames = None
    if num_frames_file is not None:
        utt2num_frames = pd.read_csv(
            num_frames_file,
            sep="\s+",
            header=None,
            names=["file_id", "num_frames"],
            index_col=0,
        )

    segments = RTTM.load(rttm_file).to_segment_list()

    if num_parts > 1:
        if fix_empy_files:
            segments_orig = copy.deepcopy(segments)
        segments = segments.split(part_idx, num_parts)
    else:
        segments_orig = segments

    with DWF.create(output_path) as writer:
        for file_id in segments.uniq_file_id:
            logging.info("processing VAD for %s" % (file_id))
            if num_frames_file is not None:
                num_frames = int(utt2num_frames.loc[file_id]["num_frames"])
            vad = segments.to_bin_vad(
                file_id, frame_shift=frame_shift, num_frames=num_frames
            )
            num_speech_frames = np.sum(vad)
            logging.info(
                "for %s detected %d/%d (%.2f %%) speech frames"
                % (
                    file_id,
                    num_speech_frames,
                    num_frames,
                    num_speech_frames / num_frames * 100,
                )
            )
            writer.write(file_id, vad)

        if fix_empy_files and part_idx == 1:
            for file_id in utt2num_frames.index:
                if not (file_id in segments_orig.uniq_file_id):
                    logging.warning(
                        "not speeech detected in %s, putting all to 1" % (file_id)
                    )
                    num_frames = int(utt2num_frames.loc[file_id]["num_frames"])
                    vad = np.ones((num_frames,), dtype="float32")
                    writer.write(file_id, vad)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="RTTM file to binary vad",
    )

    parser.add_argument("--rttm", dest="rttm_file", required=True, help="rttm file")
    parser.add_argument(
        "--num-frames",
        dest="num_frames_file",
        default=None,
        help="num. frames in feature matrix",
    )
    parser.add_argument(
        "--frame-shift",
        dest="frame_shift",
        default=10,
        type=float,
        help="frame shift of feature matrix in ms.",
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        required=True,
        help="wspecifier for binary vad file",
    )
    parser.add_argument(
        "--fix-empy-files",
        dest="fix_empy_files",
        default=False,
        action="store_true",
        help="puts all vad frames to 1 when file is missing in rttm",
    )

    parser.add_argument(
        "--part-idx",
        dest="part_idx",
        type=int,
        default=1,
        help=("splits the list of files in num-parts and process part_idx"),
    )
    parser.add_argument(
        "--num-parts",
        dest="num_parts",
        type=int,
        default=1,
        help=("splits the list of files in num-parts and process part_idx"),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="Verbose level",
    )
    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    rttm_to_bin_vad(**vars(args))
