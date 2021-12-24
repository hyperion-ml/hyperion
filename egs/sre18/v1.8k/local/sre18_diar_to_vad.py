#!/usr/bin/env python

# Copyright 2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#

import sys
import os
import argparse
import time

import numpy as np
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Diarization file to binary vad",
    )

    parser.add_argument(dest="diar_file")
    parser.add_argument(dest="num_frames_file")

    args = parser.parse_args()

    utt2num_frames = pd.read_csv(
        args.num_frames_file,
        sep=" ",
        header=None,
        names=["utt", "num_frames", "None"],
        index_col=0,
    )
    diar = pd.read_csv(
        args.diar_file, sep=" ", header=None, names=["utt", "start", "end"], index_col=0
    )

    for key in utt2num_frames.index.values:
        num_frames_i = utt2num_frames["num_frames"][key]
        vad = np.zeros((num_frames_i,), dtype=int)
        start_i = np.array(diar.loc[key]["start"], dtype=int)
        end_i = np.array(diar.loc[key]["end"], dtype=int)
        if start_i.ndim == 0:
            start_i = [start_i]
            end_i = [end_i]
        for s, e in zip(start_i, end_i):
            if e > num_frames_i - 1:
                e = num_frames_i - 1
            vad[s : e + 1] = 1

        svad = key + " [ " + " ".join([str(v) for v in vad]) + " ]"
        print(svad)
