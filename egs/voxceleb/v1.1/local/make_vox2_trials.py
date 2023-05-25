#!/bin/env python
"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, namespace_to_dict
import logging
from pathlib import Path
import math
import numpy as np
import pandas as pd

from hyperion.hyp_defs import config_logger
from hyperion.utils.segment_set import SegmentSet


def make_trials_single_gender(ft, fm, fs, segments, num_tar_trials, num_spks):

    # select spks
    rng = np.random.RandomState(seed=1123)
    spks = segments["class_id"].unique()
    spks = rng.choice(spks, size=(num_spks,), replace=False)
    snorm_segments = segments[~segments["class_id"].isin(spks)]
    for seg, spk in zip(snorm_segments["id"], snorm_segments["class_id"]):
        fs.write("%s %s\n" % (seg, spk))

    segments = segments[segments["class_id"].isin(spks)]
    num_segs_per_spk = int(
        math.ceil((1 + math.sqrt(1 + 8 * num_tar_trials // num_spks)) / 2)
    )

    n = num_spks * num_segs_per_spk
    print(num_segs_per_spk, n, num_tar_trials // num_spks, num_spks, len(spks))
    seg_ids = rng.choice(segments["id"], size=(n,), replace=False)
    segments = segments[segments["id"].isin(seg_ids)]
    seg_ids = segments["id"].values
    class_ids = segments["class_id"].values
    ntar = 0
    nnon = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            t = "target" if class_ids[i] == class_ids[j] else "nontarget"
            ft.write("%s %s %s\n" % (seg_ids[i], seg_ids[j], t))
            if t == "target":
                ntar += 1
            else:
                nnon += 1

    logging.info("Got ntar=%d and nnon=%d", ntar, nnon)
    for i in range(n - 1):
        fm.write("%s %s\n" % (seg_ids[i], seg_ids[i]))


def make_trials(data_dir, num_1k_tar_trials, num_spks):
    config_logger(1)
    logging.info("Making trial list for %s", data_dir)
    data_dir = Path(data_dir)
    segments = SegmentSet.load(data_dir / "utt2spk")
    gender = SegmentSet.load(data_dir / "spk2gender")
    segments["gender"] = gender.loc[segments["class_id"], "class_id"].values

    num_tar_trials = num_1k_tar_trials * 1000 // 2
    num_spks = num_spks // 2
    with open(data_dir / "trials", "w") as ft, open(
        data_dir / "utt2model", "w"
    ) as fm, open(data_dir / "snorm_utt2spk", "w") as fs:
        segs_m = SegmentSet(segments.loc[segments["gender"] == "m"])
        make_trials_single_gender(ft, fm, fs, segs_m, num_tar_trials, num_spks)
        segs_f = SegmentSet(segments.loc[segments["gender"] == "f"])
        make_trials_single_gender(ft, fm, fs, segs_f, num_tar_trials, num_spks)


if __name__ == "__main__":

    parser = ArgumentParser(description="makes a trial list for vox2 dev")

    parser.add_argument("--data-dir", required=True, help="Path to dataset")
    parser.add_argument(
        "--num-1k-tar-trials", type=int, default=30, help="thousands of target trials"
    )
    parser.add_argument("--num-spks", type=int, default=1000, help="number of speakers")
    args = parser.parse_args()
    make_trials(**namespace_to_dict(args))
