#!/bin/env python
"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, namespace_to_dict
import logging
from pathlib import Path
import re
import numpy as np
import pandas as pd

from hyperion.hyp_defs import config_logger
from hyperion.utils import RecordingSet, FeatureSet, SegmentSet, ClassInfo


def split_train_val(
    segments_file,
    recordings_file,
    feats_file,
    durations_file,
    ara_ary_seg_file,
    in_class_name,
    out_class_name,
    val_percent,
    remove_langs,
    seed,
    output_dir,
    verbose,
):

    config_logger(verbose)
    rng = np.random.RandomState(seed=seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segs = SegmentSet.load(segments_file)
    if durations_file is not None:
        durs = SegmentSet.load(durations_file)
        if "duration" in durs:
            segs["duration"] = durs.loc[segs["id"], "duration"]
        else:
            segs["duration"] = durs.loc[segs["id"], "class_id"].astype(float)

    if remove_langs is not None:
        for lang in remove_langs:
            segs = segs[segs[in_class_name] != lang]

        segs = SegmentSet(segs)

    if ara_ary_seg_file is not None:
        segs_ary = SegmentSet.load(ara_ary_seg_file)
        segs.loc[segs_ary["id"], in_class_name] = segs_ary["class_id"]
        n1 = len(segs)
        noary_idx = segs[in_class_name] != "ara-ary"
        segs = SegmentSet(segs.loc[noary_idx])
        logging.info("removing ara-ary segments remained %d / %d", len(segs), n1)

    logging.info("creating class_info file")
    class_ids = segs[in_class_name].drop_duplicates().sort_values()
    class_info = ClassInfo(pd.DataFrame({"id": class_ids}))
    class_info.save(output_dir / "class_file.csv")

    logging.info("splitting segments into train and val")
    segs.set_index(in_class_name)
    val_mask = np.zeros((len(segs),), dtype=bool)
    for c in class_info["id"]:
        seg_idx_c = segs.get_loc(c)
        num_val = int(val_percent * len(seg_idx_c) / 100)
        val_idx = rng.choice(seg_idx_c, size=num_val, replace=False)
        val_mask[val_idx] = True
        logging.info(
            "class %s total=%d train=%d val=%d",
            c,
            len(seg_idx_c),
            len(seg_idx_c) - num_val,
            num_val,
        )

    segs.reset_index()
    if out_class_name is not None:
        segs.rename(columns={in_class_name: out_class_name}, inplace=True)

    train_segs = SegmentSet(segs.loc[~val_mask])
    train_segs.save(output_dir / "train_segments.csv")
    val_segs = SegmentSet(segs.loc[val_mask])
    val_segs.save(output_dir / "val_segments.csv")

    if recordings_file is not None:
        logging.info("splitting recordings into train and val")
        recs = RecordingSet.load(recordings_file)
        train_recs = RecordingSet(recs.loc[train_segs.recording_ids(train_segs["id"])])
        train_recs.save(output_dir / "train_recordings.csv")
        val_recs = RecordingSet(recs.loc[val_segs.recording_ids(val_segs["id"])])
        val_recs.save(output_dir / "val_recordings.csv")

    if feats_file is not None:
        logging.info("splitting features into train and val")
        feats = FeatureSet.load(feats_file)
        train_feats = FeatureSet(feats.loc[train_segs["id"]])
        train_feats.save(output_dir / "train_feats.csv")
        val_feats = FeatureSet(feats.loc[val_segs["id"]])
        val_feats.save(output_dir / "val_feats.csv")


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Split Segment list into training and validation"
    )
    parser.add_argument(
        "--segments-file", required=True, help="path to segments file",
    )
    parser.add_argument(
        "--recordings-file",
        default=None,
        help="if not None, splits recordings file into train and val",
    )

    parser.add_argument(
        "--durations-file",
        default=None,
        help="if not None, add durations to segments file",
    )

    parser.add_argument(
        "--feats-file",
        default=None,
        help="if not None, splits features file into train and val",
    )
    parser.add_argument(
        "--ara-ary-seg-file",
        default=None,
        help="segment-file with labels for Maghrebi Arabic",
    )

    parser.add_argument(
        "--in-class-name",
        default="class_id",
        help="column name containing the class_id that we consider to make the partition",
    )
    parser.add_argument(
        "--out-class-name",
        default=None,
        help="if not None, we rename the class_id column in the output file",
    )
    parser.add_argument(
        "--val-percent", default=5.0, type=float, help="percentage of data used for val"
    )
    parser.add_argument(
        "--remove-langs", default=None, nargs="+", help="remove languages from training"
    )
    parser.add_argument("--seed", default=1123, type=int, help="random seed")

    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    split_train_val(**namespace_to_dict(args))
