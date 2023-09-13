#!/usr/bin/env python
"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path
from typing import List, Optional, Union

from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.utils import (
    ClassInfo,
    Dataset,
    EnrollmentMap,
    FeatureSet,
    InfoTable,
    PathLike,
    RecordingSet,
    SegmentSet,
)

subcommand_list = [
    "add_features",
    "set_recordings",
    "make_from_recordings",
    "remove_short_segments",
    "rebuild_class_idx",
    "remove_classes_few_segments",
    "split_train_val",
    "copy",
    "add_cols_to_segments",
]


def add_common_args(parser):
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
    )


def make_add_features_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--features-name", required=True, help="""name of the feature"""
    )
    parser.add_argument("--features-file", required=True, help="""feature set file""")
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )

    add_common_args(parser)
    return parser


def add_features(
    dataset: PathLike,
    features_name: str,
    features_file: PathLike,
    output_dataset: PathLike,
):
    if output_dataset is None:
        output_dataset = dataset

    dataset = Dataset.load(dataset, lazy=True)
    dataset.add_features(features_name, features_file)
    dataset.save(output_dataset)


def make_set_recordings_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--recordings-file", required=True, help="""recordings set file"""
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )
    parser.add_argument(
        "--remove-features",
        default=None,
        nargs="+",
        help="""removes feature files from the dataset, 
        since they maybe obsolote after modifiying the recordings""",
    )
    parser.add_argument(
        "--update-seg-durs",
        default=False,
        action=ActionYesNo,
        help="""updates the durations in the segment table""",
    )

    add_common_args(parser)
    return parser


def set_recordings(
    dataset: PathLike,
    recordings_file: PathLike,
    output_dataset: PathLike,
    remove_features: List[str],
    update_seg_durs: bool,
):
    if output_dataset is None:
        output_dataset = dataset

    dataset = Dataset.load(dataset, lazy=True)
    dataset.set_recordings(recordings_file, update_seg_durs)
    if remove_features is not None:
        for features_name in remove_features:
            dataset.remove_features(features_name)

    dataset.save(output_dataset)


def make_make_from_recordings_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--recordings-file", required=True, help="""recordings set file"""
    )

    add_common_args(parser)
    return parser


def make_from_recordings(
    dataset: PathLike,
    recordings_file: PathLike,
):
    output_dataset = dataset
    import pandas as pd

    rec_df = pd.read_csv(recordings_file)
    seg_df = rec_df[["id"]]
    segments = SegmentSet(seg_df)
    dataset = Dataset(segments, recordings=recordings_file)
    dataset.save(output_dataset)


def make_remove_short_segments_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--min-length",
        required=True,
        type=float,
        help="""minimum required length of the segment""",
    )

    parser.add_argument(
        "--length-name",
        default="duration",
        help="""name of the column indicating the length of the segment""",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )

    add_common_args(parser)
    return parser


def remove_short_segments(
    dataset: PathLike,
    min_length: float,
    length_name: str,
    output_dataset: PathLike,
):
    if output_dataset is None:
        output_dataset = dataset

    dataset = Dataset.load(dataset, lazy=True)
    dataset.remove_short_segments(min_length, length_name)
    dataset.save(output_dataset)


def make_rebuild_class_idx_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--class-name", required=True, help="""name of the class type e.g.: speaker"""
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )

    add_common_args(parser)
    return parser


def rebuild_class_idx(
    dataset: PathLike,
    class_name: str,
    output_dataset: PathLike,
):
    if output_dataset is None:
        output_dataset = dataset

    dataset = Dataset.load(dataset, lazy=True)
    dataset.rebuild_class_idx(class_name)
    dataset.save(output_dataset)


def make_remove_classes_few_segments_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--class-name", required=True, help="""name of the class type e.g.: speaker"""
    )
    parser.add_argument(
        "--min-segs", default=1, type=int, help="""min. num. of segments/class"""
    )
    parser.add_argument(
        "--rebuild-idx",
        default=False,
        action=ActionYesNo,
        help="""regenerate class indexes from 0 to new_num_classes-1""",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )

    add_common_args(parser)
    return parser


def remove_classes_few_segments(
    dataset: PathLike,
    class_name: str,
    min_segs: int,
    rebuild_idx: bool,
    output_dataset: PathLike,
):
    if output_dataset is None:
        output_dataset = dataset

    dataset = Dataset.load(dataset, lazy=True)
    dataset.remove_classes_few_segments(class_name, min_segs, rebuild_idx)
    dataset.save(output_dataset)


def make_split_train_val_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""input dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--val-prob",
        default=0.05,
        type=float,
        help="""proportion of segments used for val""",
    )
    parser.add_argument(
        "--min-train-samples",
        default=1,
        type=int,
        help="""min. number of training samples / class""",
    )

    parser.add_argument(
        "--joint-classes",
        default=None,
        nargs="+",
        help="""types of classes that need to have same classes in train and val""",
    )
    parser.add_argument(
        "--disjoint-classes",
        default=None,
        nargs="+",
        help="""types of classes that need to have different classes in train and val""",
    )
    parser.add_argument(
        "--seed",
        default=11235813,
        type=int,
        help="""random seed""",
    )

    parser.add_argument(
        "--train-dataset",
        required=True,
        help="""output train dataset dir""",
    )
    parser.add_argument(
        "--val-dataset",
        required=True,
        help="""output val dataset dir""",
    )

    add_common_args(parser)
    return parser


def split_train_val(
    dataset: PathLike,
    val_prob: float,
    joint_classes: List[str],
    disjoint_classes: List[str],
    min_train_samples: int,
    seed: int,
    train_dataset: PathLike,
    val_dataset: PathLike,
):
    dataset = Dataset.load(dataset, lazy=True)
    train_ds, val_ds = dataset.split_train_val(
        val_prob, joint_classes, disjoint_classes, min_train_samples, seed
    )
    train_ds.save(train_dataset)
    val_ds.save(val_dataset)

    num_total = len(dataset)
    num_train = len(train_ds)
    num_val = len(val_ds)
    logging.info(
        "train: %d (%.2f%%) segments, val: %d (%.2f%%) segments",
        num_train,
        num_train / num_total * 100,
        num_val,
        num_val / num_total * 100,
    )


def make_copy_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--output-dataset",
        required=True,
        help="""output dataset dir, if None, we use the same as input""",
    )

    add_common_args(parser)
    return parser


def copy(
    dataset: PathLike,
    output_dataset: PathLike,
):
    dataset = Dataset.load(dataset, lazy=True)
    dataset.save(output_dataset)


def make_add_cols_to_segments_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--right-table", required=True, help="table where the new data is"
    )
    parser.add_argument(
        "--columns",
        required=True,
        nargs="+",
        help="""columns to copy to segments table""",
    )
    parser.add_argument(
        "--on",
        default=["id"],
        nargs="+",
        help="""columns to match both tables rows""",
    )
    parser.add_argument(
        "--right-on",
        default=None,
        nargs="+",
        help="""columns to match both tables rows""",
    )

    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )

    add_common_args(parser)
    return parser


def add_cols_to_segments(
    dataset: PathLike,
    right_table: PathLike,
    column_names: List[str],
    on: List[str],
    right_on: List[str],
    output_dataset: PathLike,
):
    if output_dataset is None:
        output_dataset = dataset

    dataset = Dataset.load(dataset, lazy=True)
    dataset.add_cols_to_segments(right_table, column_names, on, right_on)
    dataset.save(output_dataset)


def main():
    parser = ArgumentParser(description="Tool to manipulates the Hyperion dataset")
    parser.add_argument("--cfg", action=ActionConfigFile)

    subcommands = parser.add_subcommands()
    for subcommand in subcommand_list:
        parser_func = f"make_{subcommand}_parser"
        subparser = globals()[parser_func]()
        subcommands.add_subcommand(subcommand, subparser)

    args = parser.parse_args()
    subcommand = args.subcommand
    kwargs = namespace_to_dict(args)[args.subcommand]
    config_logger(kwargs["verbose"])
    del kwargs["verbose"]
    del kwargs["cfg"]
    globals()[subcommand](**kwargs)


if __name__ == "__main__":
    main()
