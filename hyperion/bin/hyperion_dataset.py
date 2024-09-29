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
    EnrollmentMap,
    FeatureSet,
    HypDataset,
    InfoTable,
    PathLike,
    RecordingSet,
    SegmentSet,
)

subcommand_list = [
    "add_features",
    "add_vads",
    "add_diarizations",
    "set_recordings",
    "make_from_recordings",
    "remove_short_segments",
    "rebuild_class_idx",
    "remove_classes_few_segments",
    "remove_classes_few_toomany_segments",
    "remove_class_ids",
    "split_train_val",
    "split_folds",
    "filter_by_classes",
    "filter_by_classes_and_enrollments",
    "copy",
    "add_cols_to_segments",
    "merge",
    "from_lhotse",
    "from_kaldi",
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

    logging.info(
        "adding features %s to dataset: %s -> %s",
        features_name,
        dataset,
        output_dataset,
    )

    dataset = HypDataset.load(dataset, lazy=True)
    dataset.add_features(features_name, features_file)
    dataset.save(output_dataset)


def make_add_vads_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument("--vads-name", required=True, help="""name of the feature""")
    parser.add_argument("--vads-file", required=True, help="""feature set file""")
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )

    add_common_args(parser)
    return parser


def add_vads(
    dataset: PathLike,
    vads_name: str,
    vads_file: PathLike,
    output_dataset: PathLike,
):
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "adding vads %s to dataset: %s -> %s",
        vads_name,
        dataset,
        output_dataset,
    )

    dataset = HypDataset.load(dataset, lazy=True)
    dataset.add_vads(vads_name, vads_file)
    dataset.save(output_dataset)


def make_add_diarizations_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--diarizations-name", required=True, help="""name of the feature"""
    )
    parser.add_argument(
        "--diarizations-file", required=True, help="""feature set file"""
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )

    add_common_args(parser)
    return parser


def add_diarizations(
    dataset: PathLike,
    diarizations_name: str,
    diarizations_file: PathLike,
    output_dataset: PathLike,
):
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "adding diarizations %s to dataset: %s -> %s",
        diarizations_name,
        dataset,
        output_dataset,
    )

    dataset = HypDataset.load(dataset, lazy=True)
    dataset.add_diarizations(diarizations_name, diarizations_file)
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

    logging.info(
        "setting recording %s in dataset: %s -> %s",
        recordings_file,
        dataset,
        output_dataset,
    )
    dataset = HypDataset.load(dataset, lazy=True)
    dataset.set_recordings(recordings_file, update_seg_durs)
    if remove_features is not None:
        logging.info("removing features %s", str(remove_features))
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

    logging.info("making dataset %s from recordings %s", dataset, recordings_file)
    rec_df = pd.read_csv(recordings_file)
    seg_df = rec_df[["id"]]
    segments = SegmentSet(seg_df)
    dataset = HypDataset(segments, recordings=recordings_file)
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

    logging.info(
        "removing segments with %s<%f in dataset: %s -> %s",
        length_name,
        min_length,
        dataset,
        output_dataset,
    )
    dataset = HypDataset.load(dataset, lazy=True)
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

    logging.info(
        "rebuilding %s class index in dataset: %s -> %s",
        class_name,
        dataset,
        output_dataset,
    )

    dataset = HypDataset.load(dataset, lazy=True)
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

    logging.info(
        "removing %s with segments<%d in dataset: %s -> %s",
        class_name,
        min_segs,
        dataset,
        output_dataset,
    )

    dataset = HypDataset.load(dataset, lazy=True)
    dataset.remove_classes_few_segments(class_name, min_segs, rebuild_idx)
    dataset.save(output_dataset)


def make_remove_classes_few_toomany_segments_parser():
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
        "--max-segs", default=None, type=int, help="""max. num. of segments/class"""
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


def remove_classes_few_toomany_segments(
    dataset: PathLike,
    class_name: str,
    min_segs: int,
    max_segs: Union[int, None],
    rebuild_idx: bool,
    output_dataset: PathLike,
):
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "removing %s with segments<%d or segments>%d in dataset: %s -> %s",
        class_name,
        min_segs,
        max_segs,
        dataset,
        output_dataset,
    )
    dataset = HypDataset.load(dataset, lazy=True)
    dataset.remove_classes_few_toomany_segments(
        class_name, min_segs, max_segs, rebuild_idx
    )
    dataset.save(output_dataset)


def make_remove_class_ids_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--class-name", required=True, help="""name of the class type e.g.: speaker"""
    )
    parser.add_argument(
        "--class-ids", default=None, nargs="+", help="""class ids to remove"""
    )
    parser.add_argument(
        "--remove-na",
        default=False,
        action=ActionYesNo,
        help="remove segments NA class id",
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


def remove_class_ids(
    dataset: PathLike,
    class_name: str,
    class_ids: List[str],
    remove_na: bool,
    rebuild_idx: bool,
    output_dataset: PathLike,
):
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "removing %s with ids %s in dataset: %s -> %s",
        class_name,
        str(class_ids),
        dataset,
        output_dataset,
    )

    dataset = HypDataset.load(dataset, lazy=True)
    dataset.remove_class_ids(class_name, class_ids, remove_na, rebuild_idx)
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
    logging.info(
        "splitting %s -> train: %s + val: %s",
        dataset,
        train_dataset,
        val_dataset,
    )
    dataset = HypDataset.load(dataset, lazy=True)
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


def make_split_folds_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""input dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--num-folds",
        default=5,
        type=int,
        help="""number of folds""",
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
        "--output-path",
        required=True,
        help="""output dir""",
    )

    add_common_args(parser)
    return parser


def split_folds(
    dataset: PathLike,
    num_folds: int,
    joint_classes: List[str],
    disjoint_classes: List[str],
    seed: int,
    output_path: PathLike,
):
    logging.info(
        "splitting %s -> %s",
        dataset,
        output_path,
    )

    dataset = HypDataset.load(dataset, lazy=True)
    train_folds, test_folds = dataset.split_folds(
        num_folds, joint_classes, disjoint_classes, seed
    )

    output_path = Path(output_path)
    for i, (train_fold, test_fold) in enumerate(zip(train_folds, test_folds)):
        output_dir_i = output_path / str(i)
        logging.info("fold %d -> %s", i, output_dir_i)
        train_fold.save(output_dir_i / "train")
        train_fold.describe()
        test_fold.save(output_dir_i / "test")
        test_fold.describe()


def make_filter_by_classes_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--class-name", required=True, help="""name of the class type e.g.: speaker"""
    )
    parser.add_argument(
        "--class-file",
        required=True,
        help="""name of the file containing the classes to keep""",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )
    parser.add_argument(
        "--rebuild-idx",
        default=False,
        action=ActionYesNo,
        help="""regenerate class indexes from 0 to new_num_classes-1""",
    )
    parser.add_argument(
        "--keep",
        default=True,
        action=ActionYesNo,
        help="""whether keep or remove the classes""",
    )

    add_common_args(parser)
    return parser


def filter_by_classes(
    dataset: PathLike,
    class_name: str,
    class_file: PathLike,
    output_dataset: PathLike,
    rebuild_idx: bool = False,
    keep: bool = True,
):
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "Dataset %s filtering %s in %s -> %s",
        dataset,
        class_name,
        class_file,
        output_dataset,
    )
    classes = ClassInfo.load(class_file)
    dataset = HypDataset.load(dataset, lazy=True)
    dataset.filter_by_classes(class_name, classes, rebuild_idx=rebuild_idx, keep=keep)
    dataset.save(output_dataset)


def make_filter_by_classes_and_enrollments_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--class-name", required=True, help="""name of the class type e.g.: speaker"""
    )
    parser.add_argument(
        "--class-file",
        required=True,
        help="""name of the file containing the classes to keep""",
    )
    parser.add_argument(
        "--enrollment-name",
        required=True,
        help="""name of the enrollment file in the dataset""",
    )
    parser.add_argument(
        "--enrollment-file",
        required=True,
        help="""name of the file containing the enrollment ids to keep""",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )
    parser.add_argument(
        "--rebuild-idx",
        default=False,
        action=ActionYesNo,
        help="""regenerate class indexes from 0 to new_num_classes-1""",
    )
    parser.add_argument(
        "--keep",
        default=True,
        action=ActionYesNo,
        help="""whether keep or remove the classes""",
    )

    add_common_args(parser)
    return parser


def filter_by_classes_and_enrollments(
    dataset: PathLike,
    class_name: str,
    class_file: PathLike,
    enrollment_name: str,
    enrollment_file: PathLike,
    output_dataset: PathLike,
    rebuild_idx: bool = False,
    keep: bool = True,
):
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "Dataset %s filtering %s in %s + %s in %s -> %s",
        dataset,
        class_name,
        class_file,
        enrollment_name,
        enrollment_file,
        output_dataset,
    )
    classes = ClassInfo.load(class_file)
    enrollments = EnrollmentMap.load(enrollment_file)
    dataset = HypDataset.load(dataset, lazy=True)
    dataset.filter_by_classes_and_enrollments(
        class_name,
        classes,
        enrollment_name,
        enrollments,
        rebuild_idx=rebuild_idx,
        keep=keep,
    )
    dataset.save(output_dataset)


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
    parser.add_argument(
        "--seg-suffix",
        default=None,
        help="append sufix to segment ids",
    )

    add_common_args(parser)
    return parser


def copy(
    dataset: PathLike,
    output_dataset: PathLike,
    seg_suffix: Optional[str] = None,
):
    logging.info(
        "copying dataset: %s -> %s",
        dataset,
        output_dataset,
    )
    dataset = HypDataset.load(dataset, lazy=True)
    if seg_suffix is not None:
        dataset.append_seg_suffix(seg_suffix)
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
        "--column-names",
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

    parser.add_argument(
        "--remove-missing",
        default=False,
        action=ActionYesNo,
        help="remove dataset entries that don't have a value in the right table",
    )

    parser.add_argument(
        "--create-class-info",
        default=False,
        action=ActionYesNo,
        help="creates class-info tables for the new columns added to the dataset",
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
    remove_missing: bool = False,
    create_class_info: bool = False,
):
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "adding columnts to %s + %s -> %s", dataset, right_table, output_dataset
    )
    dataset = HypDataset.load(dataset, lazy=True)
    dataset.add_cols_to_segments(
        right_table,
        column_names,
        on,
        right_on,
        remove_missing=remove_missing,
        create_class_info=create_class_info,
    )
    dataset.save(output_dataset)


def make_merge_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--input-datasets", required=True, nargs="+", help="input datasets"
    )
    add_common_args(parser)
    return parser


def merge(dataset: PathLike, input_datasets: List[PathLike]):
    input_dataset_paths = input_datasets
    dataset_path = dataset

    logging.info("merging %s -> %s", (input_dataset_paths), dataset_path)
    input_datasets = []
    for dset_file in input_dataset_paths:
        input_datasets.append(HypDataset.load(dset_file))

    dataset = HypDataset.merge(input_datasets)
    dataset.save(dataset_path)


def make_from_lhotse_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--cuts-file",
        default=None,
        help="lhotse cuts file",
    )
    parser.add_argument(
        "--recordings-file",
        default=None,
        help="lhotse recordings set file",
    )
    parser.add_argument(
        "--supervisions-file",
        default=None,
        help="lhotse supervisions file",
    )
    add_common_args(parser)
    return parser


def from_lhotse(
    dataset: PathLike,
    cuts_file: Optional[PathLike] = None,
    recordings_file: Optional[PathLike] = None,
    supervisions_file: Optional[PathLike] = None,
):
    assert cuts_file is not None or supervisions_file is not None
    logging.info(
        "crate dataset from lhotse : %s -> %s",
        cuts_file if cuts_file is not None else supervisions_file,
        dataset,
    )
    dataset_path = dataset
    dataset = HypDataset.from_lhotse(
        cuts=cuts_file, recordings=recordings_file, supervisions=supervisions_file
    )
    dataset.save(dataset_path)


def make_from_kaldi_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--kaldi-data-dir",
        required=True,
        help="Kaldi data directory",
    )
    add_common_args(parser)
    return parser


def from_kaldi(
    dataset: PathLike,
    kaldi_data_dir: PathLike,
):
    logging.info("crate dataset from kaldi : %s -> %s", kaldi_data_dir, dataset)
    dataset_path = dataset
    dataset = HypDataset.from_kaldi(kaldi_data_dir)
    dataset.save(dataset_path)


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
