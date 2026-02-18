#!/usr/bin/env python
"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path
from typing import List, Optional

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
    HyperDataset,
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
    "from_recordings",
    "from_segments",
    "add_classes_from_segments",
    "remove_short_segments",
    "rebuild_class_idx",
    "remove_classes_few_segments",
    "remove_classes_few_toomany_segments",
    "remove_class_ids",
    "split_train_val",
    "split_folds",
    "filter_by_segments",
    "filter_by_segments_predicate",
    "filter_by_classes",
    "filter_by_classes_and_enrollments",
    "copy",
    "clean",
    "sample_random_subsegments",
    "add_cols_to_segments",
    "merge",
    "from_lhotse",
    "from_kaldi",
    "describe",
]


def add_common_args(parser: ArgumentParser) -> None:
    """Add common CLI options shared by all subcommands.

    Args:
        parser: Argument parser to augment.
    """
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="Verbosity level: 0=error, 1=warning, 2=info, 3=debug.",
    )


def make_add_features_parser() -> ArgumentParser:
    """Create parser for the ``add_features`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
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
    output_dataset: Optional[PathLike],
) -> None:
    """Add a feature table to a dataset.

    Args:
        dataset: Input dataset directory or YAML file.
        features_name: Feature table name in the dataset.
        features_file: FeatureSet file to add.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
    """
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "adding features %s to dataset: %s -> %s",
        features_name,
        dataset,
        output_dataset,
    )

    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.add_features(features_name, features_file)
    dataset.save(output_dataset)


def make_add_vads_parser() -> ArgumentParser:
    """Create parser for the ``add_vads`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument("--vads-name", required=True, help="""name of the VAD set""")
    parser.add_argument("--vads-file", required=True, help="""VAD set file""")
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
    output_dataset: Optional[PathLike],
) -> None:
    """Add a VAD table to a dataset.

    Args:
        dataset: Input dataset directory or YAML file.
        vads_name: VAD table name in the dataset.
        vads_file: VAD file to add.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
    """
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "adding vads %s to dataset: %s -> %s",
        vads_name,
        dataset,
        output_dataset,
    )

    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.add_vads(vads_name, vads_file)
    dataset.save(output_dataset)


def make_add_diarizations_parser() -> ArgumentParser:
    """Create parser for the ``add_diarizations`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--diarizations-name", required=True, help="""name of the diarization table"""
    )
    parser.add_argument(
        "--diarizations-file", required=True, help="""diarization table file"""
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
    output_dataset: Optional[PathLike],
) -> None:
    """Add a diarization table to a dataset.

    Args:
        dataset: Input dataset directory or YAML file.
        diarizations_name: Diarization table name in the dataset.
        diarizations_file: Diarization file to add.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
    """
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "adding diarizations %s to dataset: %s -> %s",
        diarizations_name,
        dataset,
        output_dataset,
    )

    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.add_diarizations(diarizations_name, diarizations_file)
    dataset.save(output_dataset)


def make_set_recordings_parser() -> ArgumentParser:
    """Create parser for the ``set_recordings`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
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
        help="""remove feature tables that may become obsolete after updating recordings""",
    )
    parser.add_argument(
        "--remove-vads",
        default=None,
        nargs="+",
        help="""remove VAD tables that may become obsolete after updating recordings""",
    )
    parser.add_argument(
        "--remove-diarizations",
        default=None,
        nargs="+",
        help="""remove diarization tables that may become obsolete after updating recordings""",
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
    output_dataset: Optional[PathLike],
    remove_features: Optional[List[str]],
    remove_vads: Optional[List[str]],
    remove_diarizations: Optional[List[str]],
    update_seg_durs: bool,
) -> None:
    """Replace dataset recordings and optionally drop stale attached tables.

    Args:
        dataset: Input dataset directory or YAML file.
        recordings_file: New RecordingSet file.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
        remove_features: Feature table names to remove.
        remove_vads: VAD table names to remove.
        remove_diarizations: Diarization table names to remove.
        update_seg_durs: Whether to recompute segment durations.
    """
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "setting recording %s in dataset: %s -> %s",
        recordings_file,
        dataset,
        output_dataset,
    )
    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.set_recordings(recordings_file, update_seg_durs)
    if remove_features is not None:
        logging.info("removing features %s", str(remove_features))
        for features_name in remove_features:
            dataset.remove_features(features_name)

    if remove_vads is not None:
        logging.info("removing vads %s", str(remove_vads))
        for vad_name in remove_vads:
            dataset.remove_vads(vad_name)

    if remove_diarizations is not None:
        logging.info("removing diarization %s", str(remove_diarizations))
        for diar_name in remove_diarizations:
            dataset.remove_diarizations(diar_name)

    dataset.save(output_dataset)


def make_make_from_recordings_parser() -> ArgumentParser:
    """Create parser for the deprecated ``make_from_recordings`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
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
) -> None:
    """Create a dataset from recordings.

    Args:
        dataset: Output dataset directory or YAML file.
        recordings_file: RecordingSet file.
    """
    output_dataset = dataset
    logging.info("making dataset %s from recordings %s", dataset, recordings_file)
    dataset = HyperDataset.from_recordings(recordings_file)
    dataset.save(output_dataset)


def make_from_recordings_parser() -> ArgumentParser:
    """Create parser for the ``from_recordings`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--recordings-file", required=True, help="""recordings set file"""
    )

    add_common_args(parser)
    return parser


def from_recordings(
    dataset: PathLike,
    recordings_file: PathLike,
) -> None:
    """Create a dataset from recordings.

    Args:
        dataset: Output dataset directory or YAML file.
        recordings_file: RecordingSet file.
    """
    output_dataset = dataset
    logging.info("making dataset %s from recordings %s", dataset, recordings_file)
    dataset = HyperDataset.from_recordings(recordings_file)
    dataset.save(output_dataset)


def make_from_segments_parser() -> ArgumentParser:
    """Create parser for the ``from_segments`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument("--segments-file", required=True, help="""segments set file""")
    parser.add_argument(
        "--recordings-file", default=None, help="""recordings set file"""
    )
    parser.add_argument(
        "--class-names", nargs="+", default=None, help="""class names"""
    )

    add_common_args(parser)
    return parser


def from_segments(
    dataset: PathLike,
    segments_file: PathLike,
    recordings_file: Optional[PathLike] = None,
    class_names: Optional[List[str]] = None,
) -> None:
    """Create a dataset from a segment table.

    Args:
        dataset: Output dataset directory or YAML file.
        segments_file: SegmentSet file.
        recordings_file: Optional RecordingSet file.
        class_names: Segment columns to convert into class tables.
    """
    output_dataset = dataset
    logging.info("making dataset %s from segments %s", dataset, segments_file)
    dataset = HyperDataset.from_segments(segments_file, recordings_file, class_names)
    dataset.save(output_dataset)


def make_add_classes_from_segments_parser() -> ArgumentParser:
    """Create parser for the ``add_classes_from_segments`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--class-names",
        required=True,
        nargs="+",
        help="""segment columns to convert into class tables""",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )

    add_common_args(parser)
    return parser


def add_classes_from_segments(
    dataset: PathLike,
    class_names: List[str],
    output_dataset: Optional[PathLike],
) -> None:
    """Add class tables derived from segment columns.

    Args:
        dataset: Input dataset directory or YAML file.
        class_names: Segment columns used to build class tables.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
    """
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "adding class info %s from segments: %s -> %s",
        str(class_names),
        dataset,
        output_dataset,
    )

    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.add_classes_from_segments(class_names)
    dataset.save(output_dataset)


def make_remove_short_segments_parser() -> ArgumentParser:
    """Create parser for the ``remove_short_segments`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
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
    output_dataset: Optional[PathLike],
) -> None:
    """Remove segments shorter than a threshold.

    Args:
        dataset: Input dataset directory or YAML file.
        min_length: Minimum allowed length.
        length_name: Segment column with duration values.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
    """
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "removing segments with %s<%f in dataset: %s -> %s",
        length_name,
        min_length,
        dataset,
        output_dataset,
    )
    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.remove_short_segments(min_length, length_name)
    dataset.save(output_dataset)


def make_rebuild_class_idx_parser() -> ArgumentParser:
    """Create parser for the ``rebuild_class_idx`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
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
    output_dataset: Optional[PathLike],
) -> None:
    """Rebuild contiguous class ids for a class table.

    Args:
        dataset: Input dataset directory or YAML file.
        class_name: Class table name, for example ``speaker``.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
    """
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "rebuilding %s class index in dataset: %s -> %s",
        class_name,
        dataset,
        output_dataset,
    )

    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.rebuild_class_idx(class_name)
    dataset.save(output_dataset)


def make_remove_classes_few_segments_parser() -> ArgumentParser:
    """Create parser for the ``remove_classes_few_segments`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
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
    output_dataset: Optional[PathLike],
) -> None:
    """Remove classes with fewer than ``min_segs`` segments.

    Args:
        dataset: Input dataset directory or YAML file.
        class_name: Class table name, for example ``speaker``.
        min_segs: Minimum number of segments required per class.
        rebuild_idx: Whether to rebuild class ids after filtering.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
    """
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "removing %s with segments<%d in dataset: %s -> %s",
        class_name,
        min_segs,
        dataset,
        output_dataset,
    )

    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.remove_classes_few_segments(class_name, min_segs, rebuild_idx)
    dataset.save(output_dataset)


def make_remove_classes_few_toomany_segments_parser() -> ArgumentParser:
    """Create parser for ``remove_classes_few_toomany_segments``."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
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
    max_segs: Optional[int],
    rebuild_idx: bool,
    output_dataset: Optional[PathLike],
) -> None:
    """Remove classes outside a min/max segment count range.

    Args:
        dataset: Input dataset directory or YAML file.
        class_name: Class table name, for example ``speaker``.
        min_segs: Minimum number of segments required per class.
        max_segs: Maximum number of segments allowed per class.
        rebuild_idx: Whether to rebuild class ids after filtering.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
    """
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
    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.remove_classes_few_toomany_segments(
        class_name, min_segs, max_segs, rebuild_idx
    )
    dataset.save(output_dataset)


def make_remove_class_ids_parser() -> ArgumentParser:
    """Create parser for the ``remove_class_ids`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
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
        help="Remove segments with NA class ids.",
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
    class_ids: Optional[List[str]],
    remove_na: bool,
    rebuild_idx: bool,
    output_dataset: Optional[PathLike],
) -> None:
    """Remove specific class ids from a class table.

    Args:
        dataset: Input dataset directory or YAML file.
        class_name: Class table name, for example ``speaker``.
        class_ids: Class ids to remove.
        remove_na: Whether to remove entries with NA class id.
        rebuild_idx: Whether to rebuild class ids after filtering.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
    """
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "removing %s with ids %s in dataset: %s -> %s",
        class_name,
        str(class_ids),
        dataset,
        output_dataset,
    )

    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.remove_class_ids(class_name, class_ids, remove_na, rebuild_idx)
    dataset.save(output_dataset)


def make_split_train_val_parser() -> ArgumentParser:
    """Create parser for the ``split_train_val`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
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
    joint_classes: Optional[List[str]],
    disjoint_classes: Optional[List[str]],
    min_train_samples: int,
    seed: int,
    train_dataset: PathLike,
    val_dataset: PathLike,
) -> None:
    """Split a dataset into train and validation sets.

    Args:
        dataset: Input dataset directory or YAML file.
        val_prob: Fraction of segments assigned to validation.
        joint_classes: Class types constrained to overlap across splits.
        disjoint_classes: Class types constrained to be disjoint across splits.
        min_train_samples: Minimum train samples per class.
        seed: Random seed.
        train_dataset: Output path for train dataset.
        val_dataset: Output path for validation dataset.
    """
    logging.info(
        "splitting %s -> train: %s + val: %s",
        dataset,
        train_dataset,
        val_dataset,
    )
    dataset = HyperDataset.load(dataset, lazy=True)
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


def make_split_folds_parser() -> ArgumentParser:
    """Create parser for the ``split_folds`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
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
    joint_classes: Optional[List[str]],
    disjoint_classes: Optional[List[str]],
    seed: int,
    output_path: PathLike,
) -> None:
    """Split a dataset into cross-validation train/test folds.

    Args:
        dataset: Input dataset directory or YAML file.
        num_folds: Number of folds.
        joint_classes: Class types constrained to overlap across train/test.
        disjoint_classes: Class types constrained to be disjoint across train/test.
        seed: Random seed.
        output_path: Output base directory for generated folds.
    """
    logging.info(
        "splitting %s -> %s",
        dataset,
        output_path,
    )

    dataset = HyperDataset.load(dataset, lazy=True)
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


def make_filter_by_segments_parser() -> ArgumentParser:
    """Create parser for the ``filter_by_segments`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--segments-file",
        required=True,
        help="""name of the file containing the segments to keep or remove""",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )
    parser.add_argument(
        "--rebuild-class-idx",
        default=False,
        action=ActionYesNo,
        help="""regenerate classes indexes from 0 to new_num_classes-1""",
    )
    parser.add_argument(
        "--keep",
        default=True,
        action=ActionYesNo,
        help="""whether keep or remove the segments""",
    )

    add_common_args(parser)
    return parser


def filter_by_segments(
    dataset: PathLike,
    segments_file: PathLike,
    output_dataset: Optional[PathLike],
    rebuild_class_idx: bool = False,
    keep: bool = True,
) -> None:
    """Filter dataset entries using a segment id list.

    Args:
        dataset: Input dataset directory or YAML file.
        segments_file: SegmentSet file with segment ids to keep/remove.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
        rebuild_class_idx: Whether to rebuild class ids after filtering.
        keep: If ``True``, keep listed segments; otherwise remove them.
    """
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "Dataset %s filtering segments in %s -> %s",
        dataset,
        segments_file,
        output_dataset,
    )
    segments = SegmentSet.load(segments_file)
    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.filter_by_segments(segments, rebuild_class_idx=rebuild_class_idx, keep=keep)
    dataset.save(output_dataset)


def make_filter_by_segments_predicate_parser() -> ArgumentParser:
    """Create parser for ``filter_by_segments_predicate``."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--predicate",
        required=True,
        help="""predicate to use for filtering""",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )
    parser.add_argument(
        "--rebuild-class-idx",
        default=False,
        action=ActionYesNo,
        help="""regenerate classes indexes from 0 to new_num_classes-1""",
    )
    parser.add_argument(
        "--keep",
        default=True,
        action=ActionYesNo,
        help="""whether keep or remove the segments""",
    )

    add_common_args(parser)
    return parser


def filter_by_segments_predicate(
    dataset: PathLike,
    predicate: str,
    output_dataset: Optional[PathLike],
    rebuild_class_idx: bool = False,
    keep: bool = True,
) -> None:
    """Filter dataset entries using a predicate over segment columns.

    Args:
        dataset: Input dataset directory or YAML file.
        predicate: Predicate expression evaluated on the segment table.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
        rebuild_class_idx: Whether to rebuild class ids after filtering.
        keep: If ``True``, keep matching segments; otherwise remove them.
    """
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "Dataset %s filtering segments by predicate %s -> %s",
        dataset,
        predicate,
        output_dataset,
    )
    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.filter_by_segments_predicate(
        predicate, rebuild_class_idx=rebuild_class_idx, keep=keep
    )
    dataset.save(output_dataset)


def make_filter_by_classes_parser() -> ArgumentParser:
    """Create parser for the ``filter_by_classes`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
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
        "--remove-na",
        default=False,
        action=ActionYesNo,
        help="Remove segments with NA class ids.",
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
    output_dataset: Optional[PathLike],
    remove_na: bool = False,
    rebuild_idx: bool = False,
    keep: bool = True,
) -> None:
    """Filter dataset entries by class ids loaded from file.

    Args:
        dataset: Input dataset directory or YAML file.
        class_name: Class table name, for example ``speaker``.
        class_file: ClassInfo file with ids to keep/remove.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
        remove_na: Whether to remove entries with NA class id.
        rebuild_idx: Whether to rebuild class ids after filtering.
        keep: If ``True``, keep listed classes; otherwise remove them.
    """
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
    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.filter_by_classes(class_name, classes, rebuild_idx=rebuild_idx, keep=keep)
    dataset.save(output_dataset)


def make_filter_by_classes_and_enrollments_parser() -> ArgumentParser:
    """Create parser for ``filter_by_classes_and_enrollments``."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
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
        "--remove-na",
        default=False,
        action=ActionYesNo,
        help="Remove segments with NA class ids.",
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
    output_dataset: Optional[PathLike],
    remove_na: bool = False,
    rebuild_idx: bool = False,
    keep: bool = True,
) -> None:
    """Filter dataset by classes and enrollment ids jointly.

    Args:
        dataset: Input dataset directory or YAML file.
        class_name: Class table name, for example ``speaker``.
        class_file: ClassInfo file with ids to keep/remove.
        enrollment_name: Enrollment map name in dataset metadata.
        enrollment_file: EnrollmentMap file with ids to keep/remove.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
        remove_na: Whether to remove entries with NA class id.
        rebuild_idx: Whether to rebuild class ids after filtering.
        keep: If ``True``, keep listed ids; otherwise remove them.
    """
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
    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.filter_by_classes_and_enrollments(
        class_name,
        classes,
        enrollment_name,
        enrollments,
        remove_na=remove_na,
        rebuild_idx=rebuild_idx,
        keep=keep,
    )
    dataset.save(output_dataset)


def make_copy_parser() -> ArgumentParser:
    """Create parser for the ``copy`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--output-dataset",
        required=True,
        help="""output dataset dir""",
    )
    parser.add_argument(
        "--seg-suffix",
        default=None,
        help="Append suffix to segment ids.",
    )

    add_common_args(parser)
    return parser


def copy(
    dataset: PathLike,
    output_dataset: PathLike,
    seg_suffix: Optional[str] = None,
) -> None:
    """Copy a dataset, optionally appending a suffix to segment ids.

    Args:
        dataset: Input dataset directory or YAML file.
        output_dataset: Output dataset directory or YAML file.
        seg_suffix: Optional suffix appended to segment ids.
    """
    logging.info(
        "copying dataset: %s -> %s",
        dataset,
        output_dataset,
    )
    dataset = HyperDataset.load(dataset, lazy=True)
    if seg_suffix is not None:
        dataset.append_seg_suffix(seg_suffix)
    dataset.save(output_dataset)


def make_clean_parser() -> ArgumentParser:
    """Create parser for the ``clean`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )
    parser.add_argument(
        "--rebuild-class-idx",
        default=False,
        action=ActionYesNo,
        help="""regenerate class indexes from 0 to new_num_classes-1""",
    )

    add_common_args(parser)
    return parser


def clean(
    dataset: PathLike,
    output_dataset: Optional[PathLike],
    rebuild_class_idx: bool,
) -> None:
    """Run dataset consistency cleanup.

    Args:
        dataset: Input dataset directory or YAML file.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
        rebuild_class_idx: Whether to rebuild class ids after cleanup.
    """
    if output_dataset is None:
        output_dataset = dataset
    logging.info(
        "cleaning dataset: %s -> %s",
        dataset,
        output_dataset,
    )
    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.clean(rebuild_class_idx=rebuild_class_idx)
    dataset.save(output_dataset)


def make_sample_random_subsegments_parser() -> ArgumentParser:
    """Create parser for the ``sample_random_subsegments`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )
    parser.add_argument(
        "--subsegments-per-segment",
        default=1,
        type=int,
        help="number of subsegments per segment",
    )
    parser.add_argument(
        "--min-duration", default=0.0, type=float, help="min. segment duration"
    )
    parser.add_argument(
        "--max-duration", default=None, type=float, help="max. segment duration"
    )
    parser.add_argument(
        "--random-start",
        default=True,
        action=ActionYesNo,
        help="the starting point of the subsegment is random or the start of the segment",
    )
    parser.add_argument(
        "--seg-suffix",
        default=None,
        help="Append suffix to segment ids.",
    )
    parser.add_argument("--seed", default=11235813, type=int, help="random seed")

    add_common_args(parser)
    return parser


def sample_random_subsegments(
    dataset: PathLike,
    output_dataset: Optional[PathLike],
    subsegments_per_segment: int = 1,
    min_duration: float = 0.0,
    max_duration: Optional[float] = None,
    seg_suffix: Optional[str] = None,
    random_start: bool = True,
    seed: int = 11235813,
) -> None:
    """Sample random subsegments from existing segments.

    Args:
        dataset: Input dataset directory or YAML file.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
        subsegments_per_segment: Number of subsegments sampled per segment.
        min_duration: Minimum subsegment duration in seconds.
        max_duration: Maximum subsegment duration in seconds.
        seg_suffix: Optional suffix appended to generated segment ids.
        random_start: Whether to sample random start offsets.
        seed: Random seed.
    """
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "generate new dataset with random subsegments: %s -> %s",
        dataset,
        output_dataset,
    )
    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.sample_random_subsegments(
        subsegments_per_segment=subsegments_per_segment,
        min_duration=min_duration,
        max_duration=max_duration,
        random_start=random_start,
        seg_suffix=seg_suffix,
        seed=seed,
        inplace=True,
    )
    dataset.save(output_dataset)


def make_cat_segments_parser() -> ArgumentParser:
    """Create parser for the ``cat_segments`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--group-by",
        required=True,
        nargs="+",
        help="columns used to group segments for concatenation",
    )
    parser.add_argument(
        "--max-duration",
        default=None,
        type=float,
        help="max duration in seconds for each concatenated segment",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )

    add_common_args(parser)
    return parser


def cat_segments(
    dataset: PathLike,
    group_by: List[str],
    output_dataset: Optional[PathLike],
    max_duration: Optional[float] = None,
) -> None:
    """Concatenate segments after grouping by one or more columns.

    Args:
        dataset: Input dataset directory or YAML file.
        group_by: Columns used to group segments before concatenation.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
        max_duration: Optional maximum duration per concatenated segment.
    """
    if output_dataset is None:
        output_dataset = dataset

    logging.info(
        "concatenating segments grouped by %s in dataset: %s -> %s",
        group_by,
        dataset,
        output_dataset,
    )
    dataset = HyperDataset.load(dataset, lazy=True)
    dataset = dataset.cat_segments(group_by=group_by, max_duration=max_duration)
    dataset.save(output_dataset)


def make_add_cols_to_segments_parser() -> ArgumentParser:
    """Create parser for the ``add_cols_to_segments`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
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
        help="Create class-info tables for newly added columns.",
    )

    add_common_args(parser)
    return parser


def add_cols_to_segments(
    dataset: PathLike,
    right_table: PathLike,
    column_names: List[str],
    on: List[str],
    right_on: Optional[List[str]],
    output_dataset: Optional[PathLike],
    remove_missing: bool = False,
    create_class_info: bool = False,
) -> None:
    """Add columns from an external table into the segments table.

    Args:
        dataset: Input dataset directory or YAML file.
        right_table: Table containing columns to merge.
        column_names: Column names copied from ``right_table``.
        on: Key columns in the segments table.
        right_on: Key columns in ``right_table``. Defaults to ``on``.
        output_dataset: Output dataset path. If ``None``, overwrite input dataset.
        remove_missing: Remove entries with missing values after merge.
        create_class_info: Build class-info tables for new columns.
    """
    if output_dataset is None:
        output_dataset = dataset

    logging.info("adding columns to %s + %s -> %s", dataset, right_table, output_dataset)
    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.add_cols_to_segments(
        right_table,
        column_names,
        on,
        right_on,
        remove_missing=remove_missing,
        create_class_info=create_class_info,
    )
    dataset.save(output_dataset)


def make_merge_parser() -> ArgumentParser:
    """Create parser for the ``merge`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--input-datasets", required=True, nargs="+", help="input datasets"
    )
    add_common_args(parser)
    return parser


def merge(dataset: PathLike, input_datasets: List[PathLike]) -> None:
    """Merge multiple datasets into one.

    Args:
        dataset: Output dataset directory or YAML file.
        input_datasets: Input datasets to merge.
    """
    input_dataset_paths = input_datasets
    dataset_path = dataset

    logging.info("merging %s -> %s", (input_dataset_paths), dataset_path)
    input_datasets = []
    for dset_file in input_dataset_paths:
        input_datasets.append(HyperDataset.load(dset_file))

    dataset = HyperDataset.merge(input_datasets)
    dataset.save(dataset_path)


def make_from_lhotse_parser() -> ArgumentParser:
    """Create parser for the ``from_lhotse`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
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
) -> None:
    """Create a dataset from Lhotse manifests.

    Args:
        dataset: Output dataset directory or YAML file.
        cuts_file: Optional cuts manifest.
        recordings_file: Optional recordings manifest.
        supervisions_file: Optional supervisions manifest.
    """
    assert cuts_file is not None or supervisions_file is not None
    logging.info(
        "create dataset from lhotse: %s -> %s",
        cuts_file if cuts_file is not None else supervisions_file,
        dataset,
    )
    dataset_path = dataset
    dataset = HyperDataset.from_lhotse(
        cuts=cuts_file, recordings=recordings_file, supervisions=supervisions_file
    )
    dataset.save(dataset_path)


def make_from_kaldi_parser() -> ArgumentParser:
    """Create parser for the ``from_kaldi`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
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
) -> None:
    """Create a dataset from a Kaldi data directory.

    Args:
        dataset: Output dataset directory or YAML file.
        kaldi_data_dir: Kaldi data directory path.
    """
    logging.info("create dataset from kaldi: %s -> %s", kaldi_data_dir, dataset)
    dataset_path = dataset
    dataset = HyperDataset.from_kaldi(kaldi_data_dir)
    dataset.save(dataset_path)


def make_describe_parser() -> ArgumentParser:
    """Create parser for the ``describe`` subcommand."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    add_common_args(parser)
    return parser


def describe(
    dataset: PathLike,
) -> None:
    """Print dataset summary statistics.

    Args:
        dataset: Dataset directory or YAML file.
    """
    dataset = HyperDataset.load(dataset, lazy=True)
    dataset.describe()


def main() -> None:
    """Parse subcommand arguments and dispatch dataset operations."""
    parser = ArgumentParser(description="Tool to manipulate the Hyperion dataset")
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a configuration file."
    )

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
