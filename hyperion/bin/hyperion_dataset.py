#!/usr/bin/env python
"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from typing import Optional, Union, List
from pathlib import Path

from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.utils import (
    PathLike,
    Dataset,
    InfoTable,
    RecordingSet,
    FeatureSet,
    ClassInfo,
    EnrollmentMap,
    SegmentSet,
)

subcommands = ["add_features"]
# table_dict = {
#     "segments": SegmentSet,
#     "recordings": RecordingSet,
#     "features": FeatureSet,
#     "classes": ClassInfo,
#     "enrollments": EnrollmentMap,
#     "generic": InfoTable,
# }


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

    add_common_args(parser)
    return parser


def add_features(
    dataset: PathLike,
    features_name: str,
    features_file: PathLike,
):
    dataset = Dataset.load(dataset, lazy=True)
    dataset.add_features(features_name, features_file)
    dataset.save(dataset)


if __name__ == "__main__":

    parser = ArgumentParser(description="Tool to manipulates the Hyperion dataset")
    parser.add_argument("--cfg", action=ActionConfigFile)

    subcommands = parser.add_subcommands()
    for subcommand in subcommands:
        parser_func = f"make_{subcommand}_parser"
        subparser = globals()[parser_func]()
        subcommands.add_subcommand(k, subparser)

    args = parser.parse_args()
    subcommand = args.subcommand
    kwargs = namespace_to_dict(args)[args.subcommand]
    config_logger(kwargs["verbose"])
    del kwargs["verbose"]

    globals()[subcommand](**kwargs)
