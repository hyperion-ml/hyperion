#!/usr/bin/env python
"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
import pandas
import os
import shutil
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import librosa
import soundfile as sf
import scipy.io.wavfile as wavf

from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)
from hyperion.torch.data import AudioDataset
from hyperion.hyp_defs import config_logger
from hyperion.utils import (
    ClassInfo,
    EnrollmentMap,
    FeatureSet,
    HypDataset,
    InfoTable,
    PathLike,
    RecordingSet,
    SegmentSet
)

from hyperion.bin.snr import get_volume, normalize_to_db

subcommand_list = [
    "add_features"
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

    dataset = HypDataset.load(dataset, lazy=True)
    dataset.add_features(features_name, features_file)
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
