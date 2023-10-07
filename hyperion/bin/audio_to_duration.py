#!/usr/bin/env python
"""
 Copyright 2022 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import logging
import os
import sys
import time

import numpy as np
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.io import SequentialAudioReader as AR
from hyperion.utils import SegmentSet


def audio_to_duration(audio_file, output_file, **kwargs):
    input_args = AR.filter_args(**kwargs)
    logging.info(f"input_args={input_args}")

    keys = []
    durations = []
    with AR(audio_file, **input_args) as reader:
        for data in reader:
            key, x, fs = data
            duration = x.shape[0] / fs
            keys.append(key)
            durations.append(duration)
            logging.info("read audio %s duration=%.3f", key, duration)

    print(len(keys), len(durations))
    seg_set = SegmentSet.from_lists(keys, ["duration"], [durations])
    seg_set.save(output_file)


def main():
    parser = ArgumentParser(description="Writes audio file durations to table")

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--audio-file", required=True)
    parser.add_argument("--output-file", required=True)
    AR.add_class_args(parser)
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

    audio_to_duration(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
