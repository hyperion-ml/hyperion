#!/usr/bin/env python
"""
 Copyright 2022 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import logging
import os
import sys
import time
from typing import Any

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
from hyperion.utils.misc import PathLike


def audio_to_duration(audio_file: PathLike, output_file: PathLike, **kwargs: Any) -> None:
    """Compute utterance durations from audio and save them to a table.

    Args:
        audio_file: Input audio rspecifier/path consumed by ``SequentialAudioReader``.
        output_file: Output table path written via ``SegmentSet.save``.
        **kwargs: Additional audio-reader configuration parameters.
    """
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


def main() -> None:
    """Parse CLI arguments and write audio durations."""
    parser = ArgumentParser(description="Write audio durations to an output table")

    parser.add_argument("--cfg", action=ActionConfigFile, help="configuration file")
    parser.add_argument(
        "--audio-file",
        required=True,
        help="input audio rspecifier/path (e.g., wav.scp or audio archive)",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="output table path with one duration entry per utterance",
    )
    AR.add_class_args(parser)
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="verbosity level (0=error, 1=warning, 2=info, 3=debug)",
    )
    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    audio_to_duration(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
