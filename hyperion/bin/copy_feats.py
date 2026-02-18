#!/usr/bin/env python
"""
 Copyright 2018 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
 Copy features/vectors and change format
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

from hyperion.hyp_defs import config_logger
from hyperion.io import CopyFeats as CF


def main() -> None:
    """Parse CLI arguments and copy features/vectors to a new destination/format."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Copy features and change format",
    )

    parser.add_argument(
        "--input",
        dest="input_spec",
        nargs="+",
        required=True,
        help="input feature rspecifier(s)",
    )
    parser.add_argument(
        "--output",
        dest="output_spec",
        required=True,
        help="output feature wspecifier/path",
    )
    parser.add_argument(
        "--write-num-frames",
        dest="write_num_frames",
        default=None,
        help="optional output file to write number of frames per utterance",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="verbosity level (0=error, 1=warning, 2=info, 3=debug)",
    )

    CF.add_argparse_args(parser)
    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    CF(**vars(args))


if __name__ == "__main__":
    main()
