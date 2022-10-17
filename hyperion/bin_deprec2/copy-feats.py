#!/usr/bin/env python
"""
 Copyright 2018 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
 Copy features/vectors and change format
"""

import sys
import os
import argparse
import time
import logging

import numpy as np

from hyperion.hyp_defs import config_logger
from hyperion.io import CopyFeats as CF


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Copy features and change format",
    )

    parser.add_argument("--input", dest="input_spec", nargs="+", required=True)
    parser.add_argument("--output", dest="output_spec", required=True)
    parser.add_argument("--write-num-frames", dest="write_num_frames", default=None)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    CF.add_argparse_args(parser)
    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    CF(**vars(args))
