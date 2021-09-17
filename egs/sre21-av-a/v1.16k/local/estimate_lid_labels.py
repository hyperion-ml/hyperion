#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""

import os
import logging
from jsonargparse import ArgumentParser, namespace_to_dict

import math
import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils import Utt2Info
from hyperion.io import RandomAccessDataReaderFactory as DRF


def estimate_lid_labels(list_file, logits_file, class_file, output_file, sre21):

    logging.info("Converting logits to labels for %s", list_file)
    utts = Utt2Info.load(list_file)
    reader = DRF.create(logits_file)
    classes = []
    with open(class_file, "r") as f:
        for line in f:
            classes.append(line.strip())

    x = reader.read(utts.key, squeeze=True)

    if sre21:
        sre21_langs = ["ENG", "CMN", "YUE"]
        log_priors = -1000 * np.ones((len(classes),), dtype=float_cpu())
        for c in sre21_langs:
            log_priors[classes.index(c)] = -math.log(len(sre21_langs))

        x += log_priors

    class_idx = np.argmax(x, axis=1)
    with open(output_file, "w") as f:
        for i, k in enumerate(utts.key):
            f.write("%s %s\n" % (k, classes[class_idx[i]]))


if __name__ == "__main__":

    parser = ArgumentParser(description="Transform xvector logits into labels")

    parser.add_argument("--list-file", required=True)
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--class-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument(
        "--sre21",
        default=False,
        action="store_true",
        help="If SRE21 only ENG/CMN/YUE are allowed",
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    estimate_lid_labels(**namespace_to_dict(args))
