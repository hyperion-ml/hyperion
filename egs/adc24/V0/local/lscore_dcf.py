#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""

import sys
import os
import argparse
import logging
from statistics import mean

import numpy as np
import jiwer

from hyperion.hyp_defs import config_logger
from hyperion.utils import SparseTrialKey, SparseTrialScores


def compute_wer(refs, hyps):
    wer_list = []
    for ref, hyp in zip(refs, hyps):
        transformation = jiwer.Compose([
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemoveEmptyStrings()
        ])
        wer = jiwer.wer(ref, hyp, truth_transform=transformation, hypothesis_transform=transformation)
        wer_list.append(wer)
    return wer_list


def score_wer(key_file, score_file, output_path):
    logging.info("Load key: %s" % key_file)
    key = SparseTrialKey.load_txt(key_file)
    logging.info("Load scores: %s" % score_file)
    scr = SparseTrialScores.load_txt(score_file)

    # Assuming scr.get_tar_non returns reference and hypothesis pairs
    refs, hyps = scr.get_tar_non(key)

    logging.info("computing WER")
    wer_list = compute_wer(refs, hyps)
    avg_wer = mean(wer_list)
    mr_wer = mean([1 / w if w != 0 else 1 for w in wer_list])  # Handling division by zero

    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file = output_path + "_results"
    with open(output_file, "w") as f:
        s = "WER: {0:.2f} MR-WER: {1:.2f}\n".format(avg_wer * 100, mr_wer)
        f.write(s)
        logging.info(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Computes WER and MR-WER",
    )

    parser.add_argument("--key-file", required=True)
    parser.add_argument("--score-file", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    score_wer(**vars(args))
