#!/usr/bin/env python
"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""
import sys
import os
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
)
import time
import logging

from pathlib import Path

import numpy as np
import yaml

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils import Utt2Info, SCPList


def make_lists(
    input_file,
    benign_wav_file,
    output_dir,
    test_min_snr,
    test_max_snr,
    test_success_category,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r") as f:
        test_attacks = yaml.load(f, Loader=yaml.FullLoader)

    k2w = SCPList.load(benign_wav_file)

    keys = []
    files = []
    classes = []
    benign_keys = []
    durs = []
    for k, v in test_attacks.items():
        s = v["success"]
        if not (
            test_success_category == "both"
            or test_success_category == "success"
            and s
            or test_success_category == "fail"
            and not s
        ):
            continue
        snr = v["snr"]
        if snr < test_min_snr or snr > test_max_snr:
            continue

        keys.append(k)
        files.append(v["wav_path"])
        classes.append(v["attack_type"])
        benign_keys.append(v["test_benign"])

    benign_keys = np.unique(benign_keys)
    for k in benign_keys:
        keys.append(k)
        classes.append("benign")
        files.append(k2w[k][0])

    test_u2c = Utt2Info.create(keys, classes)
    test_wav = SCPList(keys, files)

    test_u2c.save(output_dir / "utt2attack")
    test_wav.save(output_dir / "wav.scp")


if __name__ == "__main__":

    parser = ArgumentParser(description="prepare lists to test attack classification")

    parser.add_argument("--input-file", required=True)
    parser.add_argument("--benign-wav-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-min-snr", default=-10, type=float)
    parser.add_argument("--test-max-snr", default=100, type=float)
    parser.add_argument(
        "--test-success-category",
        default="success",
        choices=["success", "fail", "both"],
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    make_lists(**vars(args))
