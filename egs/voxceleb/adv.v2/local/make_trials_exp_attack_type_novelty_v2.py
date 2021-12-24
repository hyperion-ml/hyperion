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
from hyperion.utils import Utt2Info, SCPList, TrialKey


def make_lists(
    input_dir, known_attacks, min_snr, max_snr, success_category, output_dir
):

    rng = np.random.RandomState(seed=1234)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_dir / "test_attack_info.yaml", "r") as f:
        test_attacks = yaml.load(f, Loader=yaml.FullLoader)

    keys = []
    files = []
    classes = []
    for k, v in test_attacks.items():
        s = v["success"]
        if not (
            success_category == "both"
            or success_category == "success"
            and s
            or success_category == "fail"
            and not s
        ):
            continue
        snr = v["snr"]
        if snr < min_snr or snr > max_snr:
            continue

        keys.append(k)
        files.append(v["wav_path"])
        classes.append(v["attack_type"])
        keys.append(v["key_benign"])
        files.append(v["wav_benign"])
        classes.append("benign")

    u2c = Utt2Info.create(keys, classes)
    wav = SCPList(keys, files)

    #####
    u2c.save(output_dir / "utt2attack")
    wav.save(output_dir / "wav.scp")

    with open(output_dir / "trials", "w") as f:
        for i in range(len(u2c)):
            k = u2c.key[i]
            att = u2c.info[i]
            if att in known_attacks:
                f.write("known %s nontarget\n" % k)
            else:
                f.write("known %s target\n" % k)

    with open(output_dir / "trials_nobenign", "w") as f:
        for i in range(len(u2c)):
            k = u2c.key[i]
            att = u2c.info[i]
            if att in ["benign"]:
                continue
            if att in known_attacks:
                f.write("known %s nontarget\n" % k)
            else:
                f.write("known %s target\n" % k)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="prepare trial list to do attack type novelty det"
    )

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--known-attacks", required=True, nargs="+")
    # parser.add_argument('--benign-wav-file', required=True)
    parser.add_argument(
        "--success-category", default="success", choices=["success", "fail", "both"]
    )
    parser.add_argument("--min-snr", default=-10, type=float)
    parser.add_argument("--max-snr", default=100, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    make_lists(**vars(args))
