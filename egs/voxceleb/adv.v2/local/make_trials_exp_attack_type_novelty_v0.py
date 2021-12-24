#!/usr/bin/env python
"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""
import sys
import os
import argparse
import time
import logging

from pathlib import Path

import numpy as np
import yaml

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils import Utt2Info, SCPList, TrialKey


def make_lists(input_dir, seen_attacks, benign_wav_file, output_dir):

    rng = np.random.RandomState(seed=1234)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_dir / "test_attack_info.yml", "r") as f:
        test_attacks = yaml.load(f, Loader=yaml.FullLoader)

    k2w = SCPList.load(benign_wav_file)

    keys = []
    files = []
    classes = []
    benign_keys = []
    durs = []
    for k, v in test_attacks.items():
        keys.append(k)
        files.append(v["wav_path"])
        classes.append(v["attack_type"])
        benign_keys.append(v["benign_key"])

    benign_keys = np.unique(benign_keys)
    for k in benign_keys:
        keys.append(k)
        classes.append("benign")
        files.append(k2w[k][0])

    u2c = Utt2Info.create(keys, classes)
    # test_u2d = Utt2Info.create(keys, durs)
    wav = SCPList(keys, files)

    #####
    u2c.save(output_dir / "utt2attack")
    wav.save(output_dir / "wav.scp")

    with open(output_dir / "trials", "w") as f:
        for i in range(len(u2c)):
            k = u2c.key[i]
            att = u2c.info[i]
            if att in seen_attacks:
                f.write("seen %s nontarget\n" % k)
            else:
                f.write("seen %s target\n" % k)

    with open(output_dir / "trials_nobenign", "w") as f:
        for i in range(len(u2c)):
            k = u2c.key[i]
            att = u2c.info[i]
            if att in ["benign"]:
                continue
            if att in seen_attacks:
                f.write("seen %s nontarget\n" % k)
            else:
                f.write("seen %s target\n" % k)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="prepare trial list to do attack type novelty det",
    )

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--seen-attacks", required=True, nargs="+")
    parser.add_argument("--benign-wav-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    make_lists(**vars(args))
