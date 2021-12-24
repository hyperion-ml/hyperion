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
from hyperion.utils import Utt2Info


def split_train_test(attack_info_file, train_list, test_list, p_val, output_dir):

    rng = np.random.RandomState(seed=1234)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(attack_info_file, "r") as f:
        attack_info = yaml.load(f, Loader=yaml.FullLoader)

    benign_to_attack = {}
    for k, v in attack_info.items():
        bk = v["key_original"]
        if bk in benign_to_attack:
            benign_to_attack[bk].append(k)
        else:
            benign_to_attack[bk] = [k]

    train_utts = Utt2Info.load(train_list)
    train_val_keys = train_utts.key
    # split in train and val

    train_info = {}
    val_info = {}

    for k in train_val_keys:
        if not (k in benign_to_attack):
            continue

        attacks_k = {}
        for ak in benign_to_attack[k]:
            attacks_k[ak] = attack_info[ak]

        p = rng.rand(1)
        if p < p_val:
            val_info.update(attacks_k)
        else:
            train_info.update(attacks_k)

    test_utts = Utt2Info.load(test_list)
    test_val_keys = test_utts.key
    # split in test and val

    test_info = {}
    for k in test_val_keys:
        if not (k in benign_to_attack):
            continue
        for ak in benign_to_attack[k]:
            test_info[ak] = attack_info[ak]

    with open(output_dir / "train_attack_info.yaml", "w") as f:
        yaml.dump(train_info, f, sort_keys=True)

    with open(output_dir / "val_attack_info.yaml", "w") as f:
        yaml.dump(val_info, f, sort_keys=True)

    with open(output_dir / "test_attack_info.yaml", "w") as f:
        yaml.dump(test_info, f, sort_keys=True)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Split Yaml attacks info file into train/val/test"
    )

    parser.add_argument("--attack-info-file", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--p-val", default=0.1, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    split_train_test(**namespace_to_dict(args))
