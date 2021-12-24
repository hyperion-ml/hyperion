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
    input_dir,
    output_dir,
    train_min_snr,
    train_max_snr,
    train_success_category,
    test_min_snr,
    test_max_snr,
    test_success_category,
):

    rng = np.random.RandomState(seed=1234)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_dir / "train_attack_info.yaml", "r") as f:
        train_attacks = yaml.load(f, Loader=yaml.FullLoader)

    with open(input_dir / "val_attack_info.yaml", "r") as f:
        val_attacks = yaml.load(f, Loader=yaml.FullLoader)

    with open(input_dir / "test_attack_info.yaml", "r") as f:
        test_attacks = yaml.load(f, Loader=yaml.FullLoader)

    keys = []
    files = []
    classes = []
    durs = []
    for k, v in train_attacks.items():
        s = v["success"]
        if not (
            train_success_category == "both"
            or train_success_category == "success"
            and s
            or train_success_category == "fail"
            and not s
        ):
            continue
        snr = v["snr"]
        if snr < train_min_snr or snr > train_max_snr:
            continue

        keys.append(k)
        files.append(v["wav_path"])
        classes.append(v["threat_model"])
        durs.append(v["num_samples"] / 16000)
        keys.append(v["key_benign"])
        files.append(v["wav_benign"])
        classes.append("benign")
        durs.append(v["num_samples"] / 16000)

    train_u2c = Utt2Info.create(keys, classes)
    train_u2d = Utt2Info.create(keys, durs)
    train_wav = SCPList(keys, files)
    uclasses = np.unique(classes)

    #######

    keys = []
    files = []
    classes = []
    durs = []
    for k, v in val_attacks.items():
        s = v["success"]
        if not (
            train_success_category == "both"
            or train_success_category == "success"
            and s
            or train_success_category == "fail"
            and not s
        ):
            continue
        snr = v["snr"]
        if snr < train_min_snr or snr > train_max_snr:
            continue

        keys.append(k)
        files.append(v["wav_path"])
        classes.append(v["threat_model"])
        durs.append(v["num_samples"] / 16000)
        keys.append(v["key_benign"])
        files.append(v["wav_benign"])
        classes.append("benign")
        durs.append(v["num_samples"] / 16000)

    val_u2c = Utt2Info.create(keys, classes)
    val_u2d = Utt2Info.create(keys, durs)
    val_wav = SCPList(keys, files)

    #######

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
        classes.append(v["threat_model"])
        durs.append(v["num_samples"] / 16000)
        keys.append(v["key_benign"])
        files.append(v["wav_benign"])
        classes.append("benign")
        durs.append(v["num_samples"] / 16000)

    test_u2c = Utt2Info.create(keys, classes)
    test_u2d = Utt2Info.create(keys, durs)
    test_wav = SCPList(keys, files)

    #####
    trainval_wav = SCPList.merge([train_wav, val_wav])
    trainval_u2d = Utt2Info.merge([train_u2d, val_u2d])

    #####
    train_u2c.save(output_dir / "train_utt2attack")
    val_u2c.save(output_dir / "val_utt2attack")
    test_u2c.save(output_dir / "test_utt2attack")

    train_wav.save(output_dir / "train_wav.scp")
    val_wav.save(output_dir / "val_wav.scp")
    trainval_wav.save(output_dir / "trainval_wav.scp")
    test_wav.save(output_dir / "test_wav.scp")

    train_u2d.save(output_dir / "train_utt2dur")
    val_u2d.save(output_dir / "val_utt2dur")
    trainval_u2d.save(output_dir / "trainval_utt2dur")
    test_u2d.save(output_dir / "test_utt2dur")

    with open(output_dir / "class_file", "w") as f:
        for c in uclasses:
            f.write("%s\n" % (c))


if __name__ == "__main__":

    parser = ArgumentParser(
        description="prepare lists to train nnet to discriminate between attacks types and benign speech"
    )

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-min-snr", default=-10, type=float)
    parser.add_argument("--train-max-snr", default=100, type=float)
    parser.add_argument(
        "--train-success-category",
        default="success",
        choices=["success", "fail", "both"],
    )
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

    make_lists(**namespace_to_dict(args))
