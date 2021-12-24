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
from hyperion.utils import Utt2Info, SCPList


def make_lists(input_dir, benign_wav_file, benign_durs, output_dir):

    rng = np.random.RandomState(seed=1234)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_dir / "train_attack_info.yml", "r") as f:
        train_attacks = yaml.load(f, Loader=yaml.FullLoader)

    with open(input_dir / "val_attack_info.yml", "r") as f:
        val_attacks = yaml.load(f, Loader=yaml.FullLoader)

    with open(input_dir / "test_attack_info.yml", "r") as f:
        test_attacks = yaml.load(f, Loader=yaml.FullLoader)

    k2w = SCPList.load(benign_wav_file)
    u2d = Utt2Info.load(benign_durs)

    keys = []
    files = []
    classes = []
    benign_keys = []
    durs = []
    for k, v in train_attacks.items():
        keys.append(k)
        files.append(v["wav_path"])
        classes.append(v["threat_model"])
        benign_keys.append(v["benign_key"])
        durs.append(v["num_frames"] / 16000)

    benign_keys = np.unique(benign_keys)
    for k in benign_keys:
        keys.append(k)
        classes.append("benign")
        files.append(k2w[k][0])
        durs.append(u2d[k])

    train_u2c = Utt2Info.create(keys, classes)
    train_u2d = Utt2Info.create(keys, durs)
    train_wav = SCPList(keys, files)
    uclasses = np.unique(classes)

    #######

    keys = []
    files = []
    classes = []
    benign_keys = []
    durs = []
    for k, v in val_attacks.items():
        keys.append(k)
        files.append(v["wav_path"])
        classes.append(v["threat_model"])
        benign_keys.append(v["benign_key"])
        durs.append(v["num_frames"] / 16000)

    benign_keys = np.unique(benign_keys)
    for k in benign_keys:
        keys.append(k)
        classes.append("benign")
        files.append(k2w[k][0])
        durs.append(u2d[k])

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
        keys.append(k)
        files.append(v["wav_path"])
        classes.append(v["threat_model"])
        benign_keys.append(v["benign_key"])
        durs.append(v["num_frames"] / 16000)

    benign_keys = np.unique(benign_keys)
    for k in benign_keys:
        keys.append(k)
        classes.append("benign")
        files.append(k2w[k][0])
        durs.append(u2d[k])

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

    with open(output_dir / "class2int", "w") as f:
        for c in uclasses:
            f.write("%s\n" % (c))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="prepare lists to train nnet to discriminate between attacks types and benign speech",
    )

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--benign-wav-file", required=True)
    parser.add_argument("--benign-durs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    make_lists(**vars(args))
