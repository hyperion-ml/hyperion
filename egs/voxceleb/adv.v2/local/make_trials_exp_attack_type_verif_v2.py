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
    input_dir,
    known_attacks,
    output_dir,
    min_snr,
    max_snr,
    success_category,
    max_trials,
    num_enroll_sides,
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

    mask = rng.rand(len(u2c)) > 1 / (num_enroll_sides + 1)
    enr_key = u2c.key[mask]
    test_key = u2c.key[mask == False]
    enr_u2c = u2c.filter(enr_key)
    test_u2c = u2c.filter(test_key)

    if num_enroll_sides > 1:
        class_uniq, class_ids = np.unique(test_u2c.info, return_inverse=True)
        num_classes = len(class_uniq)
        count_sides = np.zeros((num_classes,), dtype=np.int)
        count_models = np.zeros((num_classes,), dtype=np.int)
        enroll_models = []
        for i in range(len(test_u2c)):
            j = class_ids[i]
            side = count_sides[j] % num_enroll_sides
            if side == 0:
                count_models[j] += 1
            enroll_model = "%s-%03d" % (class_uniq[j], count_models[j])
            enroll_models.append(enroll_model)
            count_sides[j] += 1

        enr_u2e = Utt2Info.create(test_u2c.key, enroll_models)
        enroll_models_uniq, idx = np.unique(enroll_models, return_index=True)
        enr_e2c = Utt2Info.create(enroll_models_uniq, test_u2c.info[idx])
    else:
        enr_e2c = enr_u2c
        enr_u2e = Utt2Info.create(enr_u2c.key, enr_u2c.key)

    trials_all = TrialKey(enr_e2c.key, test_u2c.key)
    trials_known = TrialKey(enr_e2c.key, test_u2c.key)
    trials_unknown = TrialKey(enr_e2c.key, test_u2c.key)
    for i in range(len(enr_e2c)):
        for j in range(len(test_u2c)):
            if enr_e2c.info[i] == test_u2c.info[j]:
                trials_all.tar[i, j] = True
                if enr_e2c.info[i] in known_attacks:
                    trials_known.tar[i, j] = True
                else:
                    trials_unknown.tar[i, j] = True
            else:
                trials_all.non[i, j] = True
                if (
                    enr_e2c.info[i] in known_attacks
                    and test_u2c.info[j] in known_attacks
                ):
                    trials_known.non[i, j] = True
                elif (
                    enr_e2c.info[i] not in known_attacks
                    and test_u2c.info[j] not in known_attacks
                ):
                    trials_unknown.non[i, j] = True

    max_trials = int(max_trials * 1e6)
    num_tar_trials = np.sum(trials_all.tar)
    num_non_trials = np.sum(trials_all.non)
    num_trials = num_tar_trials + num_non_trials
    if num_trials > max_trials:
        p = max_trials / num_trials
        logging.info("reducing number of trials (%d) with p=%f" % (num_trials, p))
        mask = rng.rand(*trials_all.tar.shape) > p
        trials_all.non[mask] = False
        trials_known.non[mask] = False
        trials_unknown.non[mask] = False
        trials_all.tar[mask] = False
        trials_known.tar[mask] = False
        trials_unknown.tar[mask] = False

    enr_u2e.sort(1)
    enr_u2e.save(output_dir / "utt2enr")
    trials_all.save_txt(output_dir / "trials")
    trials_known.save_txt(output_dir / "trials_known")
    trials_unknown.save_txt(output_dir / "trials_unknown")


if __name__ == "__main__":

    parser = ArgumentParser(
        description="prepare trial list to do attack type verification"
    )

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--known-attacks", required=True, nargs="+")
    # parser.add_argument('--benign-wav-file', required=True)
    # parser.add_argument('--benign-durs', required=True)
    parser.add_argument("--num-enroll-sides", default=1, type=int)
    parser.add_argument(
        "--max-trials",
        default=10,
        type=float,
        help="maximum number of trials in millions",
    )
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
