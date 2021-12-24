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

snr_levels = np.arange(0, 65, 10)


def quant_snr(snr):
    q = np.argmin((snr_levels - snr) ** 2)
    q_str = "snr-%d" % (int(snr_levels[q]))
    return q_str


def make_lists(
    input_dir, seen_attacks, benign_wav_file, max_trials, num_enroll_sides, output_dir
):

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
    snrs = []
    for k, v in test_attacks.items():
        keys.append(k)
        files.append(v["wav_path"])
        classes.append(v["attack_type"])
        snrs.append(quant_snr(v["snr"]))
        benign_keys.append(v["benign_key"])

    benign_keys = np.unique(benign_keys)
    for k in benign_keys:
        keys.append(k)
        classes.append("benign")
        snrs.append("benign")
        files.append(k2w[k][0])

    u2c = Utt2Info.create(keys, classes)
    u2snr = Utt2Info.create(keys, snrs)
    wav = SCPList(keys, files)

    #####
    u2c.save(output_dir / "utt2attack")
    wav.save(output_dir / "wav.scp")

    mask = rng.rand(len(u2c)) > 1 / (1 + num_enroll_sides)
    enr_key = u2c.key[mask]
    test_key = u2c.key[mask == False]
    enr_u2c = u2c.filter(enr_key)
    test_u2c = u2c.filter(test_key)
    enr_u2snr = u2snr.filter(enr_key)
    test_u2snr = u2snr.filter(test_key)

    if num_enroll_sides > 1:
        class_uniq, class_ids = np.unique(test_u2c.info, return_inverse=True)
        snr_uniq, snr_ids = np.unique(test_u2snr.info, return_inverse=True)
        num_classes = len(class_uniq) * len(snr_uniq)
        count_sides = np.zeros((num_classes,), dtype=np.int)
        count_models = np.zeros((num_classes,), dtype=np.int)
        enroll_models = []
        for i in range(len(test_u2c)):
            a = class_ids[i]
            b = snr_ids[i]
            j = a * len(snr_uniq) + b
            side = count_sides[j] % num_enroll_sides
            if side == 0:
                count_models[j] += 1
            enroll_model = "%s-%s-%03d" % (class_uniq[a], snr_uniq[b], count_models[j])
            enroll_models.append(enroll_model)
            count_sides[j] += 1

        enr_u2e = Utt2Info.create(test_u2c.key, enroll_models)
        enroll_models_uniq, idx = np.unique(enroll_models, return_index=True)
        enr_e2c = Utt2Info.create(enroll_models_uniq, test_u2c.info[idx])
        enr_e2snr = Utt2Info.create(enroll_models_uniq, test_u2snr.info[idx])
    else:
        enr_e2c = enr_u2c
        enr_e2snr = enr_u2snr
        enr_u2e = Utt2Info.create(enr_u2c.key, enr_u2c.key)

    trials_all = TrialKey(enr_e2snr.key, test_u2c.key)
    trials_seen = TrialKey(enr_e2snr.key, test_u2c.key)
    trials_unseen = TrialKey(enr_e2snr.key, test_u2c.key)
    for i in range(len(enr_e2snr)):
        for j in range(len(test_u2c)):
            if enr_e2snr.info[i] == test_u2snr.info[j]:
                trials_all.tar[i, j] = True
                if enr_e2c.info[i] in seen_attacks and test_u2c.info[j] in seen_attacks:
                    trials_seen.tar[i, j] = True
                elif (
                    enr_e2c.info[i] not in seen_attacks
                    and test_u2c.info[j] not in seen_attacks
                ):
                    trials_unseen.tar[i, j] = True
            else:
                trials_all.non[i, j] = True
                if enr_e2c.info[i] in seen_attacks and test_u2c.info[j] in seen_attacks:
                    trials_seen.non[i, j] = True
                elif (
                    enr_e2c.info[i] not in seen_attacks
                    and test_u2c.info[j] not in seen_attacks
                ):
                    trials_unseen.non[i, j] = True

    max_trials = int(max_trials * 1e6)
    num_tar_trials = np.sum(trials_all.tar)
    num_non_trials = np.sum(trials_all.non)
    num_trials = num_tar_trials + num_non_trials
    if num_trials > max_trials:
        p = max_trials / num_trials
        logging.info("reducing number of trials (%d) with p=%f" % (num_trials, p))
        mask = rng.rand(*trials_all.tar.shape) > p
        trials_all.non[mask] = False
        trials_seen.non[mask] = False
        trials_unseen.non[mask] = False
        trials_all.tar[mask] = False
        trials_seen.tar[mask] = False
        trials_unseen.tar[mask] = False

    enr_u2e.sort(1)
    enr_u2e.save(output_dir / "utt2enr")
    trials_all.save_txt(output_dir / "trials")
    trials_seen.save_txt(output_dir / "trials_seen")
    trials_unseen.save_txt(output_dir / "trials_unseen")

    # train_u2d.save(output_dir / 'train_utt2dur')
    # val_u2d.save(output_dir / 'val_utt2dur')
    # trainval_u2d.save(output_dir / 'trainval_utt2dur')
    # test_u2d.save(output_dir / 'test_utt2dur')

    # with open(output_dir / 'class2int', 'w') as f:
    #     for c in uclasses:
    #         f.write('%s\n' % (c))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="prepare trial list to do attack type verification",
    )

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--seen-attacks", required=True, nargs="+")
    parser.add_argument("--benign-wav-file", required=True)
    parser.add_argument("--num-enroll-sides", default=1, type=int)
    parser.add_argument("--max-trials", default=10, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    make_lists(**vars(args))
