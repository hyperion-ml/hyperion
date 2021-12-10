#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

  Trains calibration for SRE21 audio
"""
import sys
import os
import time
import logging
from jsonargparse import ArgumentParser, namespace_to_dict
from pathlib import Path

import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils import TrialScores, TrialKey, Utt2Info
from hyperion.utils.list_utils import ismember
from hyperion.metrics import compute_act_dcf, compute_min_dcf
from hyperion.classifiers import BinaryLogisticRegression as LR


def read_key_and_scores(key_file, score_file):
    logging.info("load scores: %s", score_file)
    scr = TrialScores.load_txt(score_file)
    logging.info("load key: %s", key_file)
    key = TrialKey.load_txt(key_file)
    tar, non = scr.get_tar_non(key)
    return key, tar, non


def read_nenr_sre16(nenr_file, key):
    nenr = Utt2Info.load(nenr_file)
    nenr = nenr.filter(key.model_set)
    ntar = np.sum(key.tar)
    nnon = np.sum(key.non)
    tar_nenr = np.zeros((ntar, 2), dtype=float_cpu())
    non_nenr = np.zeros((nnon, 2), dtype=float_cpu())
    for i, c in enumerate(["1", "3"]):
        mask = np.zeros_like(key.tar, dtype=float_cpu())
        mask[nenr.info == c, :] = 1.0

        cond_c = TrialScores(key.model_set, key.seg_set, mask)
        tar_c, non_c = cond_c.get_tar_non(key)
        tar_nenr[:, i] = tar_c
        non_nenr[:, i] = non_c

    assert np.all(np.sum(tar_nenr, axis=1)), "non all tars have nenroll info"
    assert np.all(np.sum(non_nenr, axis=1)), "non all nons have nenroll info"

    return tar_nenr, non_nenr


def read_nenr(key_file, key):
    ntar = np.sum(key.tar)
    nnon = np.sum(key.non)
    tar_nenr = np.zeros((ntar, 2), dtype=float_cpu())
    non_nenr = np.zeros((nnon, 2), dtype=float_cpu())
    for i, c in enumerate(["nenr1", "nenr3"]):
        logging.info("load key: %s", f"{key_file}_{c}")
        key_c = TrialKey.load_txt(f"{key_file}_{c}")
        mask = np.zeros_like(key.tar, dtype=float_cpu())
        f, idx = ismember(key_c.model_set, key.model_set)
        mask[idx, :] = 1.0
        cond_c = TrialScores(key.model_set, key.seg_set, mask)
        tar_c, non_c = cond_c.get_tar_non(key)
        tar_nenr[:, i] = tar_c
        non_nenr[:, i] = non_c

    assert np.all(np.sum(tar_nenr, axis=1)), "non all tars have nenroll info"
    assert np.all(np.sum(non_nenr, axis=1)), "non all nons have nenroll info"

    return tar_nenr, non_nenr


def read_langs(key_file, key):
    ntar = np.sum(key.tar)
    nnon = np.sum(key.non)
    tar_langs = np.zeros((ntar, 6), dtype=float_cpu())
    non_langs = np.zeros((nnon, 6), dtype=float_cpu())
    langs = ["ENG", "CMN", "YUE"]
    k = 0
    for i in range(len(langs)):
        for j in range(i, len(langs)):
            li = langs[i]
            lj = langs[j]
            logging.info("load key: %s", f"{key_file}_{li}_{lj}")
            key_c = TrialKey.load_txt(f"{key_file}_{li}_{lj}")
            mask_c = np.logical_or(key_c.tar, key_c.non).astype(dtype=float_cpu())
            mask = np.zeros_like(key.tar, dtype=float_cpu())
            f, enr_idx = ismember(key_c.model_set, key.model_set)
            f, test_idx = ismember(key_c.seg_set, key.seg_set)
            mask[np.ix_(enr_idx, test_idx)] = mask_c
            cond_c = TrialScores(key.model_set, key.seg_set, mask)
            tar_c, non_c = cond_c.get_tar_non(key)
            tar_langs[:, k] = tar_c
            non_langs[:, k] = non_c
            k += 1

    return tar_langs, non_langs


def read_sources(key_file, key):
    ntar = np.sum(key.tar)
    nnon = np.sum(key.non)
    tar_src = np.zeros((ntar, 3), dtype=float_cpu())
    non_src = np.zeros((nnon, 3), dtype=float_cpu())
    sources = ["CTS", "AFV"]
    k = 0
    for i in range(len(sources)):
        for j in range(i, len(sources)):
            li = sources[i]
            lj = sources[j]
            logging.info("load key: %s", f"{key_file}_{li}_{lj}")
            key_c = TrialKey.load_txt(f"{key_file}_{li}_{lj}")
            mask_c = np.logical_or(key_c.tar, key_c.non).astype(dtype=float_cpu())
            mask = np.zeros_like(key.tar, dtype=float_cpu())
            f, enr_idx = ismember(key_c.model_set, key.model_set)
            f, test_idx = ismember(key_c.seg_set, key.seg_set)
            mask[np.ix_(enr_idx, test_idx)] = mask_c
            cond_c = TrialScores(key.model_set, key.seg_set, mask)
            tar_c, non_c = cond_c.get_tar_non(key)
            tar_src[:, k] = tar_c
            non_src[:, k] = non_c
            k += 1

    assert np.all(np.sum(tar_src, axis=1)), "non all tars have source info"
    assert np.all(np.sum(non_src, axis=1)), "non all nons have source info"

    return tar_src, non_src


def make_yueyue(key):
    ntar = np.sum(key.tar)
    nnon = np.sum(key.non)
    tar_langs = np.zeros((ntar, 6), dtype=float_cpu())
    non_langs = np.zeros((nnon, 6), dtype=float_cpu())
    tar_langs[:, -1] = 1
    non_langs[:, -1] = 1
    return tar_langs, non_langs


def make_ctscts(key):
    ntar = np.sum(key.tar)
    nnon = np.sum(key.non)
    tar_src = np.zeros((ntar, 3), dtype=float_cpu())
    non_src = np.zeros((nnon, 3), dtype=float_cpu())
    tar_src[:, 0] = 1
    non_src[:, 0] = 1
    return tar_src, non_src


def train_calibration(
    score_file_sre16,
    score_file_sre,
    score_file_sre21,
    key_file_sre16,
    key_file_sre,
    key_file_sre21,
    nenr_sre16,
    model_file,
    prior,
    lambda_reg,
    sre_weight,
    no_lang,
    no_nenr,
    no_source,
    verbose,
):

    key, tar_sre16, non_sre16 = read_key_and_scores(key_file_sre16, score_file_sre16)
    tar_nenr_sre16, non_nenr_sre16 = read_nenr_sre16(nenr_sre16, key)
    tar_langs_sre16, non_langs_sre16 = make_yueyue(key)
    tar_src_sre16, non_src_sre16 = make_ctscts(key)

    key, tar_sre, non_sre = read_key_and_scores(key_file_sre, score_file_sre)
    tar_nenr_sre, non_nenr_sre = read_nenr(key_file_sre, key)
    tar_langs_sre, non_langs_sre = read_langs(key_file_sre, key)
    tar_src_sre, non_src_sre = make_ctscts(key)

    key, tar_sre21, non_sre21 = read_key_and_scores(key_file_sre21, score_file_sre21)
    tar_nenr_sre21, non_nenr_sre21 = read_nenr(key_file_sre21, key)
    tar_langs_sre21, non_langs_sre21 = read_langs(key_file_sre21, key)
    tar_src_sre21, non_src_sre21 = read_sources(key_file_sre21, key)

    # concatenate all datasets
    logging.info("train calibration on all data")
    ntar = len(tar_sre16) + len(tar_sre) + len(tar_sre21)
    nnon = len(non_sre16) + len(non_sre) + len(non_sre21)
    scores = np.concatenate(
        (tar_sre16, tar_sre, tar_sre21, non_sre16, non_sre, non_sre21)
    )
    y = np.concatenate(
        (np.ones((ntar,), dtype="int32"), np.zeros((nnon,), dtype="int32"))
    )

    nenr = np.concatenate(
        (
            tar_nenr_sre16,
            tar_nenr_sre,
            tar_nenr_sre21,
            non_nenr_sre16,
            non_nenr_sre,
            non_nenr_sre21,
        ),
        axis=0,
    )
    langs = np.concatenate(
        (
            tar_langs_sre16,
            tar_langs_sre,
            tar_langs_sre21,
            non_langs_sre16,
            non_langs_sre,
            non_langs_sre21,
        ),
        axis=0,
    )
    src = np.concatenate(
        (
            tar_src_sre16,
            tar_src_sre,
            tar_src_sre21,
            non_src_sre16,
            non_src_sre,
            non_src_sre21,
        ),
        axis=0,
    )
    if no_lang:
        langs[:] = 0
    if no_nenr:
        nenr[:] = 0
    if no_source:
        src[:] = 0

    x = np.concatenate((scores[:, None], nenr, langs, src), axis=1)

    # remove non eng/cts/yue langs
    if not no_lang:
        mask = np.any(langs, axis=1)
        x = x[mask]
        y = y[mask]

    lr0 = LR(
        prior=prior,
        lambda_reg=lambda_reg,
        bias_scaling=1,
        solver="liblinear",
        verbose=verbose,
    )
    lr0.fit(x, y)
    logging.info(f"SRE16+superset+21 -> A: {lr0.A} b: {lr0.b}")

    logging.info("train calibration on only sre21 data")
    ntar = len(tar_sre21)
    nnon = len(non_sre21)
    scores = np.concatenate((tar_sre21, non_sre21))
    y = np.concatenate(
        (np.ones((ntar,), dtype="int32"), np.zeros((nnon,), dtype="int32"))
    )

    nenr = np.concatenate(
        (
            tar_nenr_sre21,
            non_nenr_sre21,
        ),
        axis=0,
    )
    langs = np.concatenate(
        (
            tar_langs_sre21,
            non_langs_sre21,
        ),
        axis=0,
    )
    src = np.concatenate(
        (
            tar_src_sre21,
            non_src_sre21,
        ),
        axis=0,
    )
    if no_lang:
        langs[:] = 0
    if no_nenr:
        nenr[:] = 0
    if no_source:
        src[:] = 0

    x = np.concatenate((scores[:, None], nenr, langs, src), axis=1)

    # remove non eng/cts/yue langs
    if not no_lang:
        mask = np.any(langs, axis=1)
        x = x[mask]
        y = y[mask]
    ntar = np.sum(y == 1)
    nnon = np.sum(y == 0)

    min_dcf, p_miss, p_fa = compute_min_dcf(x[y == 1, 0], x[y == 0, 0], prior)
    n_miss = p_miss * ntar
    n_fa = p_fa * nnon
    logging.info(
        "min_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f"
        % (min_dcf, p_miss * 100, p_fa * 100, n_miss, n_fa)
    )

    lr = LR(
        prior=prior,
        lambda_reg=lambda_reg,
        bias_scaling=1,
        solver="liblinear",
        verbose=verbose,
    )
    lr.fit(x, y)
    logging.info(f"SRE21 -> A: {lr.A} b: {lr.b}")

    logging.info("calibrate scores on sre21")
    scores_cal = lr.predict(x)
    tar_cal = scores_cal[y == 1]
    non_cal = scores_cal[y == 0]
    act_dcf, p_miss, p_fa = compute_act_dcf(tar_cal, non_cal, prior)
    n_miss = p_miss * ntar
    n_fa = p_fa * nnon
    logging.info(
        "act_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f"
        % (act_dcf, p_miss * 100, p_fa * 100, n_miss, n_fa)
    )

    A = (1 - sre_weight) * lr.A + sre_weight * lr0.A
    b = (1 - sre_weight) * lr.b + sre_weight * lr0.b
    lr = LR(A=A, b=b, prior=prior)
    logging.info(f"SRE21 adapted -> A: {lr.A} b: {lr.b}")
    logging.info("calibrate scores on adapted calibration")

    scores_cal = lr.predict(x)
    tar_cal = scores_cal[y == 1]
    non_cal = scores_cal[y == 0]
    act_dcf, p_miss, p_fa = compute_act_dcf(tar_cal, non_cal, prior)
    n_miss = p_miss * ntar
    n_fa = p_fa * nnon
    logging.info(
        "act_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f"
        % (act_dcf, p_miss * 100, p_fa * 100, n_miss, n_fa)
    )

    logging.info("save calibration at %s", model_file)
    lr.save(model_file)


if __name__ == "__main__":

    parser = ArgumentParser(description="Trains llr calibration")

    parser.add_argument("--score-file-sre16", required=True)
    parser.add_argument("--score-file-sre", required=True)
    parser.add_argument("--score-file-sre21", required=True)
    parser.add_argument("--key-file-sre16", required=True)
    parser.add_argument("--key-file-sre", required=True)
    parser.add_argument("--key-file-sre21", required=True)
    parser.add_argument("--nenr-sre16", required=True)
    parser.add_argument("--model-file", dest="model_file", required=True)
    parser.add_argument("--prior", type=float, default=0.01)
    parser.add_argument("--lambda-reg", type=float, default=1e-5)
    parser.add_argument("--sre-weight", type=float, default=0.5)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    parser.add_argument("--no-lang", default=False, action="store_true")
    parser.add_argument("--no-nenr", default=False, action="store_true")
    parser.add_argument("--no-source", default=False, action="store_true")

    args = parser.parse_args()
    config_logger(args.verbose)
    logging.debug(args)

    train_calibration(**namespace_to_dict(args))
