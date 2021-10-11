#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

  Trains Backend for SRE18 video condition
"""

import sys
import os
import argparse
import time
import logging
from jsonargparse import ArgumentParser, namespace_to_dict
from pathlib import Path

import numpy as np
import pandas as pd

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.transforms import TransformList, PCA, LDA, LNorm
from hyperion.helpers import PLDAFactory as F
from hyperion.utils import Utt2Info
from hyperion.io import RandomAccessDataReaderFactory as DRF


def filter_chn_spks(df, x):

    spks = df[df["language"].isin(["CMN", "YUE"])].speaker_id.unique()
    idx = df["speaker_id"].isin(spks)
    df_chn = df[idx]
    x_chn = x[idx]
    return df_chn, x_chn


def read_vox(v_dir, vox_name):

    data_dir = Path("data") / vox_name
    u2s_file = data_dir / "utt2spk"
    u2l_file = data_dir / "utt2est_lang"
    u2s = Utt2Info.load(u2s_file)
    u2l = Utt2Info.load(u2l_file).filter(u2s.key)
    df = pd.DataFrame(
        {"segment_id": u2s.key, "speaker_id": u2s.info, "language": u2l.info}
    )

    xvec_file = v_dir / vox_name / "xvector.scp"
    r = DRF.create(str(xvec_file))
    x = r.read(df["segment_id"], squeeze=True)

    return df, x


def read_sre(v_dir, sre_name):

    data_dir = Path("data") / sre_name
    u2s_file = data_dir / "utt2spk"
    u2l_file = data_dir / "utt2lang"
    u2s = Utt2Info.load(u2s_file)
    u2l = Utt2Info.load(u2l_file).filter(u2s.key)
    df = pd.DataFrame(
        {"segment_id": u2s.key, "speaker_id": u2s.info, "language": u2l.info}
    )

    xvec_file = v_dir / sre_name / "xvector.scp"
    r = DRF.create(str(xvec_file))
    x = r.read(df["segment_id"], squeeze=True)

    return df, x


def read_sre21(v_dir, sre_name):
    data_dir = Path("data") / sre_name
    segms_file = data_dir / "segments.csv"
    df = pd.read_csv(segms_file)
    df = df[df["language"] != "OTHER"]

    xvec_file = v_dir / sre_name / "xvector.scp"
    r = DRF.create(str(xvec_file))
    x = r.read(df["segment_id"], squeeze=True)

    return df, x


def compute_single_mu_cov(x):
    mu = np.mean(x, axis=0)
    delta = x - mu
    S = np.dot(delta.T, delta) / x.shape[0]
    return mu, S


def compute_prior_mu_cov(x_tel, x_afv):
    mu_tel, S_tel = compute_single_mu_cov(x_tel)
    mu_afv, S_afv = compute_single_mu_cov(x_afv)
    S = 0.5 * S_tel + 0.5 * S_afv
    return mu_tel, S_tel, mu_afv, S_afv


def compute_post_mu_cov(x_eng, x_cmn, x_yue, mu_0, S_0, r_mu, r_s):
    mu_1, S_1 = compute_single_mu_cov(x_eng)
    mu_eng = adapt_mu(mu_1, mu_0, x_eng.shape[0], r_mu)
    S_eng = adapt_cov(mu_1, S_1, mu_0, S_0, x_eng.shape[0], r_mu, r_s)

    mu_1, S_1 = compute_single_mu_cov(x_cmn)
    mu_cmn = adapt_mu(mu_1, mu_0, x_cmn.shape[0], r_mu)
    S_cmn = adapt_cov(mu_1, S_1, mu_0, S_0, x_cmn.shape[0], r_mu, r_s)

    mu_1, S_1 = compute_single_mu_cov(x_yue)
    mu_yue = adapt_mu(mu_1, mu_0, x_yue.shape[0], r_mu)
    S_yue = adapt_cov(mu_1, S_1, mu_0, S_0, x_yue.shape[0], r_mu, r_s)

    S = S_eng + S_cmn + S_yue
    S /= 3
    return mu_eng, mu_cmn, mu_yue, S


def adapt_mu(mu, mu0, N, r):
    alpha = N / (N + r)
    return alpha * mu + (1 - alpha) * mu0


def adapt_cov(mu, S, mu0, S0, N, r_mu, r_s):
    alpha = N / (N + r_mu)
    beta = N / (N + r_s)
    return beta * S + (1 - beta) * S0 + beta * (1 - alpha) * np.outer(mu, mu0)


def split_into_langs(df, x):
    idx = df["language"] == "ENG"
    x_eng = x[idx]
    class_eng = df[idx].speaker_id.values

    idx = df["language"].isin(
        ["CMN", "CMN.JPN", "CMN.JPN.WUU", "CMN.THU.WUU", "CMN.YUE"]
    )
    x_cmn = x[idx]
    class_cmn = df[idx].speaker_id.values

    idx = df["language"].isin(["YUE", "CMN.YUE"])
    x_yue = x[idx]
    class_yue = df[idx].speaker_id.values

    return x_eng, class_eng, x_cmn, class_cmn, x_yue, class_yue


def train_be(
    v_dir,
    vox_name,
    sre_name,
    sre21_enr_name,
    sre21_test_name,
    plda_type,
    y_dim,
    z_dim,
    epochs,
    ml_md,
    md_epochs,
    pca_var_r,
    r_mu,
    r_s,
    w_mu1,
    w_B1,
    w_W1,
    w_mu2,
    w_B2,
    w_W2,
    output_path,
    **kwargs,
):

    # Read data
    logging.info("Reading data")
    v_dir = Path("scp:" + v_dir)

    df_afv, x_afv = read_vox(v_dir, vox_name)
    df_cts, x_cts = read_sre(v_dir, sre_name)
    df_sre21e, x_sre21e = read_sre21(v_dir, sre21_enr_name)
    df_sre21t, x_sre21t = read_sre21(v_dir, sre21_test_name)
    df_sre21 = pd.concat([df_sre21e, df_sre21t], ignore_index=True)
    x_sre21 = np.concatenate((x_sre21e, x_sre21t), axis=0)

    logging.info(
        "num-segms-afv=%d num-segms-cts=%d num-segms=sre21=%d",
        x_afv.shape[0],
        x_cts.shape[0],
        x_sre21.shape[0],
    )
    logging.info("Train PCA")
    mu_cts, S_cts, mu_afv, S_afv = compute_prior_mu_cov(x_cts, x_afv)

    df_cts_chn, x_cts_chn = filter_chn_spks(df_cts, x_cts)
    (
        x_cts_eng,
        classes_cts_eng,
        x_cts_cmn,
        classes_cts_cmn,
        x_cts_yue,
        classes_cts_yue,
    ) = split_into_langs(df_cts_chn, x_cts_chn)
    logging.info(
        "num-adapt-segms-cts total=%d eng=%d, cmn=%d, yue=%d",
        x_cts_chn.shape[0],
        x_cts_eng.shape[0],
        x_cts_cmn.shape[0],
        x_cts_yue.shape[0],
    )
    mu_cts_eng, mu_cts_cmn, mu_cts_yue, S_cts_chn = compute_post_mu_cov(
        x_cts_eng, x_cts_cmn, x_cts_yue, mu_cts, S_cts, r_mu, r_s
    )

    df_afv_chn, x_afv_chn = filter_chn_spks(df_afv, x_afv)
    (
        x_afv_eng,
        classes_afv_eng,
        x_afv_cmn,
        classes_afv_cmn,
        x_afv_yue,
        classes_afv_yue,
    ) = split_into_langs(df_afv_chn, x_afv_chn)
    logging.info(
        "num-adapt-segms-afv total=%d eng=%d, cmn=%d, yue=%d",
        x_afv_chn.shape[0],
        x_afv_eng.shape[0],
        x_afv_cmn.shape[0],
        x_afv_yue.shape[0],
    )
    mu_afv_eng, mu_afv_cmn, mu_afv_yue, S_afv_chn = compute_post_mu_cov(
        x_afv_eng, x_afv_cmn, x_afv_yue, mu_afv, S_afv, r_mu, r_s
    )

    # all subconditons are weighted equally
    S = 0.5 * S_cts_chn + 0.5 * S_afv_chn

    # we include the whitening in the PCA
    pca = PCA(pca_var_r=pca_var_r, min_pca_dim=25, whiten=True, update_mu=False)
    pca.fit(S=S)

    # pca = PCA(pca_var_r=pca_var_r, min_pca_dim=25, whiten=True, name="pca")
    # xx = np.vstack((x_cts, x_afv))
    # pca.fit(xx)

    logging.info("PCA dim=%d for variance-ratio=%f", pca.T.shape[1], pca_var_r)
    if y_dim > pca.T.shape[1]:
        y_dim = pca.T.shape[1]

    # lnorm doesn't center and whiten, it just the length norm
    lnorm = LNorm(name="lnorm")
    # xxx = pca.predict(xx)
    # lnorm.fit(xxx)

    conds = [
        "cts",
        "cts_eng",
        "cts_cmn",
        "cts_yue",
        "afv",
        "afv_eng",
        "afv_cmn",
        "afv_yue",
    ]
    means = [
        mu_cts,
        mu_cts_eng,
        mu_cts_cmn,
        mu_cts_yue,
        mu_afv,
        mu_afv_eng,
        mu_afv_cmn,
        mu_afv_yue,
    ]
    preps = {}  # list of preprocessors
    for c, m in zip(conds, means):
        p = TransformList([PCA(mu=m, T=pca.T, name="pca"), lnorm])
        # p = TransformList([pca, lnorm])
        preps[c] = p

    xvecs = [
        x_cts,
        x_cts_eng,
        x_cts_cmn,
        x_cts_yue,
        x_afv,
        x_afv_eng,
        x_afv_cmn,
        x_afv_yue,
    ]

    # apply pca + lnorm
    logging.info("Apply PCA")
    xvecs_ln = {}
    for c, x_c in zip(conds, xvecs):
        xvecs_ln[c] = preps[c].predict(x_c)

    # we can delete original x
    del xvecs

    logging.info("Save preprocessors")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for c in conds:
        preps[c].save(output_path / f"cent_pca_lnorm_{c}.h5")

    # Train PLDA
    logging.info("Train PLDA CTS+AFV")
    classes_cts = df_cts.speaker_id.values
    classes_afv = df_afv.speaker_id.values
    x = np.concatenate((xvecs_ln["cts"], xvecs_ln["afv"]), axis=0)
    _, class_ids = np.unique(
        np.concatenate((classes_cts, classes_afv)), return_inverse=True
    )
    plda_global = F.create_plda(plda_type, y_dim=y_dim, z_dim=z_dim, name="plda")
    elbo = plda_global.fit(
        x, class_ids, epochs=epochs, ml_md=ml_md, md_epochs=md_epochs
    )
    del x, class_ids

    logging.info("Train PLDA CTS")
    classes_cts = df_cts.speaker_id.values
    _, class_ids = np.unique(classes_cts, return_inverse=True)
    plda = plda_global.copy()
    plda_cts = plda_global.copy()
    elbo = plda.fit(
        xvecs_ln["cts"], class_ids, epochs=epochs, ml_md=ml_md, md_epochs=md_epochs
    )
    plda_cts.weighted_avg_model(plda, w_mu1, w_B1, w_W1)
    plda_cts.save(output_path / "plda_cts.h5")
    elbo = pd.DataFrame(
        {"epoch": np.arange(epochs), "elbo": elbo[0], "elbo_per_sample": elbo[1]}
    )
    elbo.to_csv(output_path / "elbo_cts.csv", sep=",", index=False)

    logging.info("Adapt PLDA CTS")
    plda = plda_cts.copy()
    plda_adapt = plda_cts.copy()
    x = np.concatenate(
        (
            xvecs_ln["cts_eng"],
            xvecs_ln["cts_cmn"],
            xvecs_ln["cts_yue"],
        ),
        axis=0,
    )
    _, class_ids = np.unique(
        np.concatenate(
            (
                classes_cts_eng,
                classes_cts_cmn,
                classes_cts_yue,
            )
        ),
        return_inverse=True,
    )
    if np.max(class_ids) + 1 < plda.y_dim:
        plda.update_V = False

    elbo = plda.fit(x, class_ids, epochs=20)
    plda_adapt.weighted_avg_model(plda, w_mu2, w_B2, w_W2)
    plda_adapt.save(output_path / "plda_adapt_cts.h5")

    logging.info("Train PLDA AFV")
    classes_afv = df_afv.speaker_id.values
    _, class_ids = np.unique(classes_afv, return_inverse=True)
    plda = plda_global.copy()
    plda_afv = plda_global.copy()
    elbo = plda.fit(
        xvecs_ln["afv"], class_ids, epochs=epochs, ml_md=ml_md, md_epochs=md_epochs
    )
    plda_afv.weighted_avg_model(plda, w_mu1, w_B1, w_W1)
    plda_afv.save(output_path / "plda_afv.h5")
    elbo = pd.DataFrame(
        {"epoch": np.arange(epochs), "elbo": elbo[0], "elbo_per_sample": elbo[1]}
    )
    elbo.to_csv(output_path / "elbo_afv.csv", sep=",", index=False)

    logging.info("Adapt PLDA AFV")
    plda = plda_afv.copy()
    plda_adapt = plda_afv.copy()
    x = np.concatenate(
        (
            xvecs_ln["afv_eng"],
            xvecs_ln["afv_cmn"],
            xvecs_ln["afv_yue"],
        ),
        axis=0,
    )
    _, class_ids = np.unique(
        np.concatenate(
            (
                classes_afv_eng,
                classes_afv_cmn,
                classes_afv_yue,
            )
        ),
        return_inverse=True,
    )
    if np.max(class_ids) + 1 < plda.y_dim:
        plda.update_V = False

    elbo = plda.fit(x, class_ids, epochs=20)
    plda_adapt.weighted_avg_model(plda, w_mu2, w_B2, w_W2)
    plda_adapt.save(output_path / "plda_adapt_afv.h5")


if __name__ == "__main__":

    parser = ArgumentParser(
        description=(
            "Train PCA+LNorm+PLDA Back-end adapted to chinese speakers, "
            "trains different PLDA for CTS and Afv"
        )
    )

    parser.add_argument("--v-dir", required=True)
    parser.add_argument("--vox-name", required=True)
    parser.add_argument("--sre-name", required=True)
    parser.add_argument("--sre21-enr-name", required=True)
    parser.add_argument("--sre21-test-name", required=True)
    F.add_class_args(parser)

    parser.add_argument("--output-path", required=True)
    # parser.add_argument("--lda-dim", type=int, default=None)
    parser.add_argument("--pca-var-r", type=float, default=1)
    parser.add_argument("--r-mu", type=float, default=20)
    parser.add_argument("--r-s", type=float, default=20)
    parser.add_argument("--w-mu1", type=float, default=0.5)
    parser.add_argument("--w-B1", type=float, default=0.5)
    parser.add_argument("--w-W1", type=float, default=0.5)
    parser.add_argument("--w-mu2", type=float, default=0.5)
    parser.add_argument("--w-B2", type=float, default=0.5)
    parser.add_argument("--w-W2", type=float, default=0.5)

    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    train_be(**namespace_to_dict(args))
