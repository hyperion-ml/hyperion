#!/bin/env python
"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, namespace_to_dict
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyperion.hyp_defs import config_logger
from hyperion.utils import Utt2Info
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.transforms import PCA, SklTSNE, LNorm

colors = ["b", "g", "r", "c", "m", "y", "k"]
markers = ["x", "o", "+", "*", "s", "h", "D", "^", "v", "p", "8"]

color_marker = [(c, m) for m in markers for c in colors]


def load_sre(v_dir, num_spks):

    rng = np.random.RandomState(seed=1123)
    data_dir = Path("data/sre_cts_superset_16k_trn")
    segms_file = data_dir / "segments.csv"
    df_segms = pd.read_csv(segms_file)
    df_segms = df_segms[
        (df_segms.language == "CMN")
        | (df_segms.language == "YUE")
        | (df_segms.language == "ENG")
    ]
    df_segms["corpus_id"] = "sre_superset"
    df_segms.drop(columns=["speech_duration", "session_id", "phone_id"], inplace=True)
    spks_cmn = df_segms[df_segms.language == "CMN"].speaker_id.unique()

    if len(spks_cmn) > num_spks:
        spks_cmn = spks_cmn[rng.permutation(len(spks_cmn))[:num_spks]]

    spks_yue = df_segms[df_segms.language == "YUE"].speaker_id.unique()
    mask = np.ones(len(spks_yue), dtype=bool)
    for i in range(len(spks_yue)):
        if spks_yue[i] in spks_cmn:
            mask[i] = False

    spks_yue = spks_yue[mask]
    if len(spks_yue) > num_spks:
        spks_yue = spks_yue[rng.permutation(len(spks_yue))[:num_spks]]

    df_cmn = df_segms[df_segms.speaker_id.isin(spks_cmn)]
    df_yue = df_segms[df_segms.speaker_id.isin(spks_yue)]
    df = pd.concat([df_cmn, df_yue])

    v_file = v_dir / "sre_cts_superset_16k_trn" / "xvector.scp"
    reader = DRF.create("scp:" + str(v_file))
    x = reader.read(df.segment_id.values, squeeze=True)
    return df, x


def load_sre16(v_dir, lang, num_spks):

    rng = np.random.RandomState(seed=1123)
    if lang == "YUE":
        name = "sre16_eval_tr60_yue"
    else:
        name = "sre16_train_dev_ceb"

    data_dir = Path("data") / name
    u2s = pd.read_csv(data_dir / "utt2spk", sep=" ", names=["segment_id", "speaker_id"])
    s2g = pd.read_csv(data_dir / "spk2gender", sep=" ", names=["speaker_id", "gender"])
    df = pd.merge(u2s, s2g, on="speaker_id")
    df["language"] = lang
    df["corpus_id"] = "sre16"

    spks = df.speaker_id.unique()
    if len(spks) > num_spks:
        spks = spks[rng.permutation(len(spks))[:num_spks]]

    df = df[df.speaker_id.isin(spks)]
    df.loc[df.gender == "f", "gender"] = "female"
    df.loc[df.gender == "m", "gender"] = "male"

    v_file = v_dir / name / "xvector.scp"
    reader = DRF.create("scp:" + str(v_file))
    x = reader.read(df.segment_id.values, squeeze=True)
    return df, x


def load_sre21(v_dir, num_spks):
    rng = np.random.RandomState(seed=1123)
    data_dir = Path("data/sre21_audio_dev_enroll")
    segms_file = data_dir / "segments.csv"
    df_e = pd.read_csv(segms_file)
    data_dir = Path("data/sre21_audio_dev_test")
    segms_file = data_dir / "segments.csv"
    df_t = pd.read_csv(segms_file)
    df_e["side"] = "enroll"
    df_t["side"] = "test"
    df_segms = pd.concat([df_e, df_t])
    df_segms = df_segms[df_segms.source_type == "cts"]
    df_segms["corpus_id"] = "sre21_dev"
    df_segms.drop(columns=["source_type"], inplace=True)
    spks_cmn = df_segms[df_segms.language == "CMN"].speaker_id.unique()

    if len(spks_cmn) > num_spks:
        spks_cmn = spks_cmn[rng.permutation(len(spks_cmn))[:num_spks]]

    spks_yue = df_segms[df_segms.language == "YUE"].speaker_id.unique()
    mask = np.ones(len(spks_yue), dtype=bool)
    for i in range(len(spks_yue)):
        if spks_yue[i] in spks_cmn:
            mask[i] = False

    spks_yue = spks_yue[mask]
    if len(spks_yue) > num_spks:
        spks_yue = spks_yue[rng.permutation(len(spks_yue))[:num_spks]]

    df_cmn = df_segms[df_segms.speaker_id.isin(spks_cmn)]
    df_yue = df_segms[df_segms.speaker_id.isin(spks_yue)]
    df = pd.concat([df_cmn, df_yue])

    v_file = v_dir / "sre21_audio_dev" / "xvector.scp"
    reader = DRF.create("scp:" + str(v_file))
    x = reader.read(df.segment_id.values, squeeze=True)
    return df, x


def make_fig(x, classes, title, output_dir, name):
    fig_file = output_dir / f"tsne_{name}.png"
    classes, class_ids = np.unique(classes, return_inverse=True)
    num_classes = min(np.max(class_ids) + 1, len(color_marker))
    for c in range(num_classes):
        idx = class_ids == c
        plt.scatter(
            x[idx, 0],
            x[idx, 1],
            c=color_marker[c][0],
            marker=color_marker[c][1],
            label=classes[c],
        )

    if num_classes < 16:
        fs = 10
    elif num_classes < 24:
        fs = 8
    elif num_classes < 32:
        fs = 7
    else:
        fs = 6

    plt.legend(fontsize=fs)
    plt.grid(True)
    plt.title(title + " " + name)
    plt.savefig(fig_file)
    plt.clf()


def plot_tsne_cts(v_dir, pca_var_r, num_spks, lnorm, title, tsne, verbose):

    config_logger(verbose)
    v_dir = Path(v_dir)
    logging.info("Reading data")
    df_sre, x_sre = load_sre(v_dir, num_spks)
    df_sre16_yue, x_sre16_yue = load_sre16(v_dir, "YUE", num_spks)
    df_sre16_cmn, x_sre16_cmn = load_sre16(v_dir, "CMN", num_spks)
    df_sre21, x_sre21 = load_sre21(v_dir, num_spks)

    df = pd.concat([df_sre, df_sre16_yue, df_sre16_cmn, df_sre21])
    x = np.vstack((x_sre, x_sre16_yue, x_sre16_cmn, x_sre21))

    logging.info("Readed %d vectors", x.shape[0])
    if lnorm:
        x = LNorm().predict(x)

    if pca_var_r < 1:
        logging.info("Training PCA")
        pca = PCA(pca_var_r=pca_var_r)
        pca.fit(x)
        x_pca = pca.predict(x)
        logging.info("pca-dim={}".format(x_pca.shape[1]))
    else:
        x_pca = x

    logging.info("Training T-SNE")
    prob_plot = 1
    tsne_args = SklTSNE.filter_args(**tsne)
    tsne = SklTSNE(**tsne_args)
    x_tsne = tsne.fit(x_pca)
    # p = np.random.rand(x_tsne.shape[0]) < prob_plot
    # x_tsne = x_tsne[p]
    # df = df[p]

    logging.info("Making figs")
    output_dir = v_dir / "tsne_cts" / f"pca{x_pca.shape[1]}_ns{num_spks}"
    output_dir.mkdir(parents=True, exist_ok=True)
    make_fig(x_tsne, df.speaker_id, title, output_dir, "speaker")
    make_fig(x_tsne, df.language, title, output_dir, "language")
    make_fig(x_tsne, df.corpus_id, title, output_dir, "corpus")
    make_fig(x_tsne, df.gender, title, output_dir, "gender")

    idx = df.corpus_id == "sre21_dev"
    df_sre21 = df[idx]
    x_sre21 = x_tsne[idx]
    make_fig(x_sre21, df_sre21.speaker_id, title, output_dir, "sre21-speaker")
    make_fig(x_sre21, df_sre21.language, title, output_dir, "sre21-language")
    make_fig(x_sre21, df_sre21.gender, title, output_dir, "sre21-gender")

    idx = df.corpus_id == "sre16"
    df_sre16 = df[idx]
    x_sre16 = x_tsne[idx]
    make_fig(x_sre16, df_sre16.speaker_id, title, output_dir, "sre16-speaker")
    make_fig(x_sre16, df_sre16.language, title, output_dir, "sre16-language")
    make_fig(x_sre16, df_sre16.gender, title, output_dir, "sre16-gender")

    idx = df.corpus_id == "sre_superset"
    df_sre = df[idx]
    x_sre = x_tsne[idx]
    make_fig(x_sre, df_sre.speaker_id, title, output_dir, "sre-superset-speaker")
    make_fig(x_sre, df_sre.language, title, output_dir, "sre-superset-language")
    make_fig(x_sre, df_sre.gender, title, output_dir, "sre-superset-gender")

    idx = df.speaker_id.isin([133009, 133081, 133112])
    df_s = df[idx]
    x_s = x_tsne[idx]
    make_fig(x_s, df_s.speaker_id, title, output_dir, "sre21-3spk-1-speaker")
    make_fig(x_s, df_s.language, title, output_dir, "sre21-3spk-1-language")
    make_fig(x_s, df_s.gender, title, output_dir, "sre21-3spk-1-gender")

    idx = df.speaker_id.isin([133009, 133062])
    df_s = df[idx]
    x_s = x_tsne[idx]
    make_fig(x_s, df_s.speaker_id, title, output_dir, "sre21-test-spks-speaker")
    make_fig(x_s, df_s.language, title, output_dir, "sre21-test-spks-language")
    make_fig(x_s, df_s.gender, title, output_dir, "sre21-test-spks-gender")

    idx = df.speaker_id.isin([133009, 133062])
    df_s = df[(~idx) & (df.corpus_id == "sre21_dev")]
    x_s = x_tsne[(~idx) & (df.corpus_id == "sre21_dev")]
    make_fig(x_s, df_s.speaker_id, title, output_dir, "sre21-notest-spks-speaker")
    make_fig(x_s, df_s.language, title, output_dir, "sre21-notest-spks-language")
    make_fig(x_s, df_s.gender, title, output_dir, "sre21-notest-spks-gender")

    idx = df.side == "enroll"
    df_s = df[idx & (df.corpus_id == "sre21_dev")]
    x_s = x_tsne[idx & (df.corpus_id == "sre21_dev")]
    make_fig(x_s, df_s.speaker_id, title, output_dir, "sre21-enroll-segs-speaker")
    make_fig(x_s, df_s.language, title, output_dir, "sre21-enroll-segs-language")
    make_fig(x_s, df_s.gender, title, output_dir, "sre21-enroll-segs-gender")

    for spk in [133009, 133062]:
        idx = df.speaker_id == spk
        df_s = df[idx]
        x_s = x_tsne[idx]
        make_fig(x_s, df_s.segment_id, title, output_dir, f"sre21-spk{spk}-segs")
        make_fig(x_s, df_s.side, title, output_dir, f"sre21-spk{spk}-side")


if __name__ == "__main__":

    parser = ArgumentParser(description="Makes TSNE plots for SRE21 CTS")
    parser.add_argument("--v-dir", required=True, help="Path to x-vector dir")
    parser.add_argument("--pca-var-r", default=0.95, type=float)
    parser.add_argument("--num-spks", default=10, type=float)
    parser.add_argument("--lnorm", default=False, action="store_true")
    parser.add_argument("--title", default="T-SNE CTS")
    SklTSNE.add_class_args(parser, prefix="tsne")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    plot_tsne_cts(**namespace_to_dict(args))
