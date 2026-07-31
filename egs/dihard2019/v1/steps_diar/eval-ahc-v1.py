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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils import Utt2Info, RTTM
from hyperion.utils.vad_utils import intersect_segment_timestamps_with_vad as istwv
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.io import VADReaderFactory as VRF
from hyperion.helpers import PLDAFactory as F
from hyperion.np.transforms import TransformList, PCA, LNorm
from hyperion.np.clustering import AHC
from hyperion.np.pdfs import GMMTiedDiagCov as GMM
from hyperion.np.diarization import DiarAHCPLDA as Diar

# from hyperion.np.pdfs import GMMDiagCov as GMM2
# from hyperion.np.pdfs import GMM as GMM3


def make_timestamps(n, win_start, win_length, win_shift, win_shrink):

    t1 = np.asarray([win_start + win_shift * i for i in range(n)])
    t2 = t1 + win_length - win_shrink
    t1 += win_shrink
    t1[t1 < 0] = 0
    assert np.all(t2 - t1 > 0)
    timestamps = np.concatenate((t1[:, None], t2[:, None]), axis=1)
    return timestamps


def init_readers(v_file, timestamps_file, vad_file, **kwargs):
    r_x = DRF.create(v_file)
    r_time = None
    if timestamps_file is not None:
        r_time = DRF.create(timetamps_file)

    r_vad = None
    if vad_file is not None:
        vad_args = VRF.filter_args(**kwargs)
        r_vad = VRF.create(vad_file, **vad_args)

    return r_x, r_time, r_vad


def load_test_list(test_list, part_idx, num_parts):
    utts = Utt2Info.load(test_list)
    utts = utts.split(part_idx, num_parts)
    keys = utts.key
    return keys


def load_feats(key, r_x, r_time, r_vad, win_start, win_length, win_shift, win_shrink):

    x = r_x.read([key])[0]
    if r_time is not None:
        timestamps = r_time.load([key])[0]
    else:
        timestamps = make_timestamps(
            len(x), win_start, win_length, win_shift, win_shrink
        )

    if r_vad is not None:
        vad_timestamps = r_vad.read_timestamps([key])[0]
        vad_bin = r_vad.read([key])[0]
        print(x.shape)
        print(timestamps)
        print(vad_timestamps)
        print(vad_bin)
        speech_idx, timestamps, ts2segs = istwv(timestamps, vad_timestamps)
        x = x[speech_idx]
    else:
        ts2segs = np.arange(len(x), dtype=np.int)

    return x, timestamps, ts2segs


def plot_score_hist(scores, output_file, thr=None, gmm=None):
    mask = np.triu(np.ones(scores.shape, dtype=np.bool), 1)
    scores_r = scores[mask].ravel()

    _, bins, _ = plt.hist(
        scores_r,
        100,
        histtype="step",
        density=True,
        color="b",
        linestyle="solid",
        linewidth=1.5,
    )

    if thr is not None:
        plt.axvline(x=thr, color="k")

    if gmm is not None:
        prob = np.exp(gmm.log_prob(bins[:, None]))
        plt.plot(bins, prob, color="r", linestyle="solid", linewidth=1.5)

    # plt.title(name)
    plt.xlabel("LLR score")
    plt.grid(True)
    # plt.legend()
    plt.savefig(output_file)
    plt.clf()


def twoGMMcalib_lin(s, niters=20):
    """
    Train two-Gaussian GMM with shared variance for calibration of scores 's'
    Returns threshold for original scores 's' that "separates" the two gaussians
    and array of linearly callibrated log odds ratio scores.
    """
    from scipy.special import softmax

    weights = np.array([0.5, 0.5])
    means = np.mean(s) + np.std(s) * np.array([-1, 1])
    var = np.var(s)
    for _ in range(niters):
        lls = (
            np.log(weights)
            - 0.5 * np.log(var)
            - 0.5 * (s[:, np.newaxis] - means) ** 2 / var
        )
        gammas = softmax(lls, axis=1)
        cnts = np.sum(gammas, axis=0)
        weights = cnts / cnts.sum()
        means = s.dot(gammas) / cnts
        var = ((s ** 2).dot(gammas) / cnts - means ** 2).dot(weights)

    logging.info("niko {} {} {}".format(means, var, weights))
    threshold = (
        -0.5
        * (np.log(weights ** 2 / var) - means ** 2 / var).dot([1, -1])
        / (means / var).dot([1, -1])
    )
    return threshold, lls[:, means.argmax()] - lls[:, means.argmin()]


def unsup_gmm_calibration(scores):
    mask = np.triu(np.ones(scores.shape, dtype=np.bool), 1)
    scores_r = scores[mask].ravel()[:, None]  # N x 1
    gmm_1c = GMM(num_comp=1)
    gmm_1c.fit(scores_r, epochs=1)
    # gmm_2c = GMM(
    #    mu=np.asarray([[np.max(scores_r), np.min(scores_r)]]).T,
    #    Lambda=gmm_1c.Lambda, pi=np.asarray([0.5, 0.5]))
    gmm_2c = gmm_1c.split_comp(2)
    # logging.info('gmm1 {} {} {}'.format(gmm_2c.mu, gmm_2c.Lambda, gmm_2c.pi))
    e = gmm_2c.fit(scores_r, epochs=20)
    # logging.info('gmm2 {} {} {} {} {} {} {}'.format(gmm_2c.mu, gmm_2c.Lambda, gmm_2c.pi, e,
    #                                              np.mean(gmm_2c.log_prob(scores_r)),
    #                                              np.sum(gmm_2c.compute_pz_nat(scores_r), axis=0),
    #                                              np.sum(gmm_2c.compute_pz_std(scores_r), axis=0)))
    # k1 = GMM2(num_comp=1)
    # k1.fit(scores_r, epochs=1)
    # k2 = k1.split_comp(2)
    # k2.fit(scores_r, epochs=20)
    # logging.info('k2 {} {} {}'.format(k2.mu, k2.Lambda, k2.pi))

    # k1 = GMM3(num_comp=1)
    # k1.fit(scores_r, epochs=1)
    # k2 = k1.split_comp(2)
    # k2.fit(scores_r, epochs=20)
    # logging.info('k3 {} {} {}'.format(k2.mu, k2.Lambda, k2.pi))

    # e = gmm_2c.fit(scores_r, epochs=1)
    # print(gmm_2c.mu, gmm_2c.Lambda, gmm_2c.pi, e, np.mean(gmm_2c.log_prob(scores_r)))
    # e = gmm_2c.fit(scores_r, epochs=1)
    # print(gmm_2c.mu, gmm_2c.Lambda, gmm_2c.pi, e, np.mean(gmm_2c.log_prob(scores_r)))
    scale = (gmm_2c.mu[0] - gmm_2c.mu[1]) * gmm_2c.Lambda
    bias = (
        (gmm_2c.mu[1] ** 2 - gmm_2c.mu[0] ** 2) * gmm_2c.Lambda / 2
        + np.log(gmm_2c.pi[0])
        - np.log(gmm_2c.pi[1])
    )
    scores = scale * scores + bias
    # scores1 = scale * scores_r + bias
    # scores2 = gmm_2c.compute_log_pz(scores_r)
    # scores2 = scores2[:,0] - scores2[:,1]

    # t, scores_niko = twoGMMcalib_lin(scores_r.ravel(), niters=20)
    # logging.info('scores={} {} {}'.format(scores1, scores2, scores_niko + np.log(gmm_2c.pi[0]) - np.log(gmm_2c.pi[1])))

    bic_lambda = 1
    n = len(scores_r)
    dparams = 4
    bic = (
        np.mean(gmm_2c.log_prob(scores_r) - gmm_1c.log_prob(scores_r))
        - bic_lambda * dparams * np.log(n) / 2 / n
    )
    return scores, bic, gmm_2c


def do_clustering(
    x,
    t_preproc,
    plda_model,
    threshold,
    pca_var_r,
    do_unsup_cal,
    use_bic,
    hist_file=None,
):

    x = t_preproc.predict(x)
    if pca_var_r < 1:
        pca = PCA(pca_var_r=pca_var_r, whiten=True)
        pca.fit(x)
        logging.info("PCA dim=%d" % pca.pca_dim)
        x = pca.predict(x)
        x = LNorm().predict(x)
        plda_model = plda_model.project(pca.T, pca.mu)

    scores = plda_model.llr_1vs1(x, x)
    if do_unsup_cal:
        scores_cal, bic, gmm_2c = unsup_gmm_calibration(scores)
        logging.info(
            "UnsupCal. BIC={} gmm.pi={} gmm.mu={} gmm.sigma={}".format(
                bic, gmm_2c.pi, gmm_2c.mu, np.sqrt(1.0 / gmm_2c.Lambda)
            )
        )
        if hist_file:
            hist_file_1 = "%s-nocal.pdf" % hist_file
            plot_score_hist(scores, hist_file_1, None, gmm_2c)
        scores = scores_cal

    if hist_file:
        hist_file_1 = "%s.pdf" % hist_file
        plot_score_hist(scores, hist_file_1, threshold)

    if use_bic and bic < 0:
        # unsup calibration detected only one Gaussian -> only target trials
        class_ids = np.zeros(len(x), dtype=np.int)
        return class_ids

    ahc = AHC()
    ahc.fit(scores)
    class_ids = ahc.get_flat_clusters(threshold)

    return class_ids


def eval_ahc(
    test_list,
    v_file,
    timestamps_file,
    vad_file,
    preproc_file,
    rttm_file,
    win_start=None,
    win_length=None,
    win_shift=None,
    win_shrink=0,
    score_hist_dir=None,
    part_idx=1,
    num_parts=1,
    **kwargs
):

    logging.info("reading utterance list %s" % test_list)
    keys = load_test_list(test_list, part_idx, num_parts)
    logging.info("init data readers")
    r_x, r_time, r_vad = init_readers(v_file, timestamps_file, vad_file, **kwargs)
    logging.info("loading embedding preprocessor: %s" % (preproc_file))
    t_preproc = TransformList.load(preproc_file)
    plda_args = F.filter_eval_args(**kwargs)
    logging.info("loading plda model={}".format(plda_args))
    plda_model = F.load_plda(**plda_args)
    diar_args = Diar.filter_args(**kwargs)
    diarizer = Diar(plda_model, t_preproc, **diar_args)

    if score_hist_dir is not None:
        score_hist_dir = Path(score_hist_dir)
        score_hist_dir.mkdir(parents=True, exist_ok=True)
    else:
        hist_file = None

    rttms = []

    for key in keys:
        logging.info("loading data for utt %s" % (key))
        x, timestamps, ts2segs = load_feats(
            key, r_x, r_time, r_vad, win_start, win_length, win_shift, win_shrink
        )

        logging.info("clustering utt {} x={}".format(key, x.shape))
        if score_hist_dir is not None:
            hist_file = score_hist_dir / key

        seg_class_ids = diarizer.cluster(x, hist_file)
        # seg_class_ids = do_clustering(
        #     x, t_preproc, plda_model, threshold, pca_var_r, do_unsup_cal, use_bic, hist_file)
        ts_class_ids = seg_class_ids[ts2segs]
        logging.info("utt %s found %d spks" % (key, np.max(seg_class_ids) + 1))

        rttm = RTTM.create_spkdiar_single_file(
            key, timestamps[:, 0], timestamps[:, 1] - timestamps[:, 0], ts_class_ids
        )
        rttm.merge_adjacent_segments()
        rttms.append(rttm)

    rttm = RTTM.merge(rttms)
    rttm.save(rttm_file)


if __name__ == "__main__":

    parser = ArgumentParser(description="Evals AHC with PLDA scoring")

    parser.add_argument("--test-list", required=True)
    parser.add_argument("--v-file", required=True)
    parser.add_argument("--timestamps-file", default=None)
    parser.add_argument("--vad-file", default=None)
    parser.add_argument("--preproc-file", default=None)
    VRF.add_argparse_args(parser, prefix="vad")

    F.add_argparse_eval_args(parser)
    Diar.add_argparse_args(parser)

    parser.add_argument("--win-start", default=-0.675, type=float)
    parser.add_argument("--win-length", default=1.5, type=float)
    parser.add_argument("--win-shift", default=0.25, type=float)
    parser.add_argument("--win-shrink", default=0.675, type=float)
    # parser.add_argument('--threshold', default=0, type=float)
    # parser.add_argument('--pca-var-r', default=1, type=float)
    # parser.add_argument('--do-unsup-cal', default=False, action='store_true')
    # parser.add_argument('--use-bic', default=False, action='store_true')

    parser.add_argument(
        "--part-idx",
        type=int,
        default=1,
        help=("splits the list of files in num-parts " "and process part_idx"),
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        default=1,
        help=("splits the list of files in num-parts " "and process part_idx"),
    )

    parser.add_argument("--rttm-file", required=True)
    parser.add_argument("--score-hist-dir", default=None)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_ahc(**vars(args))
