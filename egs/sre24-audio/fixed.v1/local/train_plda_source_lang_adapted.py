#!/usr/bin/env python
""" 
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba) 
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import logging
import os
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np.pdfs import PLDAFactory
from hyperion.np.transforms import LDA, PCA, CentWhiten, LNorm, TransformList
from hyperion.utils import SegmentSet


def load_segments_and_feats(segments_file, feats_file):  # , class_name):
    logging.info("loading segments: %s feats: %s", segments_file, feats_file)
    segments = SegmentSet.load(segments_file)
    reader = DRF.create(feats_file)
    x = reader.read(segments["id"], squeeze=True)
    is_nan = np.any(np.isnan(x), axis=1)
    print(segments[is_nan])
    x = x[~is_nan]
    segments = SegmentSet(segments[~is_nan])
    # _, y = np.unique(segments[class_name], return_inverse=True)
    return segments, x  # , y


def load_segments_and_feats_list(segments_files, feats_files):
    assert len(segments_files) == len(feats_files)
    segments_list = []
    feats_list = []
    for segments_file, feats_file in zip(segments_files, feats_files):
        segments, feats = load_segments_and_feats(segments_file, feats_file)
        segments_list.append(segments)
        feats_list.append(feats)

    segments = SegmentSet.cat(segments_list)
    feats = np.concatenate(feats_list, axis=0)
    return segments, feats


def load_data(ood_segments_files, ood_feats_files, id_segments_files, id_feats_files):
    ood_segments, ood_feats = load_segments_and_feats_list(
        ood_segments_files, ood_feats_files
    )
    logging.info("ood-segments=%d", len(ood_segments))
    if id_segments_files is not None:
        id_segments, id_feats = load_segments_and_feats_list(
            id_segments_files, id_feats_files
        )
        logging.info("id-segments=%d", len(id_segments))
    else:
        id_segments, id_feats = None, None

    return ood_segments, ood_feats, id_segments, id_feats


def select_target_lang_speakers(segments, feats, class_name, langs):

    speakers = segments.loc[segments["language"].isin(langs), class_name].unique()
    idx = segments[class_name].isin(speakers)
    segments = SegmentSet(segments[idx])
    feats = feats[idx]
    return segments, feats


def compute_mean_and_cov(x):
    mu = np.mean(x, axis=0)
    delta = x - mu
    S = np.dot(delta.T, delta) / x.shape[0]
    return mu, S


# def adapt_mu(mu_ml, mu_0, N, r):
#     alpha = N / (N + r)
#     return alpha * mu_ml + (1 - alpha) * mu_0


# def adapt_cov(mu_ml, S_ml, mu_0, S_0, N, r_mu, r_s):
#     alpha = N / (N + r_mu)
#     beta = N / (N + r_s)
#     return beta * S_ml + (1 - beta) * S0 + beta * (1 - alpha) * np.outer(mu_ml, mu0)


def adapt_mean_and_cov(mu_ml, S_ml, mu_0, S_0, N, r_mu, r_s):
    alpha = N / (N + r_mu)
    beta = N / (N + r_s)
    logging.info(
        "adapt mean and cov N=%f r_mu=%f r_s=%s alpha=%f beta=%f",
        N,
        r_mu,
        r_s,
        alpha,
        beta,
    )
    mu = alpha * mu_ml + (1 - alpha) * mu_0
    S = beta * S_ml + (1 - beta) * S_0 + beta * (1 - alpha) * np.outer(mu_ml, mu_0)
    return mu, S


def compute_prior_means_and_covs(segments, feats, source_types):
    mu = {}
    S = {}
    for source_type in source_types:
        idx = segments["source_type"] == source_type
        feats_s = feats[idx]
        logging.info(
            "prior mean/conv source: %s num_samples: %d", source_type, feats_s.shape[0]
        )
        mu_s, S_s = compute_mean_and_cov(feats_s)
        mu[source_type] = mu_s
        S[source_type] = S_s

    return mu, S


def compute_post_means_and_covs(
    segments,
    feats,
    mu_prior,
    S_prior,
    source_types,
    target_langs,
    r_mu,
    r_s,
):
    mu_post = {}
    S_post = 0
    if target_langs is None:
        weight = 1 / len(source_types)
    else:
        weight = 1 / (len(source_types) * len(target_langs))

    for source_type in source_types:
        mu_prior_s = mu_prior[source_type]
        S_prior_s = S_prior[source_type]
        if target_langs is None:
            idx = segments["source_type"] == source_type
            feats_sl = feats[idx]
            logging.info(
                "post mean/conv source: %s num_samples: %d",
                source_type,
                feats_sl.shape[0],
            )

            mu_ml, S_ml = compute_mean_and_cov(feats_sl)
            mu_post_sl, S_post_sl = adapt_mean_and_cov(
                mu_ml, S_ml, mu_prior_s, S_prior_s, feats_sl.shape[0], r_mu, r_s
            )
            mu_post[source_type] = mu_post_sl
            S_post += weight * S_post_sl
        else:
            mu_post[source_type] = 0
            for lang in target_langs:
                idx = (segments["source_type"] == source_type) & (
                    segments["language"] == lang
                )
                feats_sl = feats[idx]
                logging.info(
                    "post mean/conv source: %s lang: %s num_samples: %d",
                    source_type,
                    lang,
                    feats_sl.shape[0],
                )

                mu_ml, S_ml = compute_mean_and_cov(feats_sl)
                mu_post_sl, S_post_sl = adapt_mean_and_cov(
                    mu_ml, S_ml, mu_prior_s, S_prior_s, feats_sl.shape[0], r_mu, r_s
                )
                mu_post[f"{source_type}_{lang}"] = mu_post_sl
                mu_post[source_type] += mu_post_sl / len(target_langs)
                S_post += weight * S_post_sl

    return mu_post, S_post


def apply_pcas(segments, feats, mu, pca, source_types, target_langs):
    pcas = {}
    feats_out = np.zeros((feats.shape[0], pca.pca_dim), dtype=feats.dtype)
    done = np.zeros((feats.shape[0],), dtype=bool)
    for source_type in source_types:
        if target_langs is None:
            idx = segments["source_type"] == source_type
            logging.info(
                "apply pca source: %s num_samples: %d", source_type, np.sum(idx)
            )
            pca_s = PCA(mu=mu[source_type], T=pca.T, name="pca")
            feats_out[idx] = pca_s(feats[idx])

            pcas[source_type] = pca_s
            done[idx] = True
        else:
            for lang in target_langs:
                key = f"{source_type}_{lang}"
                idx = (segments["source_type"] == source_type) & (
                    segments["language"] == lang
                )
                logging.info(
                    "apply pca source: %s lang: %s num_samples: %d",
                    source_type,
                    lang,
                    np.sum(idx),
                )

                pca_s = PCA(mu=mu[key], T=pca.T, name="pca")
                feats_out[idx] = pca_s(feats[idx])
                pcas[key] = pca_s
                done[idx] = True

            idx = (segments["source_type"] == source_type) & (
                ~segments["language"].isin(target_langs)
            )
            logging.info(
                "apply pca source: %s lang: others num_samples: %d",
                source_type,
                np.sum(idx),
            )

            pca_s = PCA(mu=mu[source_type], T=pca.T, name="pca")
            feats_out[idx] = pca_s(feats[idx])

            pcas[source_type] = pca_s
            done[idx] = True

    assert np.all(
        done
    ), f"no pca applied for {np.sum(~done)} segments: {segments.loc[~done,['id','source_type', 'language']]}"
    return feats_out, pcas


def make_transform_lists(pca_lnorm, pcas, lda_lnorm, lda, plda_lnorm):
    transforms = {}
    for key, pca in pcas.items():
        t = []
        if pca_lnorm is not None:
            t.append(pca_lnorm)

        t.append(pca)
        if lda_lnorm is not None:
            t.append(lda_lnorm)

        if lda is not None:
            t.append(lda)

        if plda_lnorm:
            t.append(plda_lnorm)

        transforms[key] = TransformList(t)

    return transforms


def train_plda(
    ood_segments_files,
    ood_feats_files,
    id_segments_files,
    id_feats_files,
    class_name,
    source_types,
    target_langs,
    ood_speaker_langs,
    preproc_file,
    preproc_adapt_file,
    plda_file,
    plda_adapt_file,
    pca,
    pca_adapt,
    lda,
    plda,
    plda_adapt,
    do_pca_lnorm,
    do_lda,
    do_lda_lnorm,
    do_plda_lnorm,
    # plda_center,
    # plda_whiten,
    # **kwargs,
):

    ood_segments, ood_feats, id_segments, id_feats = load_data(
        ood_segments_files, ood_feats_files, id_segments_files, id_feats_files
    )

    if do_pca_lnorm:
        logging.info("LNorm before PCA")
        pca_lnorm = LNorm(name="pca_lnorm")
        ood_feats = pca_lnorm(ood_feats)
        if id_feats is not None:
            id_feats = plda_lnorm(id_feats)
    else:
        pca_lnorm = None

    # Compute source dependent prior mean and cov
    if id_segments is not None:
        segments = SegmentSet.cat([ood_segments, id_segments])
        feats = np.concatenate([ood_feats, id_feats])
    else:
        segments, feats = ood_segments, ood_feats

    mu_prior, S_prior = compute_prior_means_and_covs(segments, feats, source_types)

    # select speaker that speak the target langs in ood data
    if ood_speaker_langs is not None:
        ood_id_segments, ood_id_feats = select_target_lang_speakers(
            ood_segments, ood_feats, class_name, ood_speaker_langs
        )

        if id_segments is None:
            id_segments, id_feats = ood_id_segments, ood_id_feats
        else:
            id_segments = SegmentSet.cat([ood_id_segments, id_segments])
            id_feats = np.concatenate([ood_id_feats, id_feats], axis=0)

        # remove adaptation data from ood data
        ood_idx = ~ood_segments["id"].isin(id_segments["id"])
        ood_segments = SegmentSet(ood_segments.loc[ood_idx])
        ood_feats = ood_feats[ood_idx]

    if id_segments is None:
        # if there is no indomain data, we just use ood data for adapt
        id_segments = ood_segments
        id_feats = ood_feats

    mu_post, S_post = compute_post_means_and_covs(
        id_segments,
        id_feats,
        mu_prior,
        S_prior,
        source_types,
        target_langs,
        pca_adapt["r_mu"],
        pca_adapt["r_s"],
    )

    pca["update_mu"] = False
    pca_model = PCA(**pca)
    pca_model.fit(S=S_post)
    logging.info("pca-dim=%d", pca_model.pca_dim)

    logging.info("Center+PCA+Lnorm ood data")
    ood_feats, ood_pcas = apply_pcas(
        ood_segments, ood_feats, mu_prior, pca_model, source_types, None
    )

    logging.info("Center+PCA+Lnorm id data")
    id_feats, id_pcas = apply_pcas(
        id_segments, id_feats, mu_post, pca_model, source_types, target_langs
    )

    segments = SegmentSet.cat([ood_segments, id_segments])
    _, y = np.unique(segments[class_name], return_inverse=True)

    if do_lda and x.shape[1] > lda["lda_dim"]:
        if do_lda_lnorm:
            logging.info("LNorm before LDA")
            lda_lnorm = LNorm(name="lda_lnorm")
            ood_feats = lda_lnorm(ood_feats)
            id_feats = lda_lnorm(id_feats)

        feats = np.concatenate((ood_feats, id_feats), axis=0)

        logging.info("Training LDA")
        lda_model = LDA(**lda)
        lda_model.fit(feats, y)
        ood_feats = lda_model(ood_feats)
        id_feats = lda_model(id_feats)
    else:
        lda_lnorm, lda_model = None, None

    if do_plda_lnorm:
        logging.info("LNorm before PLDA")
        plda_lnorm = LNorm(name="plda_lnorm")
        ood_feats = plda_lnorm(ood_feats)
        id_feats = plda_lnorm(id_feats)
    else:
        plda_lnorm = None

    transforms_ood = make_transform_lists(
        pca_lnorm, ood_pcas, lda_lnorm, lda_model, plda_lnorm
    )
    transforms_id = make_transform_lists(
        pca_lnorm, id_pcas, lda_lnorm, lda_model, plda_lnorm
    )
    logging.info("Save preprocessors")
    with open(preproc_file, "wb") as f:
        pickle.dump(transforms_ood, f)

    with open(preproc_adapt_file, "wb") as f:
        pickle.dump(transforms_id, f)

    logging.info(
        "Training PLDA ood_samples: %d id_samples: %d",
        ood_feats.shape[0],
        id_feats.shape[0],
    )
    feats = np.concatenate((ood_feats, id_feats), axis=0)
    plda["y_dim"] = min(ood_feats.shape[1], plda["y_dim"])
    plda = PLDAFactory.create(**plda)
    elbo, elbo_norm = plda.fit(feats, y)

    logging.info("Saving PLDA")
    plda.save(plda_file)
    loss_file = Path(plda_file).with_suffix(".csv")
    df_loss = pd.DataFrame(
        {"epoch": np.arange(1, len(elbo) + 1), "elbo": elbo, "elbo_norm": elbo_norm}
    )
    df_loss.to_csv(loss_file, index=False)

    logging.info("Adapt PLDA id_samples: %d", id_feats.shape[0])
    _, y_id = np.unique(id_segments[class_name], return_inverse=True)

    plda_adapted = plda.copy()
    if np.max(y_id) + 1 < plda.y_dim:
        plda.update_V = False

    elbo, elbo_norm = plda.fit(id_feats, y_id)
    plda_adapted.weighted_avg_model(plda, **plda_adapt)
    # plda_adapt["w_mu"], plda_adapt["w_B"], plda_adapt["w_W"))
    logging.info("Saving Adapt PLDA")
    plda_adapted.save(plda_adapt_file)
    loss_file = Path(plda_adapt_file).with_suffix(".csv")
    df_loss = pd.DataFrame(
        {"epoch": np.arange(1, len(elbo) + 1), "elbo": elbo, "elbo_norm": elbo_norm}
    )
    df_loss.to_csv(loss_file, index=False)


def main():
    parser = ArgumentParser(
        description="""Trains PLDA model and embedding preprocessor 
        with adaptation"""
    )
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--ood-feats-files", nargs="+", required=True)
    parser.add_argument("--id-feats-files", default=None, nargs="+")
    parser.add_argument("--ood-segments-files", nargs="+", required=True)
    parser.add_argument("--id-segments-files", default=None, nargs="+")
    parser.add_argument("--class-name", default="speaker")
    parser.add_argument("--source-types", default=["cts", "afv"], nargs="+")
    parser.add_argument("--target-langs", default=None, nargs="+")
    parser.add_argument("--ood-speaker-langs", default=None, nargs="+")
    parser.add_argument("--preproc-file", required=True)
    parser.add_argument("--preproc-adapt-file", required=True)
    parser.add_argument("--plda-file", required=True)
    parser.add_argument("--plda-adapt-file", required=True)
    PCA.add_class_args(parser, prefix="pca")
    LDA.add_class_args(parser, prefix="lda")
    PLDAFactory.add_class_args(parser, prefix="plda")
    parser.add_argument("--do-pca-lnorm", default=False, action=ActionYesNo)
    parser.add_argument("--do-lda-lnorm", default=False, action=ActionYesNo)
    parser.add_argument("--do-lda", default=False, action=ActionYesNo)
    parser.add_argument("--do-plda-lnorm", default=True, action=ActionYesNo)
    # parser.add_argument("--plda-center", default=True, action=ActionYesNo)
    # parser.add_argument("--plda-whiten", default=True, action=ActionYesNo)
    parser.add_argument("--pca_adapt.r-mu", type=float, default=20)
    parser.add_argument("--pca_adapt.r-s", type=float, default=20)
    parser.add_argument("--plda_adapt.w-mu", type=float, default=0.5)
    parser.add_argument("--plda_adapt.w-B", type=float, default=0.5)
    parser.add_argument("--plda_adapt.w-W", type=float, default=0.5)

    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    args = parser.parse_args()
    config_logger(args.verbose)
    logging.debug(args)
    del args["verbose"]
    del args["cfg"]
    train_plda(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
