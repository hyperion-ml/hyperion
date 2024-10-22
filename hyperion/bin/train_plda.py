#!/usr/bin/env python
""" 
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba) 
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import logging
import os
import sys
import time
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


def load_data(segments_file, feats_file, class_name):
    logging.info("loading data")
    segments = SegmentSet.load(segments_file)
    reader = DRF.create(feats_file)
    x = reader.read(segments["id"], squeeze=True)
    _, y = np.unique(segments[class_name], return_inverse=True)
    return segments, x, y


def train_pca(x, pca_lnorm, pca_args):
    pca_var_r = pca_args["pca_var_r"]
    logging.info("computing pca pca_var_r=%f", pca_var_r)
    pca = None
    pca_lnorm = None
    if pca_var_r < 1:
        if pca_lnorm:
            logging.info("LNorm before PCA")
            pca_lnorm = LNorm(name="pca_lnorm")
            x = pca_lnorm(x)

        pca = PCA(**pca_args)
        pca.fit(x)
        x = pca(x)
        logging.info("pca-dim=%d", x.shape[1])

    return x, pca_lnorm, pca


def train_plda(
    segments_file,
    feats_file,
    class_name,
    preproc_file,
    plda_file,
    pca,
    lda,
    plda,
    pca_lnorm,
    do_lda,
    lda_lnorm,
    plda_lnorm,
    plda_center,
    plda_whiten,
):
    segments, x, y = load_data(segments_file, feats_file, class_name)
    transform_list = []

    x, pca_lnorm, pca_model = train_pca(x, pca_lnorm, pca)
    if pca_lnorm is not None:
        transform_list.append(pca_lnorm)

    if pca_model is not None:
        transform_list.append(pca_model)

    if do_lda and x.shape[1] > lda["lda_dim"]:
        if lda_lnorm:
            logging.info("LNorm before LDA")
            t = LNorm(name="lda_lnorm")
            x = t(x)
            transform_list.append(t)

        logging.info("Training LDA")
        lda_model = LDA(**lda)
        lda_model.fit(x, y)
        x = lda_model(x)
        transform_list.append(lda_model)

    if plda_center or plda_whiten:
        if plda_lnorm:
            t = LNorm(update_mu=plda_center, update_T=plda_whiten, name="plda_lnorm")
        else:
            t = CentWhiten(update_mu=plda_center, update_T=plda_whiten, name="plda_cw")

        logging.info("Training Center/Whiten/LNorm")
        t.fit(x)
        logging.info("Center/Whiten/LNorm before PLDA")
        x = t(x)
        transform_list.append(t)
    elif plda_lnorm:
        logging.info("LNorm before PLDA")
        t = LNorm(name="plda_lnorm")
        x = t(x)
        transform_list.append(t)

    logging.info("Training PLDA")
    plda["y_dim"] = min(x.shape[1], plda["y_dim"])
    plda = PLDAFactory.create(**plda)
    elbo, elbo_norm = plda.fit(x, y)

    logging.info("Saving Models")
    if len(transform_list) > 0:
        transform_list = TransformList(transform_list)
        transform_list.save(preproc_file)

    plda.save(plda_file)
    loss_file = Path(plda_file).with_suffix(".csv")
    loss_df = pd.DataFrame(
        {"epoch": np.arange(1, len(elbo) + 1), "elbo": elbo, "elbo_norm": elbo_norm}
    )
    loss_df.to_csv(loss_file, index=False)


def main():
    parser = ArgumentParser(description="Trains PLDA model and embedding preprocessor")
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--feats-file", required=True)
    parser.add_argument("--segments-file", required=True)
    parser.add_argument("--class-name", default="speaker")
    parser.add_argument("--preproc-file", required=True)
    parser.add_argument("--plda-file", required=True)
    PCA.add_class_args(parser, prefix="pca")
    LDA.add_class_args(parser, prefix="lda")
    PLDAFactory.add_class_args(parser, prefix="plda")
    parser.add_argument("--pca-lnorm", default=False, action=ActionYesNo)
    parser.add_argument("--lda-lnorm", default=False, action=ActionYesNo)
    parser.add_argument("--do-lda", default=False, action=ActionYesNo)
    parser.add_argument("--plda-lnorm", default=True, action=ActionYesNo)
    parser.add_argument("--plda-center", default=True, action=ActionYesNo)
    parser.add_argument("--plda-whiten", default=True, action=ActionYesNo)

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
