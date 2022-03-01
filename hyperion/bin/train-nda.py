#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
"""
Trains NDA
"""

import sys
import os
import argparse
import time
import logging

import numpy as np

from hyperion.hyp_defs import config_logger
from hyperion.helpers import VectorClassReader as VCR
from hyperion.np.transforms import TransformList, NDA, NSbSw


def train_nda(
    iv_file,
    train_list,
    preproc_file,
    nda_dim,
    K,
    alpha,
    name,
    save_tlist,
    append_tlist,
    output_path,
    **kwargs
):

    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    vr_args = VCR.filter_args(**kwargs)
    vcr = VCR(iv_file, train_list, preproc, **vr_args)
    x, class_ids = vcr.read()

    t1 = time.time()

    s_mat = NSbSw(K=K, alpha=alpha)
    s_mat.fit(x, class_ids)

    model = NDA(name=name)
    model.fit(mu=s_mat.mu, Sb=s_mat.Sb, Sw=s_mat.Sw, nda_dim=nda_dim)

    logging.info("Elapsed time: %.2f s." % (time.time() - t1))

    x = model.predict(x)

    s_mat = NSbSw()
    s_mat.fit(x, class_ids)
    logging.debug(s_mat.Sb[:4, :4])
    logging.debug(s_mat.Sw[:4, :4])

    if save_tlist:
        if append_tlist and preproc is not None:
            preproc.append(model)
            model = preproc
        else:
            model = TransformList(model)

    model.save(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Train NDA",
    )

    parser.add_argument("--iv-file", dest="iv_file", required=True)
    parser.add_argument("--train-list", dest="train_list", required=True)
    parser.add_argument("--preproc-file", dest="preproc_file", default=None)

    VCR.add_argparse_args(parser)

    parser.add_argument("--output-path", dest="output_path", required=True)
    parser.add_argument("--nda-dim", dest="nda_dim", type=int, default=None)
    parser.add_argument("--k", dest="K", type=int, default=10)
    parser.add_argument("--alpha", dest="alpha", type=float, default=1)

    parser.add_argument(
        "--no-save-tlist", dest="save_tlist", default=True, action="store_false"
    )
    parser.add_argument(
        "--no-append-tlist", dest="append_tlist", default=True, action="store_false"
    )
    parser.add_argument("--name", dest="name", default="nda")
    parser.add_argument("--vector-score-file", dest="vector_score_file", default=None)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    train_nda(**vars(args))
