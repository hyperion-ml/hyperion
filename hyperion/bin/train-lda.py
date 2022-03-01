#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
"""
Trains LDA
"""
import sys
import os
import argparse
import time
import logging

import numpy as np

from hyperion.hyp_defs import config_logger
from hyperion.helpers import VectorClassReader as VCR
from hyperion.np.transforms import TransformList, LDA, SbSw


def train_lda(
    iv_file,
    train_list,
    preproc_file,
    lda_dim,
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

    vcr_args = VCR.filter_args(**kwargs)
    vcr = VCR(iv_file, train_list, preproc, **vcr_args)
    x, class_ids = vcr.read()

    t1 = time.time()

    model = LDA(lda_dim=lda_dim, name=name)
    model.fit(x, class_ids)

    logging.info("Elapsed time: %.2f s." % (time.time() - t1))

    x = model.predict(x)

    s_mat = SbSw()
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
        description="Train LDA",
    )

    parser.add_argument("--iv-file", dest="iv_file", required=True)
    parser.add_argument("--train-list", dest="train_list", required=True)
    parser.add_argument("--preproc-file", dest="preproc_file", default=None)

    VCR.add_argparse_args(parser)

    parser.add_argument("--output-path", dest="output_path", required=True)
    parser.add_argument("--lda-dim", dest="lda_dim", type=int, default=None)
    parser.add_argument(
        "--no-save-tlist", dest="save_tlist", default=True, action="store_false"
    )
    parser.add_argument(
        "--no-append-tlist", dest="append_tlist", default=True, action="store_false"
    )
    parser.add_argument("--name", dest="name", default="lda")
    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    train_lda(**vars(args))
