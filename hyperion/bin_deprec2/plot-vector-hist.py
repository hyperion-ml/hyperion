#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import os
import argparse
import time
import logging

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyperion.hyp_defs import config_logger
from hyperion.helpers import VectorReader as VR
from hyperion.np.transforms import TransformList


def plot_vector_hist(
    iv_file, v_list, preproc_file, output_path, num_bins, normed, **kwargs
):

    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    vr_args = VR.filter_args(**kwargs)
    vr = VR(iv_file, v_list, preproc, **vr_args)
    x = vr.read()

    t1 = time.time()

    if not os.path.exists(output_path):
        os.makedirs(ouput_path)

    for i in range(x.shape[1]):

        fig_file = "%s/D%04d.pdf" % (output_path, i)

        plt.hist(x[:, i], num_bins, normed=normed)
        plt.xlabel("Dim %d" % i)
        plt.grid(True)
        plt.show()
        plt.savefig(fig_file)
        plt.clf()

    logging.info("Elapsed time: %.2f s." % (time.time() - t1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Plots historgrams of i-vectors",
    )

    parser.add_argument("--iv-file", dest="iv_file", required=True)
    parser.add_argument("--v-list", dest="v_list", required=True)
    parser.add_argument("--preproc-file", dest="preproc_file", default=None)

    VR.add_argparse_args(parser)

    parser.add_argument("--output-path", dest="output_path", required=True)
    parser.add_argument(
        "--no-normed", dest="normed", default=True, action="store_false"
    )
    parser.add_argument("--num-bins", dest="num_bins", type=int, default=100)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    plot_vector_hist(**vars(args))
