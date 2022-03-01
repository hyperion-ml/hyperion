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
from mpl_toolkits.mplot3d import Axes3D as plt3d

from sklearn.manifold import TSNE

from hyperion.hyp_defs import config_logger
from hyperion.io import DataWriterFactory as DWF
from hyperion.helpers import VectorClassReader as VCR
from hyperion.np.transforms import TransformList, PCA

colors = ["b", "g", "r", "c", "m", "y", "k"]
markers = ["x", "o", "+", "*", "s", "h", "D", "^", "v", "p", "8"]


def plot_vector_tsne(
    iv_file,
    v_list,
    preproc_file,
    output_path,
    save_embed,
    output_dim,
    perplexity,
    exag,
    lr,
    num_iter,
    init_method,
    rng_seed,
    verbose,
    pca_dim,
    max_classes,
    **kwargs
):

    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    vr_args = VCR.filter_args(**kwargs)
    vcr = VCR(iv_file, v_list, preproc, **vr_args)

    x, class_ids = vcr.read()

    t1 = time.time()

    if pca_dim > 0:
        pca = PCA(pca_dim=pca_dim)
        pca.fit(x)
        x = pca.predict(x)

    if not os.path.exists(output_path):
        os.makedirs(ouput_path)

    tsne_obj = lambda n: TSNE(
        n_components=n,
        perplexity=perplexity,
        early_exaggeration=exag,
        learning_rate=lr,
        n_iter=num_iter,
        init=init_method,
        random_state=rng_seed,
        verbose=verbose,
    )

    if max_classes > 0:
        index = class_ids < max_classes
        x = x[index]
        class_ids = class_ids[index]

    if output_dim > 3:
        tsne = tsne_obj(output_dim)
        y = tsne.fit_transform(x)

        if save_embed:
            h5_file = "%s/embed_%dd.h5" % (output_path, ouput_dim)
            hw = DWF.create(h5_file)
            hw.write(vcr.u2c.key, y)

    tsne = tsne_obj(2)
    y = tsne.fit_transform(x)
    if save_embed:
        h5_file = "%s/embed_2d.h5" % output_path
        hw = DWF.create(h5_file)
        hw.write(vcr.u2c.key, y)

    fig_file = "%s/tsne_2d.pdf" % (output_path)
    # plt.scatter(y[:,0], y[:,1], c=class_ids, marker='x')

    color_marker = [(c, m) for m in markers for c in colors]
    for c in np.unique(class_ids):
        idx = class_ids == c
        plt.scatter(
            y[idx, 0],
            y[idx, 1],
            c=color_marker[c][0],
            marker=color_marker[c][1],
            label=vcr.class_names[c],
        )

    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(fig_file)
    plt.clf()

    # if max_classes > 0:
    #     fig_file = '%s/tsne_2d_n%d.pdf' % (output_path, max_classes)
    #     index = class_ids < max_classes
    #     plt.scatter(y[index,0], y[index,1], c=class_ids[index], marker='x')
    #     plt.grid(True)
    #     plt.show()
    #     plt.savefig(fig_file)
    #     plt.clf()

    tsne = tsne_obj(3)
    y = tsne.fit_transform(x)
    if save_embed:
        h5_file = "%s/embed_3d.h5" % output_path
        hw = DWF.create(h5_file)
        hw.write(vcr.u2c.key, y)

    fig_file = "%s/tsne_3d.pdf" % (output_path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(y[:,0], y[:,1], y[:,2], c=class_ids, marker='x')
    for c in np.unique(class_ids):
        idx = class_ids == c
        ax.scatter(
            y[idx, 0],
            y[idx, 1],
            y[idx, 2],
            c=color_marker[c][0],
            marker=color_marker[c][1],
            label=vcr.class_names[c],
        )

    plt.grid(True)
    plt.show()
    plt.savefig(fig_file)
    plt.clf()

    # if max_classes > 0:
    #     fig_file = '%s/tsne_3d_n%d.pdf' % (output_path, max_classes)
    #     index = class_ids < max_classes
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(y[index,0], y[index,1], y[index,2], c=class_ids[index], marker='x')
    #     plt.grid(True)
    #     plt.show()
    #     plt.savefig(fig_file)
    #     plt.clf()

    logging.info("Elapsed time: %.2f s." % (time.time() - t1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Plots TSNE embeddings",
    )

    parser.add_argument("--iv-file", dest="iv_file", required=True)
    parser.add_argument("--v-list", dest="v_list", required=True)
    parser.add_argument("--preproc-file", dest="preproc_file", default=None)

    VCR.add_argparse_args(parser)

    parser.add_argument("--output-path", dest="output_path", required=True)
    parser.add_argument(
        "--save-embed", dest="save_embed", default=False, action="store_true"
    )

    parser.add_argument("--output-dim", dest="output_dim", type=int, default=3)
    parser.add_argument("--perplexity", dest="perplexity", type=float, default=30)
    parser.add_argument("--exag", dest="exag", type=float, default=12)
    parser.add_argument("--lr", dest="lr", type=float, default=200)
    parser.add_argument("--num-iter", dest="num_iter", type=int, default=1000)
    parser.add_argument(
        "--init-method", dest="init_method", default="pca", choices=["random", "pca"]
    )
    parser.add_argument("--rng-seed", dest="rng_seed", type=int, default=1024)
    parser.add_argument("--pca-dim", dest="pca_dim", type=int, default=50)
    parser.add_argument("--max-classes", dest="max_classes", type=int, default=10)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    logging.debug(args)

    plot_vector_tsne(**vars(args))
