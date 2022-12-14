#!/usr/bin/env python
""" 
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba) 
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import logging
import sys
import os
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
    ActionYesNo,
)
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt

from hyperion.hyp_defs import config_logger
from hyperion.utils import SegmentSet
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np.transforms import PCA, SklTSNE, LNorm

matplotlib.use("Agg")
colors = ["b", "g", "r", "c", "m", "y", "k"]
markers = ["x", "o", "+", "*", "s", "h", "D", "^", "v", "p", "8"]

color_marker = [(c, m) for m in markers for c in colors]


def plot_embedding_tsne(
    train_v_file,
    train_list,
    pca_var_r,
    prob_plot,
    lnorm,
    title,
    max_classes,
    unlabeled,
    plot_class_names,
    output_dir,
    **kwargs,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("loading data")
    train_segs = SegmentSet.load(train_list)
    train_reader = DRF.create(train_v_file)
    x_trn = train_reader.read(train_segs["id"], squeeze=True)
    del train_reader
    logging.info("loaded %d samples", x_trn.shape[0])
    if lnorm:
        x_trn = LNorm().predict(x_trn)

    if pca_var_r < 1:
        pca = PCA(pca_var_r=pca_var_r)
        pca.fit(x_trn)
        x_pca = pca.predict(x_trn)
        logging.info("pca-dim=%d", x_pca.shape[1])
    else:
        x_pca = x_trn

    tsne_args = SklTSNE.filter_args(**kwargs["tsne"])
    tsne = SklTSNE(**tsne_args)
    x_tsne = tsne.fit(x_pca)
    p = np.random.rand(x_tsne.shape[0]) <= prob_plot
    x_tsne = x_tsne[p]
    logging.info("plots %d samples", x_tsne.shape[0])

    if unlabeled:
        plot_class_names = ["none"]

    for col in plot_class_names:
        fig_file = f"{output_dir}/train_tsne_{col}.png"
        if not unlabeled:
            classes = train_segs.loc[p, col]
            classes, class_ids = np.unique(classes, return_inverse=True)
            if max_classes is not None:
                index = class_ids < max_classes
                x_tsne_filtered = x_tsne[index]
                class_ids = class_ids[index]
            else:
                x_tsne_filtered = x_tsne

        else:
            class_ids = np.zeros((len(x_tsne.shape[0]),), dtype=np.int)
            classes = [None]

        for c in range(np.max(class_ids) + 1):
            idx = class_ids == c
            if not unlabeled:
                logging.info("plot class %s with %d samples", classes[c], np.sum(idx))
            plt.scatter(
                x_tsne_filtered[idx, 0],
                x_tsne_filtered[idx, 1],
                c=color_marker[c][0],
                marker=color_marker[c][1],
                label=classes[c],
            )

        if not unlabeled:
            plt.legend()
        plt.grid(True)
        plt.title(title)
        plt.savefig(fig_file)
        plt.clf()

    # fig_file = "%s/tsne_3d.pdf" % (output_dir)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # # ax.scatter(y[:,0], y[:,1], y[:,2], c=class_ids, marker='x')
    # for c in np.unique(class_ids):
    #     idx = class_ids == c
    #     ax.scatter(
    #         y[idx, 0],
    #         y[idx, 1],
    #         y[idx, 2],
    #         c=color_marker[c][0],
    #         marker=color_marker[c][1],
    #         label=vcr.class_names[c],
    #     )

    # plt.grid(True)
    # plt.show()
    # plt.savefig(fig_file)
    # plt.clf()


if __name__ == "__main__":

    parser = ArgumentParser(description="Projects embeddings using TSNE")

    parser.add_argument("--train-v-file", required=True)
    parser.add_argument("--train-list", required=True)

    parser.add_argument("--pca-var-r", default=0.95, type=float)
    parser.add_argument("--prob-plot", default=0.1, type=float)
    parser.add_argument("--lnorm", default=False, action=ActionYesNo)
    parser.add_argument("--unlabeled", default=False, action=ActionYesNo)
    parser.add_argument(
        "--plot-class-names",
        default=["class_id"],
        nargs="+",
        help="names of the class columns we plot",
    )
    parser.add_argument("--title", default="")
    SklTSNE.add_class_args(parser, prefix="tsne")

    parser.add_argument(
        "--max-classes", default=None, type=int, help="max number of clases to plot"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    plot_embedding_tsne(**namespace_to_dict(args))


# #!/usr/bin/env python
# """
#  Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# """

# import sys
# import os
# from jsonargparse import (
#     ArgumentParser,
#     ActionConfigFile,
#     ActionParser,
#     namespace_to_dict,
# )
# import time
# import logging

# import numpy as np
# import pandas as pd
# import matplotlib

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D as plt3d

# from sklearn.manifold import TSNE

# from hyperion.hyp_defs import config_logger
# from hyperion.io import DataWriterFactory as DWF
# from hyperion.helpers import VectorClassReader as VCR
# from hyperion.np.transforms import TransformList, PCA

# matplotlib.use("Agg")
# colors = ["b", "g", "r", "c", "m", "y", "k"]
# markers = ["x", "o", "+", "*", "s", "h", "D", "^", "v", "p", "8"]


# def plot_embedding_tsne(
#     v_file,
#     v_list,
#     preproc_file,
#     output_dir,
#     save_embed,
#     output_dim,
#     perplexity,
#     exag,
#     lr,
#     num_iter,
#     init_method,
#     rng_seed,
#     verbose,
#     pca_dim,
#     max_classes,
#     **kwargs
# ):

#     if preproc_file is not None:
#         preproc = TransformList.load(preproc_file)
#     else:
#         preproc = None

#     vr_args = VCR.filter_args(**kwargs)
#     vcr = VCR(iv_file, v_list, preproc, **vr_args)

#     x, class_ids = vcr.read()

#     t1 = time.time()

#     if pca_dim > 0:
#         pca = PCA(pca_dim=pca_dim)
#         pca.fit(x)
#         x = pca.predict(x)

#     if not os.path.exists(output_path):
#         os.makedirs(ouput_path)

#     tsne_obj = lambda n: TSNE(
#         n_components=n,
#         perplexity=perplexity,
#         early_exaggeration=exag,
#         learning_rate=lr,
#         n_iter=num_iter,
#         init=init_method,
#         random_state=rng_seed,
#         verbose=verbose,
#     )

#     if max_classes > 0:
#         index = class_ids < max_classes
#         x = x[index]
#         class_ids = class_ids[index]

#     if output_dim > 3:
#         tsne = tsne_obj(output_dim)
#         y = tsne.fit_transform(x)

#         if save_embed:
#             h5_file = "%s/embed_%dd.h5" % (output_path, ouput_dim)
#             hw = DWF.create(h5_file)
#             hw.write(vcr.u2c.key, y)

#     tsne = tsne_obj(2)
#     y = tsne.fit_transform(x)
#     if save_embed:
#         h5_file = "%s/embed_2d.h5" % output_path
#         hw = DWF.create(h5_file)
#         hw.write(vcr.u2c.key, y)

#     fig_file = "%s/tsne_2d.pdf" % (output_path)
#     # plt.scatter(y[:,0], y[:,1], c=class_ids, marker='x')

#     color_marker = [(c, m) for m in markers for c in colors]
#     for c in np.unique(class_ids):
#         idx = class_ids == c
#         plt.scatter(
#             y[idx, 0],
#             y[idx, 1],
#             c=color_marker[c][0],
#             marker=color_marker[c][1],
#             label=vcr.class_names[c],
#         )

#     plt.legend()
#     plt.grid(True)
#     plt.show()
#     plt.savefig(fig_file)
#     plt.clf()

#     # if max_classes > 0:
#     #     fig_file = '%s/tsne_2d_n%d.pdf' % (output_path, max_classes)
#     #     index = class_ids < max_classes
#     #     plt.scatter(y[index,0], y[index,1], c=class_ids[index], marker='x')
#     #     plt.grid(True)
#     #     plt.show()
#     #     plt.savefig(fig_file)
#     #     plt.clf()

#     tsne = tsne_obj(3)
#     y = tsne.fit_transform(x)
#     if save_embed:
#         h5_file = "%s/embed_3d.h5" % output_path
#         hw = DWF.create(h5_file)
#         hw.write(vcr.u2c.key, y)

#     fig_file = "%s/tsne_3d.pdf" % (output_path)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     # ax.scatter(y[:,0], y[:,1], y[:,2], c=class_ids, marker='x')
#     for c in np.unique(class_ids):
#         idx = class_ids == c
#         ax.scatter(
#             y[idx, 0],
#             y[idx, 1],
#             y[idx, 2],
#             c=color_marker[c][0],
#             marker=color_marker[c][1],
#             label=vcr.class_names[c],
#         )

#     plt.grid(True)
#     plt.show()
#     plt.savefig(fig_file)
#     plt.clf()

#     # if max_classes > 0:
#     #     fig_file = '%s/tsne_3d_n%d.pdf' % (output_path, max_classes)
#     #     index = class_ids < max_classes
#     #     ax = fig.add_subplot(111, projection='3d')
#     #     ax.scatter(y[index,0], y[index,1], y[index,2], c=class_ids[index], marker='x')
#     #     plt.grid(True)
#     #     plt.show()
#     #     plt.savefig(fig_file)
#     #     plt.clf()

#     logging.info("Elapsed time: %.2f s." % (time.time() - t1))
