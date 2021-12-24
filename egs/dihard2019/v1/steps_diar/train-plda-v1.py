#!/usr/bin/env python
"""                                                                                                     
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)         
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
)
import time

import numpy as np
import pandas as pd

from hyperion.hyp_defs import config_logger
from hyperion.utils import Utt2Info

# from hyperion.helpers import VectorClassReader as VCR
from hyperion.transforms import TransformList, LDA, LNorm, PCA
from hyperion.helpers import PLDAFactory as F
from hyperion.io import RandomAccessDataReaderFactory as DRF

from numpy.linalg import matrix_rank


def load_train_list(train_list, inter_session):
    u2c = Utt2Info.load(train_list, sep=" ")
    if inter_session:
        class_ids = u2c.info
    else:
        class_ids = ["%s-%s" % (k, s) for k, s in zip(u2c.key, u2c.info)]

    _, class_ids = np.unique(u2c.info, return_inverse=True)
    return u2c.key, class_ids


# from memory_profiler import profile
# @profile
from pympler.classtracker import ClassTracker
from pympler import asizeof


def load_vectors(v_file, keys, class_ids, subsampling):

    x = []
    out_class_ids = []
    num_files = 0
    num_read = 0
    with DRF.create(v_file) as reader:
        # tracker = ClassTracker()
        # tracker.track_object(reader)
        for key, class_id in zip(keys, class_ids):
            x_i = reader.read(key)[0]
            if subsampling > 1:
                x_i = x_i[::subsampling, :].copy()

            if len(x_i) == 0:
                logging.info("read empty matrix from key={}".format(key, x_i.shape))
                continue

            x.append(x_i)
            num_read_i = x_i.shape[0]
            out_class_ids += [class_id] * num_read_i
            num_files += 1
            num_read += num_read_i
            logging.info(
                "read vectors from key={} with shape={}".format(key, x_i.shape)
            )
            # logging.info('read vectors from key={} with shape={} {} {}'.format(
            #     key, x_i.shape, np.sum(np.isnan(x_i)), matrix_rank(x_i)))
            logging.info("total read files={} vectors={}".format(num_files, num_read))
            assert not np.any(np.isnan(x_i))
            # if num_files > 60000:
            #     break
            # tracker.create_snapshot()

            # logging.info('1 {}'.format(asizeof.asized(x, detail=1).format()))
            # logging.info('2 {}'.format(asizeof.asized(x_i, detail=1).format()))
            # logging.info('3 {} {} {}'.format(x_i.shape[0]*x_i.shape[1]*x_i.itemsize, x_i.nbytes, x_i.size*x_i.itemsize))
            # logging.info('4 {} {} {}'.format(xb.shape[0]*xb.shape[1]*xb.itemsize, xb.nbytes, xb.size*xb.itemsize))

        # tracker.stats.print_summary()
    x = np.concatenate(tuple(x), axis=0)
    out_class_ids = np.asarray(out_class_ids, dtype=np.int)
    return x, out_class_ids


def train_plda(
    v_file,
    train_list,
    lda_dim,
    plda_type,
    y_dim,
    z_dim,
    epochs,
    ml_md,
    md_epochs,
    output_path,
    inter_session,
    subsampling,
    **kwargs
):

    keys, class_ids = load_train_list(train_list, inter_session)
    logging.info(
        "reading {} utts with {} classes".format(len(keys), np.max(class_ids) + 1)
    )
    x, class_ids = load_vectors(v_file, keys, class_ids, subsampling)
    logging.info("read x={} with num_classes={}".format(x.shape, np.max(class_ids) + 1))

    t1 = time.time()
    # logging.info('%d %d' % (np.sum(np.isnan(x)), np.sum(np.isinf(x))))
    rank = PCA.get_pca_dim_for_var_ratio(x, 1)
    logging.info("x-rank=%d" % (rank))
    pca = None
    if rank < x.shape[1]:
        # do PCA if rank of x is smaller than its dimension
        logging.info("PCA rank=%d" % (rank))
        pca = PCA(pca_dim=rank, name="pca")
        pca.fit(x)
        x = pca.predict(x)
        if lda_dim > rank:
            lda_dim = rank
        if y_dim > rank:
            y_dim = rank

    # Train LDA
    lda = LDA(lda_dim=lda_dim, name="lda")
    lda.fit(x, class_ids)

    x = lda.predict(x)
    logging.info("PCA-LDA Elapsed time: %.2f s." % (time.time() - t1))

    # Train centering and whitening
    t1 = time.time()
    lnorm = LNorm(name="lnorm")
    lnorm.fit(x)

    x = lnorm.predict(x)
    logging.info("LNorm Elapsed time: %.2f s." % (time.time() - t1))

    # Train PLDA
    t1 = time.time()
    plda = F.create_plda(plda_type, y_dim=y_dim, z_dim=z_dim, name="plda")
    elbo, elbo_norm = plda.fit(
        x, class_ids, epochs=epochs, ml_md=ml_md, md_epochs=md_epochs
    )

    logging.info("PLDA Elapsed time: %.2f s." % (time.time() - t1))

    # Save models
    if pca is None:
        preproc = TransformList([lda, lnorm])
    else:
        preproc = TransformList([pca, lda, lnorm])

    if not os.path.exists(output_path):
        os.makedirs(ouput_path)

    preproc.save(output_path + "/lda_lnorm.h5")
    plda.save(output_path + "/plda.h5")

    print(epochs, elbo.shape)
    pd.DataFrame(
        {
            "epochs": np.arange(1, epochs + 1, dtype=np.int),
            "elbo": elbo,
            "elbo_per_sample": elbo_norm,
        }
    ).to_csv(output_path + "/elbo.csv", index=False)


if __name__ == "__main__":

    parser = ArgumentParser(description="Train LDA/PLDA back-end for diarization")

    parser.add_argument("--v-file", required=True, help="embedding read specifier")
    parser.add_argument(
        "--train-list", required=True, help="train list utterance spkid"
    )

    F.add_argparse_train_args(parser)

    parser.add_argument("--output-path", required=True)
    parser.add_argument("--lda-dim", type=int, default=None)
    parser.add_argument(
        "--inter-session",
        default=False,
        action="store_true",
        help=(
            "if True, model inter-session variability, "
            "if False, model intra-session variability"
        ),
    )
    parser.add_argument(
        "--subsampling",
        default=1,
        type=int,
        help=("subsamples the embeddings to reduce memory and " "computing cost"),
    )

    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    train_plda(**vars(args))
