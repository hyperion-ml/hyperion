"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging
import numpy as np

from hyperion.utils.utt2info import Utt2Info
from hyperion.utils.math import softmax
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.transforms import LNorm
from hyperion.clustering import AHC


def lnorm(x):
    mx = np.sqrt(np.sum(x ** 2, axis=1, keepdims=True)) + 1e-10
    return x / mx


def cosine_scr(x1, x2):

    # t = LNorm()
    # x1 = t.predict(x1)
    # x2 = t.predict(x2)
    x1 = lnorm(x1)
    x2 = lnorm(x2)
    return np.dot(x1, x2.T)


def fill_missing_ref_with_facedet_avg(x_ref, x_e, seg_names):

    assert len(x_ref) == len(x_e)
    # get embed dim
    for i in range(len(x_e)):
        if x_e[i].shape[0] > 0:
            x_dim = x_e[i].shape[1]
            break

    for i in range(len(x_ref)):
        if x_ref[i].shape[0] == 0:
            if x_e[i].shape[0] > 0:
                logging.warning(
                    "Empty reference for enroll %s, we put the average of faces in enroll file"
                    % (seg_names[i])
                )
                x_ref[i] = np.mean(x_e[i], axis=0, keepdims=True)
            else:
                logging.warning(
                    "Empty reference for enroll %s, we use zero vector, no faces were detected in enroll file "
                    % (seg_names[i])
                )
                x_ref[i] = np.zeros((1, x_dim))

    return x_ref


def fill_missing_test_with_zero(x_t, seg_names):

    # get embed dim
    for i in range(len(x_t)):
        if x_t[i].shape[0] > 0:
            x_dim = x_t[i].shape[1]
            break

    for i in range(len(x_t)):
        if x_t[i].shape[0] == 0:
            logging.warning("Empty test %s, we use zero vector " % (seg_names[i]))
            x_t[i] = np.zeros((1, x_dim))

    return x_t


def concat_embed_matrices(x):

    seg_idx = []
    for i in range(len(x)):
        seg_idx_i = i * np.ones((x[i].shape[0],), dtype=np.int)
        seg_idx.append(seg_idx_i)

    seg_idx = np.concatenate(tuple(seg_idx))
    x = np.concatenate(tuple(x), axis=0)

    return x, seg_idx


def max_combine_scores_1vsM(scores_in, test_idx):
    num_test = np.max(test_idx) + 1
    scores_out = np.zeros((scores_in.shape[0], num_test))
    for j in range(num_test):
        idx = test_idx == j
        scores_j = scores_in[:, idx]
        scores_j = np.max(scores_j, axis=1)
        scores_out[:, j] = scores_j

    return scores_out


def max_combine_scores_NvsM(scores, enr_idx, test_idx):

    max_scores_cols = max_combine_scores_1vsM(scores, test_idx)
    max_scores_trans = max_combine_scores_1vsM(max_scores_cols.T, enr_idx)
    return max_scores_trans.T


def read_cohort(v_file, coh_list):
    r = DRF.create(v_file)
    coh = Utt2Info.load(coh_list, sep=" ")
    x = r.read(coh.key, squeeze=False)
    for i in range(len(x)):
        print(x[i].shape)
    x = np.concatenate(tuple(x), axis=0)
    return x


def compute_avg_per_vid(x):

    x_avg = []
    for i in range(len(x)):
        x_avg.append(np.mean(x[i], axis=0, keepdims=True))

    return x_avg


def compute_median_per_vid(x):

    x_avg = []
    for i in range(len(x)):
        x_avg.append(np.median(x[i], axis=0, keepdims=True))

    return x_avg


def cluster_embeds_ahc(x, thr):

    ahc = AHC(method="average", metric="llr")
    x_clusters = []
    for i in range(len(x)):
        x_i = x[i]
        if x_i.shape[0] == 1:
            x_clusters.append(x_i)
            continue

        scores = cosine_scr(x_i, x_i)
        ahc.fit(scores)
        class_ids = ahc.get_flat_clusters(thr, criterion="threshold")
        x_dim = x_i.shape[1]
        num_classes = np.max(class_ids) + 1
        logging.info("AHC file %d from %d -> %d" % (i, x_i.shape[0], num_classes))
        x_clusters_i = np.zeros((num_classes, x_dim))
        for j in range(num_classes):
            idx = class_ids == j
            x_clusters_i[j] = np.mean(x_i[idx], axis=0)

        x_clusters.append(x_clusters_i)

    return x_clusters


def compute_self_att_embeds(x, a):

    x_att = []
    for i in range(len(x)):
        x_i = x[i]
        if x_i.shape[0] == 1:
            x_att.append(x_i)
            continue

        scores = a * cosine_scr(x_i, x_i)
        p_att = softmax(scores, axis=1)
        x_att_i = np.dot(p_att, x_i)
        x_att.append(x_att_i)

    return x_att


def compute_att_test_embeds(x_e, x_t, a):

    x_att = []
    for i in range(len(x_t)):
        x_i = x_t[i]
        if x_i.shape[0] == 1:
            x_att.append(x_i)
            continue

        print(x_i.shape)
        scores = a * cosine_scr(x_e, x_i)
        p_att = softmax(scores, axis=1)
        x_att_i = np.dot(p_att, x_i)
        x_att.append(x_att_i)

    return x_att
