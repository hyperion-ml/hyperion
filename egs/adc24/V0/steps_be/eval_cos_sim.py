#!/usr/bin/env python
"""
 Dialect Identification Evaluation Script
"""

import sys
import os
import logging
from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import jiwer

from hyperion.hyp_defs import config_logger
from hyperion.utils import SegmentSet
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np.transforms import TransformList


def compute_metrics(y_true, y_pred, labels):
    acc = np.mean(y_true == y_pred)
    logging.info("Test accuracy: %.2f %%", acc * 100)
    logging.info("Non-normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, labels)
    print_confusion_matrix(C, labels)
    logging.info("Normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, labels, normalize=True)
    print_confusion_matrix(C * 100, labels, fmt=".2f")


def compute_confusion_matrix(y_true, y_pred, labels, normalize=False):
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


def print_confusion_matrix(cm, labels, fmt=".2f"):
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print(df_cm.to_string(float_format=fmt))


def compute_wer(y_true_texts, y_pred_texts):
    wer = jiwer.wer(y_true_texts, y_pred_texts)
    logging.info("Word Error Rate (WER): %.2f %%", wer * 100)
    return wer


def compute_cavg(scores, y_true, num_classes):
    C_avg = 0
    for i in range(num_classes):
        P_target = np.mean(y_true == i)
        P_non_target = 1 - P_target

        target_scores = scores[y_true == i, i]
        non_target_scores = scores[y_true != i, i]

        thresholds = np.sort(np.concatenate([target_scores, non_target_scores]))
        C_det = np.inf

        for threshold in thresholds:
            P_miss = np.mean(target_scores < threshold)
            P_fa = np.mean(non_target_scores >= threshold)

            C_det_tmp = P_target * P_miss + P_non_target * P_fa
            if C_det_tmp < C_det:
                C_det = C_det_tmp

        C_avg += C_det

    C_avg /= num_classes
    logging.info("C_avg: %.4f", C_avg)
    return C_avg


def train_be(
    v_file,
    trial_list,
    class_name,
    has_labels,
    model_dir,
    score_file,
    verbose,
):
    config_logger(verbose)
    model_dir = Path(model_dir)
    output_dir = Path(score_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Loading data")
    segs = SegmentSet.load(trial_list)
    reader = DRF.create(v_file)
    x = reader.read(segs["id"], squeeze=True)
    del reader
    logging.info("Loaded %d samples", x.shape[0])

    trans_file = model_dir / "preproc.h5"
    if trans_file.is_file():
        logging.info("Loading transform file %s", trans_file)
        trans = TransformList.load(trans_file)
        logging.info("Applying transform")
        x = trans(x)

    logging.info("Calculating cosine similarity")
    scores = cosine_similarity(x)

    if has_labels:
        logging.info("Calculating haslabels similarity")
        class_ids = segs[class_name]
        labels, y_true = np.unique(class_ids, return_inverse=True)
        y_pred = np.argmax(scores[:, :len(labels)], axis=-1)
        logging.info("y_pred : %s ", y_pred)  # Ensure the shape matches the number of labels
        compute_metrics(y_true, y_pred, labels)
        compute_cavg(scores, y_true, len(labels))

        # Calculate WER if text data is available
        if 'text' in segs:
            y_true_texts = [segs['text'][i] for i in y_true]
            y_pred_texts = [segs['text'][i] for i in y_pred]
            compute_wer(y_true_texts, y_pred_texts)

    logging.info("Saving scores to %s", score_file)
    score_table = {"segmentid": segs["id"]}
    for i, key in enumerate(labels):
        score_table[key] = scores[:, i]

    score_table = pd.DataFrame(score_table)
    score_table.to_csv(score_file, sep="\t", index=False)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Evaluate cosine similarity for dialect identification",
    )

    parser.add_argument("--v-file", required=True, help="Path to the file containing x-vectors or features")
    parser.add_argument("--trial-list", required=True, help="Path to the file containing trial list (segment set)")
    parser.add_argument("--class-name", default="class_id", help="Name of the class label in the segment set")
    parser.add_argument("--has-labels", default=False, action=ActionYesNo, help="Indicate if labels are available")
    parser.add_argument("--model-dir", required=True, help="Directory containing model and transform files")
    parser.add_argument("--score-file", required=True, help="Path to the file for saving scores")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int, help="Verbosity level"
    )

    args = parser.parse_args()
    train_be(**namespace_to_dict(args))
