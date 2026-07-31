#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import os
import logging
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

from hyperion.hyp_defs import config_logger
from hyperion.utils import SegmentSet
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.helpers import VectorClassReader as VCR
from hyperion.np.transforms import TransformList
from hyperion.np.classifiers import LinearGBE as GBE
from hyperion.np.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    print_confusion_matrix,
)


def compute_metrics(y_true, y_pred, labels):

    acc = compute_accuracy(y_true, y_pred)
    logging.info("test acc: %.2f %%", acc * 100)
    logging.info("non-normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=False)
    print_confusion_matrix(C, labels)
    logging.info("normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=True)
    print_confusion_matrix(C * 100, labels)


def train_be(
    v_file,
    trial_list,
    class_name,
    has_labels,
    gbe,
    model_dir,
    score_file,
    verbose,
):
    config_logger(verbose)
    model_dir = Path(model_dir)
    output_dir = Path(score_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("loading data")
    segs = SegmentSet.load(trial_list)
    reader = DRF.create(v_file)
    x = reader.read(segs["id"], squeeze=True)
    del reader
    logging.info("loaded %d samples", x.shape[0])

    trans_file = model_dir / "transforms.h5"
    if trans_file.is_file():
        logging.info("loading transform file %s", trans_file)
        trans = TransformList.load(trans_file)
        logging.info("applies transform")
        x = trans(x)

    gbe_file = model_dir / "model_gbe.h5"
    logging.info("loading GBE file %s", gbe_file)
    gbe_model = GBE.load(gbe_file)
    logging.info("GBE args=%s", str(gbe))
    logging.info("evals GBE")
    scores = gbe_model(x, **gbe)

    if has_labels:
        class_ids = segs[class_name]
        y_true = np.asarray([gbe_model.labels.index(l) for l in class_ids])
        # labels, y_true = np.unique(class_ids, return_inverse=True)
        y_pred = np.argmax(scores, axis=-1)
        compute_metrics(y_true, y_pred, gbe_model.labels)

    logging.info("Saving scores to %s", score_file)
    score_table = {"segmentid": segs["id"]}
    for i, key in enumerate(gbe_model.labels):
        score_table[key] = scores[:, i]

    score_table = pd.DataFrame(score_table)
    score_table.to_csv(score_file, sep="\t", index=False)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Evals linear GBE",
    )

    parser.add_argument("--v-file", required=True)
    parser.add_argument("--trial-list", required=True)
    GBE.add_eval_args(parser, prefix="gbe")
    parser.add_argument("--class-name", default="class_id")
    parser.add_argument("--has-labels", default=False, action=ActionYesNo)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--score-file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    train_be(**namespace_to_dict(args))
