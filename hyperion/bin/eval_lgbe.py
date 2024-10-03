#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

"""
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np import NPModel
from hyperion.np.classifiers import LinearGBE as GBE
from hyperion.np.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    print_confusion_matrix,
)
from hyperion.np.transforms import TransformList
from hyperion.utils import SegmentSet


def load_data(segments_file, feats_file, class_name):
    logging.info("loading data")
    segments = SegmentSet.load(segments_file)
    reader = DRF.create(feats_file)
    x = reader.read(segments["id"], squeeze=True)
    if class_name in segments:
        y = segments[class_name]
    else:
        y = None

    return segments, x, y


def compute_metrics(y_true, y_pred, labels):

    acc = compute_accuracy(y_true, y_pred)
    logging.info("test acc: %.2f %%", acc * 100)
    logging.info("non-normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=False)
    print_confusion_matrix(C, labels)
    logging.info("normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=True)
    print_confusion_matrix(C * 100, labels)


def eval_lgbe(
    segments_file,
    feats_file,
    class_name,
    lgbe,
    lgbe_file,
    preproc_file,
    score_file,
):

    logging.info("loading data")
    segments, x, class_ids = load_data(segments_file, feats_file, class_name)
    logging.info("loaded %d samples", x.shape[0])

    if preproc_file is not None:
        logging.info("Loading Preprocessor %s", preproc_file)
        preprocessor = TransformList.load(preproc_file)
        logging.info("applies proprocessing transform")
        x = preprocessor(x)

    logging.info("loading GBE file %s", lgbe_file)
    gbe_model = GBE.load(lgbe_file)
    logging.info("eval GBE with args=%s", str(lgbe))
    scores = gbe_model(x, **lgbe)

    if class_ids is not None:
        y_true = np.asarray([gbe_model.labels.index(l) for l in class_ids])
        # labels, y_true = np.unique(class_ids, return_inverse=True)
        y_pred = np.argmax(scores, axis=-1)
        compute_metrics(y_true, y_pred, gbe_model.labels)

    logging.info("Saving scores to %s", score_file)
    score_table = {"id": segments["id"].values}
    for i, key in enumerate(gbe_model.labels):
        score_table[key] = scores[:, i]

    score_table = pd.DataFrame(score_table)
    score_file = Path(score_file)
    output_dir = score_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    sep = "\t" if score_file.suffix == ".tsv" else ","
    score_table.to_csv(score_file, sep=sep, index=False)


def main():

    parser = ArgumentParser(
        description="Evals linear Gaussian back-end",
    )

    parser.add_argument("--feats-file", required=True)
    parser.add_argument("--segments-file", required=True)
    GBE.add_eval_args(parser, prefix="lgbe")
    parser.add_argument("--class-name", default="language")
    parser.add_argument("--preproc-file", default=None)
    parser.add_argument("--lgbe-file", required=True)
    parser.add_argument("--score-file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_lgbe(**namespace_to_dict(args))
