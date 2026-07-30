#!/usr/bin/env python
"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from hyperion.np import HyperNPModel
from hyperion.np.classifiers import LinearGBE as GBE
from hyperion.np.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    print_confusion_matrix,
)
from hyperion.np.transforms import TransformList
from hyperion.utils import SegmentSet
from hyperion.utils.misc import PathLike


def load_data(
    segments_file: PathLike, feats_file: PathLike, class_name: str
) -> Tuple[SegmentSet, np.ndarray, Optional[pd.Series]]:
    """Load segment metadata, embeddings, and optional class labels."""
    logging.info("loading data")
    segments = SegmentSet.load(segments_file)
    reader = DRF.create(feats_file)
    x = reader.read(segments["id"], squeeze=True)
    if class_name in segments:
        y = segments[class_name]
    else:
        y = None

    return segments, x, y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> None:
    """Compute and log classification accuracy and confusion matrices."""

    acc = compute_accuracy(y_true, y_pred)
    logging.info("test acc: %.2f %%", acc * 100)
    logging.info("non-normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=False)
    print_confusion_matrix(C, labels)
    logging.info("normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=True)
    print_confusion_matrix(C * 100, labels)


def eval_lgbe(
    segments_file: PathLike,
    feats_file: PathLike,
    class_name: str,
    lgbe: Dict[str, Any],
    lgbe_file: PathLike,
    preproc_file: Optional[PathLike],
    score_file: PathLike,
) -> None:
    """Evaluate a Linear GBE model and write class-score table."""

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


def main() -> None:
    """Parse CLI arguments and run Linear GBE evaluation."""

    parser = ArgumentParser(
        description="Evals linear Gaussian back-end",
    )

    parser.add_argument(
        "--feats-file", required=True, help="input embedding/feature file"
    )
    parser.add_argument(
        "--segments-file",
        required=True,
        help="SegmentSet file with sample ids and optional labels",
    )
    GBE.add_eval_args(parser, prefix="lgbe")
    parser.add_argument(
        "--class-name",
        default="language",
        help="segments-file column containing ground-truth class labels",
    )
    parser.add_argument(
        "--preproc-file",
        default=None,
        help="optional preprocessing transform list applied before scoring",
    )
    parser.add_argument("--lgbe-file", required=True, help="path to trained GBE model")
    parser.add_argument(
        "--score-file",
        required=True,
        help="output score table path (.csv or .tsv)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="verbosity level (0=error, 1=warning, 2=info, 3=debug)",
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_lgbe(**namespace_to_dict(args))
