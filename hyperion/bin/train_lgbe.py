#!/usr/bin/env python
""" 
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba) 
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np.classifiers import LinearGBE as GBE
from hyperion.np.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    print_confusion_matrix,
)
from hyperion.np.transforms import LDA, PCA, CentWhiten, LNorm, TransformList
from hyperion.utils import PathLike, SegmentSet


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> None:
    """Compute and log classification metrics on training data.

    Args:
        y_true: Ground-truth class ids.
        y_pred: Predicted class ids.
        labels: Class label names/ids for confusion-matrix display.
    """
    acc = compute_accuracy(y_true, y_pred)
    logging.info("training acc: %.2f %%", acc * 100)
    logging.info("non-normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=False)
    print_confusion_matrix(C, labels)
    logging.info("normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=True)
    print_confusion_matrix(C * 100, labels)


def load_segments_and_feats(
    segments_file: PathLike, feats_file: PathLike
) -> Tuple[SegmentSet, np.ndarray]:
    """Load segment metadata and corresponding embeddings.

    Args:
        segments_file: Segment table file with utterance ids and labels.
        feats_file: Feature file containing embeddings indexed by segment ids.
    """
    logging.info("loading segments: %s feats: %s", segments_file, feats_file)
    segments = SegmentSet.load(segments_file)
    reader = DRF.create(feats_file)
    x = reader.read(segments["id"], squeeze=True)
    return segments, x


def load_data(
    segments_files: List[PathLike], feats_files: List[PathLike], class_name: str
) -> Tuple[SegmentSet, np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate segments/features from multiple inputs.

    Args:
        segments_files: List of segment table files.
        feats_files: List of embedding files aligned with ``segments_files``.
        class_name: Segment column name containing class labels.
    """
    assert len(segments_files) == len(feats_files)
    segments_list = []
    feats_list = []
    for segments_file, feats_file in zip(segments_files, feats_files):
        segments, feats = load_segments_and_feats(segments_file, feats_file)
        segments_list.append(segments)
        feats_list.append(feats)

    segments = SegmentSet.cat(segments_list)
    feats = np.concatenate(feats_list, axis=0)
    labels, y = np.unique(segments[class_name], return_inverse=True)
    return segments, feats, y, labels


def train_pca(
    x: np.ndarray, pca_lnorm: bool, pca_args: Dict[str, Any]
) -> Tuple[np.ndarray, Optional[LNorm], Optional[PCA]]:
    """Optionally train/apply PCA (and pre-PCA length norm).

    Args:
        x: Input embedding matrix.
        pca_lnorm: Whether to apply length normalization before PCA.
        pca_args: PCA constructor arguments.
    """
    pca_var_r = pca_args["pca_var_r"]
    logging.info("computing pca pca_var_r=%f", pca_var_r)
    pca = None
    pca_lnorm = None
    if pca_var_r < 1:
        if pca_lnorm:
            logging.info("LNorm before PCA")
            pca_lnorm = LNorm(name="pca_lnorm")
            x = pca_lnorm(x)

        pca = PCA(**pca_args)
        pca.fit(x)
        x = pca(x)
        logging.info("pca-dim=%d", x.shape[1])

    return x, pca_lnorm, pca


def train_lgbe(
    segments_files: List[PathLike],
    feats_files: List[PathLike],
    class_name: str,
    preproc_file: PathLike,
    lgbe_file: PathLike,
    pca: Dict[str, Any],
    lda: Dict[str, Any],
    lgbe: Dict[str, Any],
    pca_lnorm: bool,
    do_lda: bool,
    lda_lnorm: bool,
    lgbe_lnorm: bool,
    lgbe_center: bool,
    lgbe_whiten: bool,
) -> None:
    """Train preprocessing transforms and a Linear GBE backend.

    Args:
        segments_files: List of segment table files.
        feats_files: List of embedding files aligned with ``segments_files``.
        class_name: Segment column name containing class labels.
        preproc_file: Output path for preprocessing transform list.
        lgbe_file: Output path for trained Linear GBE model.
        pca: PCA configuration arguments.
        lda: LDA configuration arguments.
        lgbe: Linear GBE configuration arguments.
        pca_lnorm: Apply length normalization before PCA.
        do_lda: Enable LDA stage.
        lda_lnorm: Apply length normalization before LDA.
        lgbe_lnorm: Apply length normalization before GBE stage.
        lgbe_center: Center embeddings before GBE stage.
        lgbe_whiten: Whiten embeddings before GBE stage.
    """
    segments, x, y, labels = load_data(segments_files, feats_files, class_name)
    transform_list = []

    x, pca_lnorm, pca_model = train_pca(x, pca_lnorm, pca)
    if pca_lnorm is not None:
        transform_list.append(pca_lnorm)

    if pca_model is not None:
        transform_list.append(pca_model)

    if do_lda and x.shape[1] > lda["lda_dim"]:
        if lda_lnorm:
            logging.info("LNorm before LDA")
            t = LNorm(name="lda_lnorm")
            x = t(x)
            transform_list.append(t)

        logging.info("Training LDA")
        lda_model = LDA(**lda)
        lda_model.fit(x, y)
        x = lda_model(x)
        transform_list.append(lda_model)

    if lgbe_center or lgbe_whiten:
        if lgbe_lnorm:
            t = LNorm(update_mu=lgbe_center, update_T=lgbe_whiten, name="lgbe_lnorm")
        else:
            t = CentWhiten(update_mu=lgbe_center, update_T=lgbe_whiten, name="lgbe_cw")

        logging.info("Training Center/Whiten/LNorm")
        t.fit(x)
        logging.info("Center/Whiten/LNorm before GBE")
        x = t(x)
        transform_list.append(t)
    elif lgbe_lnorm:
        logging.info("LNorm before GBE")
        t = LNorm(name="lgbe_lnorm")
        x = t(x)
        transform_list.append(t)

    logging.info("Training GBE with args=%s", str(lgbe))
    gbe = GBE(labels=labels, **lgbe)
    gbe.fit(x, y)
    logging.info("trained GBE")
    scores = gbe.eval_linear(x)
    y_pred = np.argmax(scores, axis=-1)

    compute_metrics(y, y_pred, labels)

    logging.info("Saving Models")
    if len(transform_list) > 0:
        transform_list = TransformList(transform_list)
        transform_list.save(preproc_file)

    gbe.save(lgbe_file)


def main() -> None:
    """Parse CLI arguments and train LGBE/preprocessing models."""
    parser = ArgumentParser(
        description="Trains Linear Gaussian Back-end model and embedding preprocessor"
    )
    parser.add_argument(
        "--cfg",
        action=ActionConfigFile,
        help="Path to a configuration file.",
    )
    parser.add_argument(
        "--feats-files",
        nargs="+",
        required=True,
        help="Input embedding files used for training.",
    )
    parser.add_argument(
        "--segments-files",
        nargs="+",
        required=True,
        help="Input segment tables aligned with --feats-files.",
    )
    parser.add_argument(
        "--class-name",
        default="language",
        help="Segment column used as class label.",
    )
    parser.add_argument(
        "--preproc-file",
        required=True,
        help="Output file for preprocessing transform list.",
    )
    parser.add_argument(
        "--lgbe-file",
        required=True,
        help="Output file for trained Linear GBE model.",
    )
    PCA.add_class_args(parser, prefix="pca")
    LDA.add_class_args(parser, prefix="lda")
    GBE.add_class_args(parser, prefix="lgbe")
    parser.add_argument(
        "--pca-lnorm",
        default=False,
        action=ActionYesNo,
        help="Apply length normalization before PCA.",
    )
    parser.add_argument(
        "--lda-lnorm",
        default=False,
        action=ActionYesNo,
        help="Apply length normalization before LDA.",
    )
    parser.add_argument(
        "--do-lda",
        default=False,
        action=ActionYesNo,
        help="Enable LDA stage after PCA.",
    )
    parser.add_argument(
        "--lgbe-lnorm",
        default=True,
        action=ActionYesNo,
        help="Apply length normalization before the GBE backend.",
    )
    parser.add_argument(
        "--lgbe-center",
        default=True,
        action=ActionYesNo,
        help="Center embeddings before the GBE backend.",
    )
    parser.add_argument(
        "--lgbe-whiten",
        default=True,
        action=ActionYesNo,
        help="Whiten embeddings before the GBE backend.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="Verbosity level: 0=error, 1=warning, 2=info, 3=debug.",
    )
    args = parser.parse_args()
    config_logger(args.verbose)
    logging.debug(args)
    del args["verbose"]
    del args["cfg"]
    train_lgbe(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
