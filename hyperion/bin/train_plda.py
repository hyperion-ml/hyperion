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
from hyperion.np.pdfs import PLDAFactory
from hyperion.np.transforms import LDA, PCA, CentWhiten, LNorm, TransformList
from hyperion.utils import PathLike, SegmentSet


def load_segments_and_feats(
    segments_file: PathLike, feats_file: PathLike
) -> Tuple[SegmentSet, np.ndarray]:
    """Load segments and corresponding embeddings, removing NaN rows.

    Args:
        segments_file: Segment table file with utterance ids and labels.
        feats_file: Feature file containing embeddings indexed by segment ids.
    """
    logging.info("loading segments: %s feats: %s", segments_file, feats_file)
    segments = SegmentSet.load(segments_file)
    reader = DRF.create(feats_file)
    x = reader.read(segments["id"], squeeze=True)
    is_nan = np.any(np.isnan(x), axis=1)
    num_nan = np.sum(is_nan)
    if num_nan > 0:
        logging.info(
            "Found %d NaN features: %s", num_nan, str(segments.loc[is_nan, "id"])
        )
        x = x[~is_nan]
        segments = SegmentSet(segments.loc[~is_nan])

    return segments, x


def load_data(
    segments_files: List[PathLike], feats_files: List[PathLike], class_name: str
) -> Tuple[SegmentSet, np.ndarray, np.ndarray]:
    """Load and concatenate data from multiple segment/feature sources.

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
    _, y = np.unique(segments[class_name], return_inverse=True)
    return segments, feats, y


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
    if pca_var_r is not None and pca_var_r < 1:
        if pca_lnorm:
            logging.info("LNorm before PCA")
            pca_lnorm = LNorm(name="pca_lnorm")
            x = pca_lnorm(x)

        pca = PCA(**pca_args)
        pca.fit(x)
        x = pca(x)
        logging.info("pca-dim=%d", x.shape[1])

    return x, pca_lnorm, pca


def train_plda(
    segments_files: List[PathLike],
    feats_files: List[PathLike],
    class_name: str,
    preproc_file: PathLike,
    plda_file: PathLike,
    pca: Dict[str, Any],
    lda: Dict[str, Any],
    plda: Dict[str, Any],
    pca_lnorm: bool,
    do_lda: bool,
    lda_lnorm: bool,
    plda_lnorm: bool,
    plda_center: bool,
    plda_whiten: bool,
) -> None:
    """Train preprocessing transforms and a PLDA backend.

    Args:
        segments_files: List of segment table files.
        feats_files: List of embedding files aligned with ``segments_files``.
        class_name: Segment column name containing class labels.
        preproc_file: Output path for preprocessing transform list.
        plda_file: Output path for trained PLDA model.
        pca: PCA configuration arguments.
        lda: LDA configuration arguments.
        plda: PLDA configuration arguments.
        pca_lnorm: Apply length normalization before PCA.
        do_lda: Enable LDA stage.
        lda_lnorm: Apply length normalization before LDA.
        plda_lnorm: Apply length normalization before PLDA stage.
        plda_center: Center embeddings before PLDA stage.
        plda_whiten: Whiten embeddings before PLDA stage.
    """
    segments, x, y = load_data(segments_files, feats_files, class_name)
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

    if plda_center or plda_whiten:
        if plda_lnorm:
            t = LNorm(update_mu=plda_center, update_T=plda_whiten, name="plda_lnorm")
        else:
            t = CentWhiten(update_mu=plda_center, update_T=plda_whiten, name="plda_cw")

        logging.info("Training Center/Whiten/LNorm")
        t.fit(x)
        logging.info("Center/Whiten/LNorm before PLDA")
        x = t(x)
        transform_list.append(t)
    elif plda_lnorm:
        logging.info("LNorm before PLDA")
        t = LNorm(name="plda_lnorm")
        x = t(x)
        transform_list.append(t)

    logging.info("Training PLDA")
    plda["y_dim"] = min(x.shape[1], plda["y_dim"])
    plda = PLDAFactory.create(**plda)
    elbo, elbo_norm = plda.fit(x, y)

    logging.info("Saving Models")
    if len(transform_list) > 0:
        transform_list = TransformList(transform_list)
        transform_list.save(preproc_file)

    plda.save(plda_file)
    loss_file = Path(plda_file).with_suffix(".csv")
    loss_df = pd.DataFrame(
        {"epoch": np.arange(1, len(elbo) + 1), "elbo": elbo, "elbo_norm": elbo_norm}
    )
    loss_df.to_csv(loss_file, index=False)


def main() -> None:
    """Parse CLI arguments and train PLDA/preprocessing models."""
    parser = ArgumentParser(description="Trains PLDA model and embedding preprocessor")
    parser.add_argument(
        "--cfg",
        action=ActionConfigFile,
        help="Path to a configuration file.",
    )
    parser.add_argument(
        "--feats-files",
        required=True,
        nargs="+",
        help="Input embedding files used for training.",
    )
    parser.add_argument(
        "--segments-files",
        required=True,
        nargs="+",
        help="Input segment tables aligned with --feats-files.",
    )
    parser.add_argument(
        "--class-name",
        default="speaker",
        help="Segment column used as class label.",
    )
    parser.add_argument(
        "--preproc-file",
        required=True,
        help="Output file for preprocessing transform list.",
    )
    parser.add_argument(
        "--plda-file",
        required=True,
        help="Output file for trained PLDA model.",
    )
    PCA.add_class_args(parser, prefix="pca")
    LDA.add_class_args(parser, prefix="lda")
    PLDAFactory.add_class_args(parser, prefix="plda")
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
        "--plda-lnorm",
        default=True,
        action=ActionYesNo,
        help="Apply length normalization before the PLDA backend.",
    )
    parser.add_argument(
        "--plda-center",
        default=True,
        action=ActionYesNo,
        help="Center embeddings before the PLDA backend.",
    )
    parser.add_argument(
        "--plda-whiten",
        default=True,
        action=ActionYesNo,
        help="Whiten embeddings before the PLDA backend.",
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
    train_plda(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
