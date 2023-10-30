#!/bin/env python
"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path
from jsonargparse import ArgumentParser, namespace_to_dict
import numpy as np

from hyperion.hyp_defs import config_logger
from hyperion.utils import SegmentSet


def split_dev(segs_file, output_dir, num_folds, verbose):
    config_logger(verbose)
    segs = SegmentSet.load(segs_file)
    assert "subclass_idx" in segs
    class_ids = segs["class_id"]
    _, class_idx = np.unique(class_ids, return_inverse=True)
    logging.info("splitting segments into %d folds", num_folds)
    folds = [[] for i in range(num_folds)]
    for c in range(np.max(class_idx) + 1):
        c_idx = class_idx == c
        subclass_idx = segs.loc[c_idx, "subclass_idx"]
        num_c = len(subclass_idx)
        num_c_pf = num_c / num_folds
        _, counts = np.unique(subclass_idx, return_counts=True)
        acc_counts = np.cumsum(counts)
        logging.info(
            f"class {c} subclass-counts={counts}, subclass-acc-counts={acc_counts}"
        )
        c_idx = np.nonzero(c_idx)[0]
        first = 0
        for f in range(num_folds):
            if f < num_folds - 1:
                last = np.argmin(np.abs(acc_counts - (f + 1) * num_c_pf))
            else:
                last = np.max(subclass_idx)
            f_idx = np.logical_and(subclass_idx >= first, subclass_idx <= last)
            folds[f].extend(c_idx[f_idx])
            logging.info(
                (
                    f"class {c} fold {f} add {np.sum(f_idx)} samples,"
                    f"accum {len(folds[f])} samples, "
                    f"first-subclass={first}, last-subclass={last}"
                )
            )
            first = last + 1

    output_dir = Path(output_dir)
    for f in range(num_folds):
        logging.info(
            "fold %d, train-samples=%d test-samples=%d",
            f,
            len(segs) - len(folds[f]),
            len(folds[f]),
        )
        f_dir = output_dir / f"fold_{f}"
        f_dir.mkdir(parents=True, exist_ok=True)
        mask = np.zeros((len(segs),), dtype=bool)
        mask[folds[f]] = True
        segs_test = SegmentSet(segs.loc[mask])
        segs_test.save(f_dir / "test_segments.csv")
        segs_train = SegmentSet(segs.loc[~mask])
        segs_train.save(f_dir / "train_segments.csv")


if __name__ == "__main__":

    parser = ArgumentParser(description="Splits LRE22 into folds")
    parser.add_argument(
        "--segs-file", required=True, help="Segments file with subclass_idx column",
    )
    parser.add_argument("--output-dir", required=True, help="output path")
    parser.add_argument("--num-folds", default=2, type=int, help="number of folds")
    parser.add_argument("-v", "--verbose", default=1, choices=[0, 1, 2, 3], type=int)

    args = parser.parse_args()
    split_dev(**namespace_to_dict(args))
