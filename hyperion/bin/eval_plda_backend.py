#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

"""
import logging
import time
from pathlib import Path

import numpy as np
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np import NPModel
from hyperion.np.pdfs import PLDAFactory, PLDALLRNvsMMethod
from hyperion.np.score_norm import AdaptSNorm
from hyperion.np.transforms import LNorm, TransformList
from hyperion.utils import EnrollmentMap, SegmentSet, TrialKey, TrialNdx, TrialScores
from hyperion.utils.math_funcs import cosine_scoring


def load_trial_data(
    enroll_map_file,
    ndx_file,
    enroll_feats_file,
    feats_file,
    enroll_part_idx,
    num_enroll_parts,
    test_part_idx,
    num_test_parts,
):
    test_feats_reader = DRF.create(feats_file)
    if enroll_feats_file is not None and enroll_feats_file != feats_file:
        enroll_feats_reader = DRF.create(enroll_feats_file)
    else:
        enroll_feats_reader = test_feats_reader

    enroll_map = EnrollmentMap.load(enroll_map_file)
    try:
        ndx = TrialNdx.load(ndx_file)
    except:
        ndx = TrialKey.load(ndx_file).to_ndx()

    if num_enroll_parts > 1 or num_test_parts > 1:
        ndx = ndx.split(
            enroll_part_idx, num_enroll_parts, test_part_idx, num_test_parts
        )

    enroll_map = enroll_map.filter(items=ndx.model_set)
    x_e = enroll_feats_reader.read(enroll_map["segmentid"], squeeze=True)
    x_t = test_feats_reader.read(ndx.seg_set, squeeze=True)
    return enroll_map, ndx, x_e, x_t


def load_cohort_data(segments_file, feats_file):
    segments = SegmentSet.load(segments_file)
    feats_reader = DRF.create(feats_file)
    x = feats_reader.read(segments["id"], squeeze=True)
    return segments, x


def eval_backend(
    enroll_map_file,
    ndx_file,
    enroll_feats_file,
    feats_file,
    preproc_file,
    plda_file,
    llr_method,
    score_file,
    enroll_part_idx,
    num_enroll_parts,
    test_part_idx,
    num_test_parts,
    cohort_segments_file,
    cohort_feats_file,
    cohort_nbest,
    avg_cohort_by,
):
    logging.info("loading data")
    enroll_map, ndx, x_e, x_t = load_trial_data(
        enroll_map_file,
        ndx_file,
        enroll_feats_file,
        feats_file,
        enroll_part_idx,
        num_enroll_parts,
        test_part_idx,
        num_test_parts,
    )
    enroll_set, enroll_ids = np.unique(enroll_map["id"], return_inverse=True)
    if len(enroll_set) == np.max(enroll_ids) + 1:
        is_Nvs1 = False
    else:
        is_Nvs1 = True

    t1 = time.time()

    if preproc_file is not None:
        logging.info("Loading Preprocessor")
        preprocessor = TransformList.load(preproc_file)
        x_e = preprocessor(x_e)
        x_t = preprocessor(x_t)
        if llr_method == PLDALLRNvsMMethod.vavg and isinstance(
            preprocessor.transforms[-1], LNorm
        ):
            llr_method = PLDALLRNvsMMethod.lnorm_vavg

    assert llr_method == PLDALLRNvsMMethod.lnorm_vavg, preprocessor.transforms
    logging.info("Loading PLDA model")
    plda_model = NPModel.auto_load(plda_file)
    logging.info("computing score")
    if is_Nvs1:
        scores = plda_model.llr_Nvs1(x_e, x_t, ids1=enroll_ids, method=llr_method)
    else:
        scores = plda_model.llr_1vs1(x_e, x_t)

    dt = time.time() - t1
    num_trials = scores.shape[0] * scores.shape[1]
    logging.info(
        "scoring elapsed time: %.2f s. elapsed time per trial: %.2f ms.",
        dt,
        dt / num_trials * 1000,
    )

    if cohort_segments_file is not None:
        t1 = time.time()
        cohort_segments, x_coh = load_cohort_data(
            cohort_segments_file, cohort_feats_file
        )
        if preproc_file is not None:
            x_coh = preprocessor(x_coh)

        if avg_cohort_by is not None:
            cohort_class = cohort_segments[avg_cohort_by]
            _, cohort_ids = np.unique(cohort_class, return_inverse=True)
        else:
            cohort_ids = None

        logging.info("computing enroll vs cohort")
        scores_enr_coh = plda_model.llr_NvsM(
            x_e, x_coh, ids1=enroll_ids, ids2=cohort_ids, method=llr_method
        )
        logging.info("computing cohort vs test")
        scores_coh_test = plda_model.lrr_Nvs1(
            x_coh, x_t, ids1=cohort_ids, method=llr_method
        )
        snorm = AdaptSNorm(cohort_nbest)
        scores = snorm(scores, scores_coh_test, scores_enr_coh)
        dt = time.time() - t1
        logging.info(
            "s-norm elapsed time: %.2f s. elapsed time per trial: %.2f ms.",
            dt,
            dt / num_trials * 1000,
        )

    if num_enroll_parts > 1 or num_test_parts > 1:
        score_file = Path(score_file)
        new_suffix = f".{enroll_part_idx}.{test_part_idx}{score_file.suffix}"
        score_file = score_file.with_suffix(new_suffix)

    logging.info("saving scores to %s", score_file)
    # sort scores rows to match the ndx model_set order
    sort_idx = [np.nonzero(enroll_set == e)[0][0] for e in ndx.model_set]
    scores = scores[sort_idx]
    scores = TrialScores(ndx.model_set, ndx.seg_set, scores, ndx.trial_mask)
    scores.save(score_file)


def main():
    parser = ArgumentParser(description="Eval PLDA LLR with optional AS-Norm")

    parser.add_argument("--enroll-feats-file", default=None)
    parser.add_argument("--feats-file", required=True)
    parser.add_argument("--ndx-file", required=True)
    parser.add_argument("--enroll-map-file", required=True)
    parser.add_argument("--preproc-file", default=None)
    parser.add_argument("--plda-file", required=True)
    parser.add_argument(
        "--llr-method",
        default=PLDALLRNvsMMethod.vavg,
        choices=PLDALLRNvsMMethod.choices(),
    )
    parser.add_argument("--cohort-segments-file", default=None)
    parser.add_argument("--cohort-feats-file", default=None)
    parser.add_argument("--cohort-nbest", type=int, default=1000)
    parser.add_argument(
        "--avg-cohort-by",
        default=None,
        help="segments file column to average vectors from same class class",
    )
    parser.add_argument("--score-file", required=True)
    parser.add_argument(
        "--enroll-part-idx", default=1, type=int, help="enroll part index"
    )
    parser.add_argument(
        "--num-enroll-parts",
        default=1,
        type=int,
        help="""number of parts in which we divide the enroll
                list to run evaluation in parallel""",
    )
    parser.add_argument("--test-part-idx", default=1, type=int, help="test part index")
    parser.add_argument(
        "--num-test-parts",
        default=1,
        type=int,
        help="""number of parts in which we divide the test list
                to run evaluation in parallel""",
    )

    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_backend(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
