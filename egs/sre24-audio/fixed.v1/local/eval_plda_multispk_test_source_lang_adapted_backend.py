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
import pandas as pd
import pickle

from hyperion.hyp_defs import config_logger
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np import HyperNPModel
from hyperion.np.pdfs import PLDAFactory, PLDALLRNvsMMethod
from hyperion.np.score_norm import AdaptSNorm
from hyperion.np.transforms import LNorm, TransformList
from hyperion.utils import EnrollmentMap, SegmentSet, TrialKey, TrialNdx, TrialScores
from hyperion.utils.math_funcs import cosine_scoring


def get_seg_conditions(segments, source_types, langs):
    assert np.all(segments["source_type"].isin(source_types))
    if langs is None:
        logging.info("getting segments source conditions")
        segments["language"] = "OTHER"
        segments["cond"] = segments["source_type"]

    else:
        logging.info("getting segments source-lang conditions")
        others = ~segments["language"].isin(langs)
        logging.info("segments other lang %d", np.sum(others))
        segments.loc[others, "language"] = "OTHER"
        seg_conds = [
            s if l == "OTHER" else f"{s}_{l}"
            for s, l, in zip(segments["source_type"], segments["language"])
        ]
        segments["cond"] = seg_conds

    v_counts = segments["cond"].value_counts()
    logging.info(f"segment condition summary: {v_counts}")
    return segments


def get_enroll_conditions(segments, modelids, enroll_map, source_types, langs):
    segments = SegmentSet(segments.loc[enroll_map["segmentid"]])
    segments = get_seg_conditions(segments, source_types, langs)
    table = []
    for m_id in modelids:
        idx = enroll_map["id"].values == m_id
        segs_m = segments.loc[idx]
        source = segs_m.iloc[0]["source_type"]
        if len(segs_m) > 1:
            assert source == segs_m.iloc[1]["source_type"]
        if langs is None:
            lang = None
            cond = source
        else:
            lang = segs_m.iloc[0]["source_type"]
            if (lang not in langs) or (
                len(segs_m) > 1 and lang != segs_m.iloc[1]["language"]
            ):
                lang = "OTHER"
                cond = source
            else:
                cond = f"{source}_{lang}"

        row = {"modelid": m_id, "source_type": source, "language": lang, "cond": cond}
        table.append(row)

    enroll_conds = pd.DataFrame(table)
    v_counts = enroll_conds["cond"].value_counts()
    logging.info(f"Enrollment conditions summary: {v_counts}")
    return segments, enroll_conds


def get_test_conditions(segments, seg_ids, source_types, langs):
    segments = segments.loc[seg_ids]
    return get_seg_conditions(segments, source_types, langs)


def get_enroll_ids(enroll_map, modelids):
    enroll_ids = np.zeros((len(enroll_map)), dtype=int)
    for i, modelid in enumerate(modelids):
        idx = enroll_map["id"] == modelid
        enroll_ids[idx] = i

    return enroll_ids


def load_trial_data(
    enroll_map_file,
    ndx_file,
    enroll_segments_file,
    test_segments_file,
    enroll_feats_file,
    feats_file,
    source_types,
    langs,
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

    enroll_segments = SegmentSet.load(enroll_segments_file)
    test_segments = SegmentSet.load(test_segments_file)

    if num_enroll_parts > 1 or num_test_parts > 1:
        ndx = ndx.split(
            enroll_part_idx, num_enroll_parts, test_part_idx, num_test_parts
        )

    enroll_map = enroll_map.filter(items=ndx.model_set)
    enroll_segments, enroll_conds = get_enroll_conditions(
        enroll_segments, ndx.model_set, enroll_map, source_types, langs
    )
    test_segments = get_test_conditions(test_segments, ndx.seg_set, source_types, langs)
    x_e = enroll_feats_reader.read(enroll_map["segmentid"], squeeze=True)
    x_t = test_feats_reader.read(ndx.seg_set, squeeze=False)
    test_map = np.hstack([[i] * len(x_t_i) for i, x_t_i in enumerate(x_t)])
    x_t = np.vstack(x_t)

    return (
        enroll_map,
        ndx,
        enroll_segments,
        enroll_conds,
        test_segments,
        test_map,
        x_e,
        x_t,
    )


def load_cohort_data(segments_files, feats_files, source_types, langs):

    segments = []
    x = []
    for segments_file, feats_file in zip(segments_files, feats_files):
        segments_i = SegmentSet.load(segments_file)
        feats_reader = DRF.create(feats_file)
        x_i = feats_reader.read(segments_i["id"], squeeze=True)
        segments.append(segments_i)
        x.append(x_i)

    segments = SegmentSet.cat(segments)
    segments = get_seg_conditions(segments, source_types, langs)
    x = np.concatenate(x, axis=0)

    return segments, x


def apply_preprocessors(x, segments, preprocessors, feat2seg_map=None):

    conditions = segments["cond"].unique()
    x_out = None
    for cond in conditions:
        if feat2seg_map is None:
            idx = segments["cond"] == cond
        else:
            seg_cond = segments.iloc[feat2seg_map]["cond"]
            idx = seg_cond == cond

        preprocessor_c = preprocessors[cond]
        x_out_c = preprocessor_c(x[idx])
        if x_out is None:
            x_out = np.zeros((x.shape[0], x_out_c.shape[1]), dtype=x.dtype)

        x_out[idx] = x_out_c

    return x_out


def eval_backend(
    enroll_map_file,
    ndx_file,
    enroll_segments_file,
    test_segments_file,
    enroll_feats_file,
    feats_file,
    source_types,
    langs,
    preproc_file,
    plda_file,
    llr_method,
    score_file,
    enroll_part_idx,
    num_enroll_parts,
    test_part_idx,
    num_test_parts,
    cohort_segments_files,
    cohort_feats_files,
    cohort_nbest,
    avg_cohort_by,
):
    logging.info("loading data")
    (
        enroll_map,
        ndx,
        enroll_segments,
        enroll_conds,
        test_segments,
        test_map,
        x_e,
        x_t,
    ) = load_trial_data(
        enroll_map_file,
        ndx_file,
        enroll_segments_file,
        test_segments_file,
        enroll_feats_file,
        feats_file,
        source_types,
        langs,
        enroll_part_idx,
        num_enroll_parts,
        test_part_idx,
        num_test_parts,
    )

    enroll_ids = get_enroll_ids(enroll_map, ndx.model_set)
    if len(ndx.model_set) == len(enroll_ids):
        is_Nvs1 = False
    else:
        is_Nvs1 = True

    t1 = time.time()

    if preproc_file is not None:
        logging.info("Loading Preprocessor")
        with open(preproc_file, "rb") as f:
            preprocessors = pickle.load(f)

        _, p_0 = next(iter(preprocessors.items()))
        if llr_method == PLDALLRNvsMMethod.vavg and isinstance(
            p_0.transforms[-1], LNorm
        ):
            llr_method = PLDALLRNvsMMethod.lnorm_vavg

        assert llr_method == PLDALLRNvsMMethod.lnorm_vavg, p_0.transforms

        x_e = apply_preprocessors(x_e, enroll_segments, preprocessors)
        x_t = apply_preprocessors(x_t, test_segments, preprocessors, test_map)

    logging.info("Loading PLDA model")
    plda_model = HyperNPModel.auto_load(plda_file)
    logging.info("computing score")
    if is_Nvs1:
        scores = plda_model.llr_Nvs1(x_e, x_t, ids1=enroll_ids, method=llr_method)
    else:
        x_e_sorted = np.zeros_like(x_e)
        x_e_sorted[enroll_idx, :] = x_e
        x_e = x_e_sorted
        scores = plda_model.llr_1vs1(x_e, x_t)

    dt = time.time() - t1
    num_trials = scores.shape[0] * scores.shape[1]
    logging.info(
        "scoring elapsed time: %.2f s. elapsed time per trial: %.2f ms.",
        dt,
        dt / num_trials * 1000,
    )

    if cohort_segments_files is not None:
        t1 = time.time()
        cohort_segments, x_coh = load_cohort_data(
            cohort_segments_files,
            cohort_feats_files,
            source_types,
            langs,
        )
        if preproc_file is not None:
            print("zzz", cohort_segments, flush=True)
            x_coh = apply_preprocessors(x_coh, cohort_segments, preprocessors)

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
        scores_coh_test = plda_model.llr_Nvs1(
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

    scores_max = np.zeros((scores.shape[0], len(ndx.seg_set)), dtype=scores.dtype)
    for i in range(scores_max.shape[1]):
        idx = test_map == i
        scores_max[:, i] = np.max(scores[:, idx], axis=1)

    scores = scores_max

    if num_enroll_parts > 1 or num_test_parts > 1:
        score_file = Path(score_file)
        new_suffix = f".{enroll_part_idx}.{test_part_idx}{score_file.suffix}"
        score_file = score_file.with_suffix(new_suffix)

    logging.info("saving scores to %s", score_file)
    # sort scores rows to match the ndx model_set order
    scores = TrialScores(ndx.model_set, ndx.seg_set, scores, ndx.trial_mask)
    scores.save(score_file)


def main():
    parser = ArgumentParser(description="Eval PLDA LLR with optional AS-Norm")

    parser.add_argument("--enroll-feats-file", default=None)
    parser.add_argument("--feats-file", required=True)
    parser.add_argument("--ndx-file", required=True)
    parser.add_argument("--enroll-map-file", required=True)
    parser.add_argument("--enroll-segments-file", required=True)
    parser.add_argument("--test-segments-file", required=True)
    parser.add_argument("--source-types", default=["cts", "afv"], nargs="+")
    parser.add_argument("--langs", default=None, nargs="+")
    parser.add_argument("--preproc-file", default=None)
    parser.add_argument("--plda-file", required=True)
    parser.add_argument(
        "--llr-method",
        default=PLDALLRNvsMMethod.vavg,
        choices=PLDALLRNvsMMethod.choices(),
    )
    parser.add_argument("--cohort-segments-files", default=None, nargs="+")
    parser.add_argument("--cohort-feats-files", default=None, nargs="+")
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
