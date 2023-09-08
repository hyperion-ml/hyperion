#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

"""
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
)
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hyperion.hyp_defs import config_logger
from hyperion.utils import (
    TrialNdx,
    TrialKey,
    TrialScores,
    EnrollmentMap,
    SegmentSet,
    InfoTable,
)
from hyperion.utils.math_funcs import cosine_scoring, average_vectors
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np.transforms import TransformList
from hyperion.np.score_norm import AdaptSNorm
from hyperion.np.classifiers import BinaryLogisticRegression as LR


def get_precomp_qm_names(quality_measures):
    # snorm qm will be calculated later
    return [q for q in quality_measures if q not in ["snorm-mu", "snorm-mu/s"]]


def normalize_duration(q, min_dur, max_dur, frame_rate):
    q = q / frame_rate
    q = np.log(np.clip(q / frame_rate, a_min=min_dur, a_max=max_dur))
    log_min_dur = np.log(min_dur)
    log_max_dur = np.log(max_dur)
    q = (q - log_min_dur) / (log_max_dur - log_min_dur)
    return q


def load_trial_data(
    enroll_map_file,
    ndx_file,
    enroll_feats_file,
    feats_file,
    enroll_segments_file,
    segments_file,
    quality_measures,
    min_dur,
    max_dur,
    frame_rate,
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

    # quality measures may be in segments file or/and feature_set file
    # so we combine both if both are given
    if segments_file is not None:
        test_segments = SegmentSet.load(segments_file)
        if enroll_segments_file is not None and segments_file != enroll_segments_file:
            enroll_segments = SegmentSet.load(enroll_segments_file)
        else:
            enroll_segments = test_segments

    test_feats_set = test_feats_reader.feature_set
    enroll_feats_set = enroll_feats_reader.feature_set
    if segments_file:
        test_segments.add_columns(test_feats_set)
        if enroll_feats_set != test_feats_set or enroll_segments != test_segments:
            enroll_segments.add_columns(enroll_feats_set)
    else:
        test_segments = test_feats_set
        enroll_segments = enroll_feats_set

    # now we retrive the quality measures
    q_e = []
    q_t = []
    # snorm qm will be calculated later
    retrieve_qm = get_precomp_qm_names(quality_measures)
    q_e = enroll_segments.loc[enroll_map["segmentid"], retrieve_qm]
    q_t = test_segments.loc[ndx.seg_set, retrieve_qm]

    # normalize durations
    if "speech_duration" in retrieve_qm:
        q_e["speech_duration"] = normalize_duration(
            q_e["speech_duration"], min_dur, max_dur, 1
        )
        q_t["speech_duration"] = normalize_duration(
            q_t["speech_duration"], min_dur, max_dur, 1
        )

    if "num_speech_frames" in retrieve_qm:
        q_e["num_speech_frames"] = normalize_duration(
            q_e["num_speech_frames"], min_dur, max_dur, frame_rate
        )
        q_t["num_speech_frames"] = normalize_duration(
            q_t["num_speech_frames"], min_dur, max_dur, frame_rate
        )

    # q_e = np.asarray(q_e)
    # q_t = np.asarray(q_t)

    return enroll_map, ndx, x_e, x_t, q_e, q_t


def load_cohort_data(segments_file, feats_file):
    segments = SegmentSet.load(segments_file)
    feats_reader = DRF.create(feats_file)
    x = feats_reader.read(segments["id"], squeeze=True)

    # segments.add_columns(feats_reader.feature_set)

    # retrieve_qm = get_precomp_qm_names(quality_measures)
    # q = np.asarray(segments[retrieve_qm])
    return segments, x  # , q


def average_qm(q, model_set, ids):
    q_avg = average_vectors(q.values, ids)
    q_avg = pd.DataFrame(q, columns=q.columns)
    q_avg["id"] = model_set
    q_avg.set_index("id", drop=False, inplace=True)
    return q_avg


def get_score_filepath(
    score_file,
    score_name,
    enroll_part_idx,
    num_enroll_parts,
    test_part_idx,
    num_test_parts,
):
    score_file = Path(score_file)
    new_suffix = ""
    if score_name is not None:
        new_suffix = f".{score_name}"

    if num_enroll_parts > 1 or num_test_parts > 1:
        new_suffix = f"{new_suffix}.{enroll_part_idx}.{test_part_idx}"

    if new_suffix:
        new_suffix = f"{new_suffix}{score_file.suffix}"
        score_file = score_file.with_suffix(new_suffix)

    return score_file


def save_scores(
    ndx,
    scores,
    score_file,
    score_name,
    q_measures,
    enroll_part_idx,
    num_enroll_parts,
    test_part_idx,
    num_test_parts,
):
    score_file = get_score_filepath(
        score_file,
        score_name,
        enroll_part_idx,
        num_enroll_parts,
        test_part_idx,
        num_test_parts,
    )
    logging.info("saving scores with to %s", score_file)
    scores = TrialScores(
        ndx.model_set, ndx.seg_set, scores, ndx.trial_mask, q_measures=q_measures
    )
    scores.save(score_file)


def save_empty_scores(
    ndx,
    score_file,
    score_name,
    q_measures,
    enroll_part_idx,
    num_enroll_parts,
    test_part_idx,
    num_test_parts,
):
    scores = np.zeros(ndx.trial_mask.shape, dtype="float32")
    if q_measures is not None:
        q_measures = {k: scores for k in q_measures}

    save_scores(
        ndx,
        scores,
        score_file,
        score_name,
        q_measures,
        enroll_part_idx,
        num_enroll_parts,
        test_part_idx,
        num_test_parts,
    )


def segment_to_trial_qm(q_e, q_t):
    q_trial = {}
    for q_name in ["speech_duration", "num_speech_frames"]:
        if q_name in q_e:
            q_trial_name = f"max_{q_name}"
            q_trial[q_trial_name] = np.maximum(
                q_e[q_name].values[:, None], q_t[q_name].values[None, :]
            )
            q_trial_name = f"min_{q_name}"
            q_trial[q_trial_name] = np.minimum(
                q_e[q_name].values[:, None], q_t[q_name].values[None, :]
            )

    return q_trial


def align_scores_to_ndx(enroll_set, ndx, scores, scores_norm, q_trial):
    # sort scores rows to match the ndx model_set order
    sort_idx = [np.nonzero(enroll_set == e)[0][0] for e in ndx.model_set]
    scores = scores[sort_idx]
    if scores_norm is not None:
        scores_norm = scores_norm[sort_idx]
    for qm in q_trial:
        q_trial[qm] = q_trial[qm][sort_idx]

    return scores, scores_norm, q_trial


# def make_qm_table(ndx, scores, scores_norm, q_trial):
#     if scores_norm is None:
#         scores = scores[ndx.trial_mask]
#     else:
#         scores = scores_norm[ndx.trial_mask]

#     for qm in q_trial:
#         q_trial[qm] = q_trial[qm][ndx.trial_mask]

#     I, J = np.nonzero(ndx.trial_mask)
#     modelid = ndx.model_set[I]
#     segmentid = ndx.seg_set[J]
#     unique_id = [f"{a}-{b}" for a, b in zip(modelid, segmentid)]

#     q_dict = {
#         "id": unique_id,
#         "modelid": modelid,
#         "segmentid": segmentid,
#         "scores": scores,
#     }
#     q_dict.update(q_trial)
#     df = pd.DataFrame(q_dict)
#     return InfoTable(df)


def eval_backend(
    enroll_map_file,
    ndx_file,
    enroll_feats_file,
    feats_file,
    enroll_segments_file,
    segments_file,
    preproc_file,
    qmf_file,
    quality_measures,
    min_dur,
    max_dur,
    frame_rate,
    cohort_segments_file,
    cohort_feats_file,
    cohort_nbest,
    avg_cohort_by,
    score_file,
    enroll_part_idx,
    num_enroll_parts,
    test_part_idx,
    num_test_parts,
):
    logging.info("loading data")
    enroll_map, ndx, x_e, x_t, q_e, q_t = load_trial_data(
        enroll_map_file,
        ndx_file,
        enroll_feats_file,
        feats_file,
        enroll_segments_file,
        segments_file,
        quality_measures,
        min_dur,
        max_dur,
        frame_rate,
        enroll_part_idx,
        num_enroll_parts,
        test_part_idx,
        num_test_parts,
    )

    if not np.any(ndx.trial_mask):
        # this part doesn't have any trials, save empty files
        if qmf_file is not None:
            quality_measures = None
            save_empty_scores(
                ndx,
                score_file,
                "snorm.qmf" if cohort_segments_file is not None else "qmf",
                quality_measures,
                enroll_part_idx,
                num_enroll_parts,
                test_part_idx,
                num_test_parts,
            )

        save_empty_scores(
            ndx,
            score_file,
            None,
            quality_measures,
            enroll_part_idx,
            num_enroll_parts,
            test_part_idx,
            num_test_parts,
        )

        if cohort_segments_file is not None:
            save_empty_scores(
                ndx,
                score_file,
                "snorm",
                quality_measures,
                enroll_part_idx,
                num_enroll_parts,
                test_part_idx,
                num_test_parts,
            )
        return

    enroll_set, enroll_ids = np.unique(enroll_map["id"], return_inverse=True)
    q_e = average_qm(q_e, enroll_set, enroll_ids)

    t1 = time.time()
    logging.info("computing score")
    if preproc_file is not None:
        preprocessor = TransformList.load(preproc_file)
        x_e = preprocessor(x_e)
        x_t = preprocessor(x_t)

    scores = cosine_scoring(x_e, x_t, ids1=enroll_ids)
    dt = time.time() - t1
    num_trials = scores.shape[0] * scores.shape[1]
    logging.info(
        "scoring elapsed time: %.2f s. elapsed time per trial: %.2f ms.",
        dt,
        dt / num_trials * 1000,
    )

    q_trial = segment_to_trial_qm(q_e, q_t)
    scores_norm = None
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
        scores_enr_coh = cosine_scoring(x_e, x_coh, ids2=cohort_ids)
        logging.info("computing cohort vs test")
        scores_coh_test = cosine_scoring(x_coh, x_t, ids1=cohort_ids)
        snorm = AdaptSNorm(cohort_nbest)
        scores_norm, mu_z, s_z, mu_t, s_t = snorm(
            scores, scores_coh_test, scores_enr_coh, return_stats=True
        )
        if "snorm-mu" in quality_measures:
            q_trial["max_snorm-mu"] = np.maximum(mu_z, mu_t)
            q_trial["min_snorm-mu"] = np.minimum(mu_z, mu_t)
        if "snorm-mu/s" in quality_measures:
            mu_z = mu_z / s_z
            mu_t = mu_t / s_t
            q_trial["max_snorm-mu/s"] = np.maximum(mu_z, mu_t)
            q_trial["min_snorm-mu/s"] = np.minimum(mu_z, mu_t)

        dt = time.time() - t1
        logging.info(
            "s-norm elapsed time: %.2f s. elapsed time per trial: %.2f ms.",
            dt,
            dt / num_trials * 1000,
        )

    scores, scores_norm, q_trial = align_scores_to_ndx(
        enroll_set, ndx, scores, scores_norm, q_trial
    )
    if qmf_file is None:
        save_scores(
            ndx,
            scores,
            score_file,
            None,
            q_trial,
            enroll_part_idx,
            num_enroll_parts,
            test_part_idx,
            num_test_parts,
        )

        if scores_norm is not None:
            save_scores(
                ndx,
                scores_norm,
                score_file,
                "snorm",
                q_trial,
                enroll_part_idx,
                num_enroll_parts,
                test_part_idx,
                num_test_parts,
            )
        # qm_table = make_qm_table(ndx, scores, scores_norm, q_trial)
        # qm_file = get_score_filepath(
        #     score_file,
        #     "qm",
        #     enroll_part_idx,
        #     num_enroll_parts,
        #     test_part_idx,
        #     num_test_parts,
        # )
        # qm_table.save(qm_file)
        return

    save_scores(
        ndx,
        scores,
        score_file,
        None,
        None,
        enroll_part_idx,
        num_enroll_parts,
        test_part_idx,
        num_test_parts,
    )

    if scores_norm is not None:
        save_scores(
            ndx,
            scores_norm,
            score_file,
            "snorm",
            None,
            enroll_part_idx,
            num_enroll_parts,
            test_part_idx,
            num_test_parts,
        )

    logging.info("applying qmf")
    if scores_norm is None:
        score_name = "qmf"
        scores_fus = [scores.ravel()]
    else:
        score_name = "snorm.qmf"
        scores_fus = [scores_norm.ravel()]

    q_names = list(q_trial.keys())
    q_names.sort()
    for q_name in q_names:
        scores_fus.append(q_trial[q_name].ravel())

    scores_fus = np.vstack(scores_fus).T
    lr = LR.load(qmf_file)
    scores_fus = lr.predict(scores_fus)
    scores_fus = np.reshape(scores_fus, (ndx.num_models, ndx.num_tests))
    save_scores(
        ndx,
        scores_fus,
        score_file,
        score_name,
        None,
        enroll_part_idx,
        num_enroll_parts,
        test_part_idx,
        num_test_parts,
    )

    # score_file_nonorm = get_score_filepath(
    #     score_file,
    #     None,
    #     enroll_part_idx,
    #     num_enroll_parts,
    #     test_part_idx,
    #     num_test_parts,
    # )
    # logging.info("saving scores to %s", score_file_nonorm)
    # scores = TrialScores(ndx.model_set, ndx.seg_set, scores, ndx.trial_mask)
    # scores.save(score_file_nonorm)

    # if scores_norm is not None:
    #     score_file_snorm = get_score_filepath(
    #         score_file,
    #         "snorm",
    #         enroll_part_idx,
    #         num_enroll_parts,
    #         test_part_idx,
    #         num_test_parts,
    #     )
    #     logging.info("saving scores with AS-Norm to %s", score_file_snorm)
    #     scores.scores = scores_norm
    #     scores.save(score_file_snorm)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Eval cosine-scoring with optional AS-Norm and QMF"
    )

    parser.add_argument("--enroll-feats-file", default=None)
    parser.add_argument("--feats-file", required=True)
    parser.add_argument("--ndx-file", required=True)
    parser.add_argument("--enroll-map-file", required=True)
    parser.add_argument("--enroll-segments-file", default=None)
    parser.add_argument("--segments-file", default=None)
    parser.add_argument("--preproc-file", default=None)
    parser.add_argument("--qmf-file", default=None)
    parser.add_argument(
        "--quality-measures",
        default=["snorm-mu/s", "speech_duration"],
        nargs="+",
        choices=["snorm-mu/s", "snorm-mu", "speech_duration", "num_speech_frames"],
    )
    parser.add_argument(
        "--min-dur", default=0.1, type=float, help="lower bound to clip durations"
    )
    parser.add_argument(
        "--max-dur", default=30.0, type=float, help="upper bound to clip durations"
    )
    parser.add_argument(
        "--frame-rate",
        default=100,
        type=float,
        help="frames/sec when durationa are expressed in frames",
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
