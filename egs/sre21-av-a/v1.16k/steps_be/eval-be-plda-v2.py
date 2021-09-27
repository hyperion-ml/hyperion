#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

  Evals PLDA LLR
"""
import sys
import os
import argparse
import time
import logging
from jsonargparse import ArgumentParser, namespace_to_dict

import numpy as np
import pandas as pd

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils import TrialNdx, TrialScores, Utt2Info
from hyperion.helpers import TrialDataReader as TDR
from hyperion.helpers import PLDAFactory as F
from hyperion.transforms import TransformList

conds = [
    "cts_eng",
    "cts_cmn",
    "cts_yue",
    "afv_eng",
    "afv_cmn",
    "afv_yue",
]


def get_source_type(source_type, keys, table_file):
    if source_type in ["cts", "afv"]:
        return np.asarray([source_type] * len(keys))

    df = pd.read_csv(table_file)
    if "segment_id" in df:
        df.index = df.segment_id
    else:
        df.index = df.model_id

    return df.loc[keys].source_type.values


def get_lang(utt2lang, key):
    u2l = Utt2Info.load(utt2lang)
    u2l = u2l.filter(key)
    langs = u2l.info
    langs[langs == "OTHER"] = "YUE"
    langs[langs == "other"] = "YUE"
    langs[langs == "CMN.YUE"] = "YUE"
    langs[langs == "CMN.WUU"] = "CMN"
    langs[langs == "UND"] = "CMN"
    langs[langs == "THA"] = "CMN"
    langs[langs == "USE"] = "ENG"
    langs[langs == "WUU"] = "CMN"
    return langs


def get_conditions(source, lang):
    return np.asarray([f"{s}_{l.lower()}" for s, l in zip(source, lang)])


def apply_preproc(x_e, x_t, conds_e, conds_t, preproc_file_basename):
    x_ln_e = None
    x_ln_t = None
    n_e = 0
    n_t = 0
    for cond in conds:
        preproc_file = f"{preproc_file_basename}_{cond}.h5"
        logging.info("load preproc condition: %s  file: %s", cond, preproc_file)
        preproc = TransformList.load(preproc_file)
        idx = conds_e == cond
        n_c = np.sum(idx)
        n_e += n_c
        logging.info("condition: %s enroll-segms: %d", cond, n_c)
        if n_c > 0:
            x_e_c = preproc.predict(x_e[idx])
            if x_ln_e is None:
                x_ln_e = np.zeros((x_e.shape[0], x_e_c.shape[1]), dtype=x_e.dtype)
            x_ln_e[idx] = x_e_c

        idx = conds_t == cond
        n_c = np.sum(idx)
        n_t += n_c
        logging.info("condition: %s test-segms: %d", cond, n_c)
        if n_c > 0:
            x_t_c = preproc.predict(x_t[idx])
            if x_ln_t is None:
                x_ln_t = np.zeros((x_t.shape[0], x_t_c.shape[1]), dtype=x_t.dtype)
            x_ln_t[idx] = x_t_c

    assert n_e == x_e.shape[0], (
        f"enrollments segments processed {n_e} != {x_e.shape[0]}, "
        f"enroll-conditions={np.unique(conds_e)}"
    )
    assert n_t == x_t.shape[0], (
        f"enrollments segments processed {n_t} != {x_t.shape[0]}, "
        f"test-conditions={np.unique(conds_t)}"
    )

    return x_ln_e, x_ln_t


def eval_plda(
    v_file,
    ndx_file,
    enroll_file,
    preproc_file_basename,
    source_type,
    enroll_table,
    test_table,
    enroll_lang,
    test_lang,
    model_file,
    score_file,
    plda_type,
    **kwargs,
):

    logging.info("loading data")
    tdr = TDR(v_file, ndx_file, enroll_file, None, None)
    x_e, x_t, enroll, ndx = tdr.read()
    source_e = get_source_type(source_type, tdr.enroll.key, enroll_table)
    source_t = get_source_type(source_type, ndx.seg_set, test_table)
    lang_e = get_lang(enroll_lang, tdr.enroll.key)
    lang_t = get_lang(test_lang, ndx.seg_set)
    conds_e = get_conditions(source_e, lang_e)
    conds_t = get_conditions(source_t, lang_t)

    logging.info("load preprocessors and apply")
    x_e, x_t = apply_preproc(x_e, x_t, conds_e, conds_t, preproc_file_basename)
    enroll, ids_e = np.unique(enroll, return_inverse=True)

    logging.info("loading plda model: %s", model_file)
    model = F.load_plda(plda_type, model_file)

    t1 = time.time()
    logging.info("computing llr")
    scores = model.llr_Nvs1(x_e, x_t, method="vavg-lnorm", ids1=ids_e)

    dt = time.time() - t1
    num_trials = len(enroll) * x_t.shape[0]
    logging.info(
        "scoring elapsed time: %.2f s. elapsed time per trial: %.2f ms."
        % (dt, dt / num_trials * 1000)
    )

    logging.info("saving scores to %s" % (score_file))
    s = TrialScores(enroll, ndx.seg_set, scores)
    s.save_txt(score_file)


if __name__ == "__main__":

    parser = ArgumentParser(description="Evals PLDA")

    parser.add_argument("--v-file", required=True)
    parser.add_argument("--ndx-file", default=None)
    parser.add_argument("--enroll-file", required=True)
    parser.add_argument("--preproc-file-basename", required=True)
    parser.add_argument("--source-type", required=True, choices=["cts", "afv", "mixed"])
    parser.add_argument("--enroll-table", default=None)
    parser.add_argument("--test-table", default=None)
    parser.add_argument("--enroll-lang", required=True)
    parser.add_argument("--test-lang", required=True)

    TDR.add_argparse_args(parser)
    F.add_argparse_eval_args(parser)

    parser.add_argument("--score-file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_plda(**vars(args))
