#!/usr/bin/env python
"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path

import pandas as pd
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.metrics import VerificationAnonymizationEvaluator as VE


def eval_verification_anonymization_metrics(
    key_files,
    score_orig_orig_files,
    score_orig_anon_files,
    score_anon_anon_files,
    enroll_map_files,
    anon_enroll_segments_files,
    anon_test_segments_files,
    key_names,
    score_names,
    p_tar,
    c_miss,
    c_fa,
    calibrate_on_orig,
    sparse,
    class_column,
    anon_class_column,
    output_file,
    output_fig_path,
):
    assert len(key_files) == len(key_names)
    assert len(score_orig_orig_files) == len(score_names)
    assert len(score_orig_anon_files) == len(score_names)
    assert len(score_anon_anon_files) == len(score_names)
    if (
        enroll_map_files is not None
        and anon_enroll_segments_files is not None
        and anon_test_segments_files is not None
    ):
        assert len(enroll_map_files) == len(key_names) or len(enroll_map_files) == 1
        assert (
            len(anon_enroll_segments_files) == len(key_names)
            or len(anon_enroll_segments_files) == 1
        )
        assert (
            len(anon_test_segments_files) == len(key_names)
            or len(anon_test_segments_files) == 1
        )

        if len(enroll_map_files) == 1:
            enroll_map_files = enroll_map_files * len(key_names)
        if len(anon_enroll_segments_files) == 1:
            anon_enroll_segments_files = anon_enroll_segments_files * len(key_names)
        if len(anon_test_segments_files) == 1:
            anon_test_segments_files = anon_test_segments_files * len(key_names)
    else:
        enroll_map_files = [None] * len(key_names)
        anon_enroll_segments_files = [None] * len(key_names)
        anon_test_segments_files = [None] * len(key_names)

    dfs = []
    for (
        score_orig_orig_file,
        score_orig_anon_file,
        score_anon_anon_file,
        score_name,
    ) in zip(
        score_orig_orig_files, score_orig_anon_files, score_anon_anon_files, score_names
    ):
        for (
            key_file,
            enroll_map_file,
            anon_enroll_segments_file,
            anon_test_segments_file,
            key_name,
        ) in zip(
            key_files,
            enroll_map_files,
            anon_enroll_segments_files,
            anon_test_segments_files,
            key_names,
        ):
            logging.info("Evaluating %s - %s", score_name, key_name)
            evaluator = VE(
                key_file,
                score_orig_orig_file,
                score_orig_anon_file,
                score_anon_anon_file,
                enroll_map_file,
                anon_enroll_segments_file,
                anon_test_segments_file,
                p_tar=p_tar,
                c_miss=c_miss,
                c_fa=c_fa,
                key_name=key_name,
                score_name=score_name,
                calibrate_on_orig=calibrate_on_orig,
                class_column=class_column,
                anon_class_column=anon_class_column,
                sparse=sparse,
            )
            df_ij = evaluator.compute_dcf_eer()
            if df_ij is not None:
                dfs.append(df_ij)

            if output_fig_path is not None:
                evaluator.plot_privacy_score_hist(output_fig_path)
                evaluator.plot_cons_div_score_hist(output_fig_path)

    df = pd.concat(dfs)
    logging.info("saving results to %s", output_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    sep = "\t" if output_file.suffix == ".tsv" else ","
    df.to_csv(output_file, sep=sep, index=False, float_format="{:.4f}".format)

    pd.options.display.float_format = "{:.4}".format
    print(df.to_string(), flush=True)


def main():
    parser = ArgumentParser(
        description="Evaluate speaker verification metrics for anonymization"
    )
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--key-files", required=True, nargs="+")
    parser.add_argument("--score-orig-orig-files", required=True, nargs="+")
    parser.add_argument("--score-orig-anon-files", required=True, nargs="+")
    parser.add_argument("--score-anon-anon-files", required=True, nargs="+")
    parser.add_argument("--enroll-map-files", required=False, nargs="+")
    parser.add_argument("--anon-enroll-segments-files", required=False, nargs="+")
    parser.add_argument("--anon-test-segments-files", required=False, nargs="+")
    parser.add_argument("--key-names", required=True, nargs="+")
    parser.add_argument("--score-names", required=True, nargs="+")

    parser.add_argument(
        "--p-tar",
        default=[0.05, 0.01],
        nargs="+",
        type=float,
        help="target priors",
    )
    parser.add_argument(
        "--c-miss", default=None, nargs="+", type=float, help="cost of miss"
    )
    parser.add_argument(
        "--c-fa", default=None, nargs="+", type=float, help="cost of false alarm"
    )
    parser.add_argument("--sparse", default=False, action=ActionYesNo)
    parser.add_argument("--class-column", default="speaker")
    parser.add_argument("--anon-class-column", default="pseudo_speaker")
    parser.add_argument("--calibrate-on-orig", default=False, action=ActionYesNo)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--output-fig-path", default=None, help="output figure path")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
    )

    args = parser.parse_args()
    kwargs = namespace_to_dict(args)
    config_logger(kwargs["verbose"])
    del kwargs["verbose"]
    del kwargs["cfg"]
    eval_verification_anonymization_metrics(**kwargs)


if __name__ == "__main__":
    main()
