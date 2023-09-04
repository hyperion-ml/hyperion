#!/usr/bin/env python
"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path
import pandas as pd

from hyperion.hyp_defs import config_logger
from hyperion.np.metrics import VerificationEvaluator as VE

from jsonargparse import (
    ActionConfigFile,
    ActionYesNo,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)


def eval_verification_metrics(
    key_files,
    score_files,
    key_names,
    score_names,
    p_tar,
    c_miss,
    c_fa,
    sparse,
    output_file,
):

    assert len(key_files) == len(key_names)
    assert len(score_files) == len(score_names)
    dfs = []
    for score_file, score_name in zip(score_files, score_names):
        for key_file, key_name in zip(key_files, key_names):
            logging.info("Evaluating %s - %s", score_name, key_name)
            evaluator = VE(
                key_file,
                score_file,
                p_tar,
                c_miss,
                c_fa,
                key_name,
                score_name,
                sparse=sparse,
            )
            df_ij = evaluator.compute_dcf_eer()
            dfs.append(df_ij)

    df = pd.concat(dfs)
    logging.info("saving results to %s", output_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    sep = "\t" if output_file.suffix == ".tsv" else ","
    df.to_csv(output_file, sep=sep, index=False, float_format="{:,.4f}".format)

    pd.options.display.float_format = "{:.4}".format
    print(df.to_string(), flush=True)


if __name__ == "__main__":

    parser = ArgumentParser(description="Evaluate speaker verification metrics")
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--key-files", required=True, nargs="+")
    parser.add_argument("--score-files", required=True, nargs="+")
    parser.add_argument("--key-names", required=True, nargs="+")
    parser.add_argument("--score-names", required=True, nargs="+")
    parser.add_argument(
        "--p-tar",
        default=[0.05, 0.01, 0.005, 0.001],
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
    parser.add_argument("--output-file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int,
    )

    args = parser.parse_args()
    kwargs = namespace_to_dict(args)
    config_logger(kwargs["verbose"])
    del kwargs["verbose"]
    del kwargs["cfg"]
    eval_verification_metrics(**kwargs)
