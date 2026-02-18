#!/usr/bin/env python
"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.metrics import VerificationEvaluator as VE
from hyperion.utils.misc import PathLike


def eval_verification_metrics(
    key_files: List[PathLike],
    score_files: List[PathLike],
    key_names: List[str],
    score_names: List[str],
    avg_key_names: Optional[List[str]],
    eq_key_names: Optional[List[str]],
    p_tar: List[float],
    c_miss: Optional[List[float]],
    c_fa: Optional[List[float]],
    sparse: bool,
    output_file: PathLike,
) -> None:
    """Evaluate verification metrics across multiple keys and score systems.

    Args:
        key_files: Trial key file(s) containing target/non-target labels.
        score_files: Trial score file(s) to evaluate.
        key_names: Display name(s) corresponding to ``key_files``.
        score_names: Display name(s) corresponding to ``score_files``.
        avg_key_names: Optional subset of key names to average into a summary row.
        eq_key_names: Optional subset of key names used for equalized DCF/EER.
        p_tar: Target prior(s) used in DCF computation.
        c_miss: Optional miss cost(s) used in DCF computation.
        c_fa: Optional false-alarm cost(s) used in DCF computation.
        sparse: Whether to treat trial keys as sparse when loading.
        output_file: Destination CSV/TSV file for aggregated metrics.
    """
    assert len(key_files) == len(key_names)
    assert len(score_files) == len(score_names)
    dfs: List[pd.DataFrame] = []
    eq_tars: List[Any] = []
    eq_nons: List[Any] = []
    for score_file, score_name in zip(score_files, score_names):
        dfs_avg = []
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
            if df_ij is not None:
                dfs.append(df_ij)

            if avg_key_names is not None and key_name in avg_key_names:
                dfs_avg.append(df_ij)

            if eq_key_names is not None and key_name in eq_key_names:
                eq_tar_k, eq_non_k = evaluator.get_tar_non()
                eq_tars.append(eq_tar_k)
                eq_nons.append(eq_non_k)

        if avg_key_names is not None and len(dfs_avg) > 0:
            dfs_avg = pd.concat(dfs_avg)
            df_avg = {"scores": [score_name], "key": ["average"]}
            for column in dfs_avg.columns[2:]:
                df_avg[column] = [dfs_avg[column].mean()]
            df_avg = pd.DataFrame(df_avg)
            dfs.append(df_avg)

        if eq_key_names is not None and len(eq_tars) > 0:
            df_eq = evaluator.compute_equalized_dcf_eer(eq_tars, eq_nons)
            dfs.append(df_eq)

    df = pd.concat(dfs)
    logging.info("saving results to %s", output_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    sep = "\t" if output_file.suffix == ".tsv" else ","
    df.to_csv(output_file, sep=sep, index=False, float_format="{:.4f}".format)

    pd.options.display.float_format = "{:.4}".format
    print(df.to_string(), flush=True)


def main() -> None:
    """Parse CLI arguments and evaluate verification metrics."""
    parser = ArgumentParser(description="Evaluate speaker verification metrics")
    parser.add_argument("--cfg", action=ActionConfigFile, help="configuration file")
    parser.add_argument(
        "--key-files",
        required=True,
        nargs="+",
        help="trial key file(s) containing target/non-target labels",
    )
    parser.add_argument(
        "--score-files",
        required=True,
        nargs="+",
        help="trial score file(s) to evaluate",
    )
    parser.add_argument(
        "--key-names",
        required=True,
        nargs="+",
        help="display name(s) corresponding to --key-files",
    )
    parser.add_argument(
        "--score-names",
        required=True,
        nargs="+",
        help="display name(s) corresponding to --score-files",
    )
    parser.add_argument(
        "--avg-key-names",
        default=None,
        nargs="+",
        help="subset of key names to average into an additional summary row",
    )
    parser.add_argument(
        "--eq-key-names",
        default=None,
        nargs="+",
        help="subset of key names used to compute equalized DCF/EER",
    )
    parser.add_argument(
        "--p-tar",
        default=[0.05, 0.01, 0.005, 0.001],
        nargs="+",
        type=float,
        help="target prior(s) used in DCF computation",
    )
    parser.add_argument(
        "--c-miss",
        default=None,
        nargs="+",
        type=float,
        help="miss cost(s) used in DCF computation",
    )
    parser.add_argument(
        "--c-fa",
        default=None,
        nargs="+",
        type=float,
        help="false-alarm cost(s) used in DCF computation",
    )
    parser.add_argument(
        "--sparse",
        default=False,
        action=ActionYesNo,
        help="treat trial keys as sparse when loading",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="destination CSV/TSV file for aggregated metrics",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="verbosity level (0=warning, 1=info, 2=debug, 3=trace)",
    )

    args = parser.parse_args()
    kwargs: Dict[str, Any] = namespace_to_dict(args)
    config_logger(kwargs["verbose"])
    del kwargs["verbose"]
    del kwargs["cfg"]
    eval_verification_metrics(**kwargs)


if __name__ == "__main__":
    main()
