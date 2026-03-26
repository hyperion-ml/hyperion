#!/usr/bin/env python
"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from hyperion.metrics import VerificationEvaluator as VE
from hyperion.np.metrics.dcf_plot import NormDCFPlot
from hyperion.np.metrics.det_plot import DETPlot, DETPlotWindowType
from hyperion.np.metrics.utils import effective_prior
from hyperion.utils.misc import PathLike

MATPLOTLIB_COLOR_STRINGS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]
MATPLOTLIB_LINE_TYPES = ["-", "--", "-.", ":"]


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
    plot_det: bool,
    plot_dcf: bool,
    det_plot_window: str,
    det_plot_title: Optional[str],
    dcf_plot_title: Optional[str],
    dcf_plot_min_prior: float,
    dcf_plot_max_prior: float,
    dcf_plot_min: float,
    dcf_plot_max: float,
    plot_det_min_dcf: bool,
    plot_det_act_dcf: bool,
    line_width: float,
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
        plot_det: If True, create a DET plot object and pass it to evaluator.
        plot_dcf: If True, create a normalized DCF plot object and pass it to evaluator.
        det_plot_window: DET plot window preset.
        det_plot_title: Optional DET plot title.
        dcf_plot_title: Optional normalized DCF plot title.
        dcf_plot_min_prior: Min target prior for normalized DCF x-axis.
        dcf_plot_max_prior: Max target prior for normalized DCF x-axis.
        dcf_plot_min: Min y-axis value for normalized DCF plot.
        dcf_plot_max: Max y-axis value for normalized DCF plot.
        plot_det_min_dcf: If True, plot DET min-DCF points.
        plot_det_act_dcf: If True, plot DET act-DCF points.
        line_width: Line width used for DET/DCF plotted curves.
        output_file: Destination CSV/TSV file for aggregated metrics.
    """
    assert len(key_files) == len(key_names)
    assert len(score_files) == len(score_names)
    if line_width <= 0:
        raise ValueError(f"line_width must be > 0, got {line_width}")

    plot_priors = np.atleast_1d(np.asarray(p_tar, dtype=float))
    if c_miss is not None and c_fa is not None:
        c_miss_arr = np.atleast_1d(np.asarray(c_miss, dtype=float))
        c_fa_arr = np.atleast_1d(np.asarray(c_fa, dtype=float))
        if c_miss_arr.size == 1 and plot_priors.size > 1:
            c_miss_arr = np.full(plot_priors.shape, c_miss_arr.item(), dtype=float)
        if c_fa_arr.size == 1 and plot_priors.size > 1:
            c_fa_arr = np.full(plot_priors.shape, c_fa_arr.item(), dtype=float)
        plot_priors = effective_prior(plot_priors, c_miss_arr, c_fa_arr)

    det_plot = (
        DETPlot(
            plot_window=det_plot_window,
            plot_title=det_plot_title,
            priors=plot_priors,
        )
        if plot_det
        else None
    )

    norm_dcf_plot = (
        NormDCFPlot(
            min_prior=dcf_plot_min_prior,
            max_prior=dcf_plot_max_prior,
            min_dcf=dcf_plot_min,
            max_dcf=dcf_plot_max,
            plot_title=dcf_plot_title,
        )
        if plot_dcf
        else None
    )

    dfs: List[pd.DataFrame] = []
    style_idx = 0
    for score_file, score_name in zip(score_files, score_names):
        dfs_avg = []
        eq_tars: List[Any] = []
        eq_nons: List[Any] = []
        for key_file, key_name in zip(key_files, key_names):
            logging.info("Evaluating %s - %s", score_name, key_name)
            color = MATPLOTLIB_COLOR_STRINGS[style_idx % len(MATPLOTLIB_COLOR_STRINGS)]
            det_line_type = MATPLOTLIB_LINE_TYPES[
                (style_idx // len(MATPLOTLIB_COLOR_STRINGS))
                % len(MATPLOTLIB_LINE_TYPES)
            ]
            style_idx += 1
            evaluator = VE(
                key_file,
                score_file,
                p_tar,
                c_miss,
                c_fa,
                key_name,
                score_name,
                dcf_plot=det_plot,
                norm_dcf_plot=norm_dcf_plot,
                color=color,
                det_line_type=det_line_type,
                line_width=line_width,
                plot_det_min_dcf=plot_det_min_dcf,
                plot_det_act_dcf=plot_det_act_dcf,
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
        "--plot-det",
        default=False,
        action=ActionYesNo,
        help="create DET plot and pass it to VerificationEvaluator",
    )
    parser.add_argument(
        "--plot-dcf",
        default=False,
        action=ActionYesNo,
        help="create normalized DCF plot and pass it to VerificationEvaluator",
    )
    parser.add_argument(
        "--det-plot-window",
        default="sre12",
        choices=DETPlotWindowType.choices(),
        help="DET window preset",
    )
    parser.add_argument(
        "--det-plot-title",
        default="DET",
        help="DET plot title",
    )
    parser.add_argument(
        "--dcf-plot-title",
        default="Norm DCF",
        help="normalized DCF plot title",
    )
    parser.add_argument(
        "--dcf-plot-min-prior",
        default=1e-3,
        type=float,
        help="minimum target prior for normalized DCF plot",
    )
    parser.add_argument(
        "--dcf-plot-max-prior",
        default=0.5,
        type=float,
        help="maximum target prior for normalized DCF plot",
    )
    parser.add_argument(
        "--dcf-plot-min",
        default=0.0,
        type=float,
        help="minimum y-axis value for normalized DCF plot",
    )
    parser.add_argument(
        "--dcf-plot-max",
        default=1.2,
        type=float,
        help="maximum y-axis value for normalized DCF plot",
    )
    parser.add_argument(
        "--plot-det-min-dcf",
        default=True,
        action=ActionYesNo,
        help="plot DET min-DCF points",
    )
    parser.add_argument(
        "--plot-det-act-dcf",
        default=True,
        action=ActionYesNo,
        help="plot DET act-DCF points",
    )
    parser.add_argument(
        "--line-width",
        default=1.5,
        type=float,
        help="line width used for DET/DCF plotted curves",
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
