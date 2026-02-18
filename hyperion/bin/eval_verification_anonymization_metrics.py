#!/usr/bin/env python
"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
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
from hyperion.metrics import VerificationAnonymizationEvaluator as VE
from hyperion.utils.misc import PathLike


def eval_verification_anonymization_metrics(
    key_files: List[PathLike],
    ref_key_file: Optional[PathLike],
    score_orig_orig_files: List[PathLike],
    score_orig_anon_files: List[PathLike],
    score_anon_anon_files: List[PathLike],
    score_ref_anon_files: Optional[List[PathLike]],
    enroll_map_files: Optional[List[PathLike]],
    anon_enroll_segments_files: Optional[List[PathLike]],
    anon_test_segments_files: Optional[List[PathLike]],
    key_names: List[str],
    score_names: List[str],
    p_tar: List[float],
    c_miss: Optional[List[float]],
    c_fa: Optional[List[float]],
    calibrate_on_orig: bool,
    sparse: bool,
    class_column: str,
    anon_class_column: str,
    output_file: PathLike,
    output_fig_path: Optional[PathLike],
) -> None:
    """Compute verification/anonymization metrics for multiple score sets.

    Args:
        key_files (list[str]): Trial lists describing target/non-target labels for
            each corpus.
        ref_key_file (str | None): Optional trial list describing target/non-target
            labels for reference pseudo-speakers vs anonymized.
        score_orig_orig_files (list[str]): Score files produced with original enroll
            and original test trials.
        score_orig_anon_files (list[str]): Score files for original enroll versus
            anonymized test trials.
        score_anon_anon_files (list[str]): Score files for anonymized enroll versus
            anonymized test trials.
        score_ref_anon_files (list[str] | None): Optional score files for reference
            pseudo-speaker vs anonymized test trials.
        enroll_map_files (list[str] | None): Optional mapping files linking
            original and anonymized enroll segments.
        anon_enroll_segments_files (list[str] | None): Optional segment tables for
            anonymized enrollment audio.
        anon_test_segments_files (list[str] | None): Optional segment tables for
            anonymized test audio.
        key_names (list[str]): Human readable names for each key file (used in
            reporting).
        score_names (list[str]): Names identifying each scoring backend or system.
        p_tar (list[float]): Target priors used when computing minC and actC.
        c_miss (list[float] | None): Miss costs for DCF calculation. Defaults to
            ASVspoof conventions if ``None``.
        c_fa (list[float] | None): False-alarm costs for DCF calculation. Defaults
            to ASVspoof conventions if ``None``.
        calibrate_on_orig (bool): Whether to learn calibration on
            original-original scores before evaluation.
        sparse (bool): If True, treat the keys as sparse matrices during scoring.
        class_column (str): Column name with speaker IDs in the key metadata.
        anon_class_column (str): Column name with anonymized speaker IDs, if
            available.
        output_file (str): Destination CSV/TSV path where the aggregated metrics
            are saved.
        output_fig_path (str | None): Directory path for storing optional score
            histograms.
    """
    assert len(key_files) == len(key_names)
    assert len(score_orig_orig_files) == len(score_names)
    assert len(score_orig_anon_files) == len(score_names)
    assert len(score_anon_anon_files) == len(score_names)
    if score_ref_anon_files is not None:
        assert len(score_ref_anon_files) == len(score_names)
    else:
        score_ref_anon_files = [None] * len(score_names)
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

    dfs: List[pd.DataFrame] = []
    for (
        score_orig_orig_file,
        score_orig_anon_file,
        score_anon_anon_file,
        score_name,
        score_ref_anon_file,
    ) in zip(
        score_orig_orig_files,
        score_orig_anon_files,
        score_anon_anon_files,
        score_names,
        score_ref_anon_files,
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
                ref_key=ref_key_file if key_name == key_names[0] else None,
                scores_ref_anon=(
                    score_ref_anon_file if key_name == key_names[0] else None
                ),
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


def main() -> None:
    """Parse CLI arguments and launch the evaluation pipeline."""
    parser = ArgumentParser(
        description="Evaluate speaker verification metrics for anonymization"
    )
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="Path to a YAML/JSON config file."
    )
    parser.add_argument(
        "--key-files",
        required=True,
        nargs="+",
        help="Trial key file(s) containing target/non-target labels.",
    )
    parser.add_argument(
        "--ref-key-file",
        default=None,
        help="Trial key file containing target/non-target labels for reference pseudo-speakers vs anonymized.",
    )
    parser.add_argument(
        "--score-orig-orig-files",
        required=True,
        nargs="+",
        help="Score file(s) for original enroll vs. original test trials.",
    )
    parser.add_argument(
        "--score-orig-anon-files",
        required=True,
        nargs="+",
        help="Score file(s) for original enroll vs. anonymized test trials.",
    )
    parser.add_argument(
        "--score-anon-anon-files",
        required=True,
        nargs="+",
        help="Score file(s) for anonymized enroll vs. anonymized test trials.",
    )
    parser.add_argument(
        "--score-ref-anon-files",
        default=None,
        nargs="+",
        help="Score file(s) for reference pseudo-speaker vs. anonymized test trials.",
    )
    parser.add_argument(
        "--enroll-map-files",
        required=False,
        nargs="+",
        help="Optional mapping file(s) between original and anonymized enroll IDs.",
    )
    parser.add_argument(
        "--anon-enroll-segments-files",
        required=False,
        nargs="+",
        help="Optional segment table(s) describing anonymized enrollment audio.",
    )
    parser.add_argument(
        "--anon-test-segments-files",
        required=False,
        nargs="+",
        help="Optional segment table(s) describing anonymized test audio.",
    )
    parser.add_argument(
        "--key-names",
        required=True,
        nargs="+",
        help="Human-readable name(s) for each key file used in reports.",
    )
    parser.add_argument(
        "--score-names",
        required=True,
        nargs="+",
        help="Label(s) identifying each scoring backend/system.",
    )

    parser.add_argument(
        "--p-tar",
        default=[0.05, 0.01],
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
        help="Whether to treat the keys as sparse when loading them.",
    )
    parser.add_argument(
        "--class-column",
        default="speaker",
        help="Column name with original speaker identifiers in the metadata.",
    )
    parser.add_argument(
        "--anon-class-column",
        default="pseudo_speaker",
        help="Column name with anonymized speaker identifiers, if provided.",
    )
    parser.add_argument(
        "--calibrate-on-orig",
        default=False,
        action=ActionYesNo,
        help="Fit a calibration model on original-original scores before scoring.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Destination CSV/TSV file where aggregated metrics are written.",
    )
    parser.add_argument(
        "--output-fig-path",
        default=None,
        help="Directory to store optional privacy and consistency histograms.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="Verbosity level: 0=warning, 1=info, 2=debug, 3=trace.",
    )

    args = parser.parse_args()
    kwargs: Dict[str, Any] = namespace_to_dict(args)
    config_logger(kwargs["verbose"])
    del kwargs["verbose"]
    del kwargs["cfg"]
    eval_verification_anonymization_metrics(**kwargs)


if __name__ == "__main__":
    main()
