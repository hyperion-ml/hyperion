"""
 Copyright 2024 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""

import logging
from pathlib import Path

import pandas as pd
from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict

from hyperion.hyp_defs import config_logger, float_cpu, set_float_cpu
from hyperion.utils import TrialKey, TrialNdx, TrialScores


def cm_scores_to_asvspoof5_format(ndx_file, in_score_file, out_score_file):
    logging.info("%s + %s -> %s", ndx_file, in_score_file, out_score_file)
    try:
        ndx = TrialKey.load(ndx_file).to_ndx()
    except:
        ndx = TrialNdx.load(ndx_file)

    scores = TrialScores.load(in_score_file)
    scores = scores.align_with_ndx(ndx)
    idx = scores.score_mask[0, :]
    df = pd.DataFrame(
        {"filename": scores.seg_set[idx], "cm-score": scores.scores[0, idx]}
    )
    df.sort_values(by=["filename"], inplace=True)

    output_dir = Path(out_score_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_score_file, sep="\t", index=False)


def main():
    parser = ArgumentParser(description="""Transform Spoofing logits to TrialScores""")

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--ndx-file", required=True)
    parser.add_argument("--in-score-file", required=True)
    parser.add_argument("--out-score-file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    del args.cfg
    logging.debug(args)

    cm_scores_to_asvspoof5_format(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
