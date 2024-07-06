"""
 Copyright 2024 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict

from hyperion.hyp_defs import config_logger, float_cpu, set_float_cpu
from hyperion.utils import TrialKey, TrialNdx, TrialScores


def asvspoof_scores_to_asvspoof5_format(
    ndx_file, asv_score_file, cm_score_file, asvspoof_score_file, out_score_file
):
    logging.info(
        "%s + %s + %s + %s -> %s",
        ndx_file,
        asv_score_file,
        cm_score_file,
        asvspoof_score_file,
        out_score_file,
    )
    try:
        ndx = TrialKey.load(ndx_file).to_ndx()
    except:
        ndx = TrialNdx.load(ndx_file)

    asv_scores = TrialScores.load(asv_score_file)
    asv_scores = asv_scores.align_with_ndx(ndx)
    asvspoof_scores = TrialScores.load(asvspoof_score_file)
    asvspoof_scores = asvspoof_scores.align_with_ndx(ndx)
    cm_scores = TrialScores.load(cm_score_file)
    cm_scores = cm_scores.filter(cm_scores.model_set, asv_scores.seg_set)

    I, J = asv_scores.score_mask.nonzero()
    modelids = asv_scores.model_set[I]
    segmentids = asv_scores.seg_set[J]
    asv_score_vec = asv_scores.scores[asv_scores.score_mask]
    asvspoof_score_vec = asvspoof_scores.scores[asv_scores.score_mask]
    cm_score_vec = np.tile(cm_scores.scores, (len(asv_scores.model_set), 1))[
        asv_scores.score_mask
    ]
    df = pd.DataFrame(
        {
            "filename": segmentids,
            "cm-score": cm_score_vec,
            "asv-score": asv_score_vec,
            "sasv-score": asvspoof_score_vec,
        }
    )
    df.sort_values(by=["filename"], inplace=True)

    output_dir = Path(out_score_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_score_file, sep="\t", index=False)


def main():
    parser = ArgumentParser(description="""Transform Spoofing logits to TrialScores""")

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--ndx-file", required=True)
    parser.add_argument("--asv-score-file", required=True)
    parser.add_argument("--cm-score-file", required=True)
    parser.add_argument("--asvspoof-score-file", required=True)
    parser.add_argument("--out-score-file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    del args.cfg
    logging.debug(args)

    asvspoof_scores_to_asvspoof5_format(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
