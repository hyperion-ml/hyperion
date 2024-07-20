"""
 Copyright 2024 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""

import logging

from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict

from hyperion.hyp_defs import config_logger, float_cpu, set_float_cpu
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.utils import SegmentSet, TrialScores


def spoof_logits_to_trial_scores(segments_file, logits_spec, score_file):
    segments = SegmentSet.load(segments_file)
    reader = DRF.create(logits_spec)
    logits = reader.read(segments["id"], squeeze=True)
    scores = logits[:, 0] - logits[:, 1]
    modelids = ["bonafide"]
    scores = TrialScores(modelids, segments["id"].values, scores[None, :])
    scores.save(score_file)


def main():
    parser = ArgumentParser(description="""Transform Spoofing logits to TrialScores""")

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--segments-file", required=True)
    parser.add_argument("--logits-spec", required=True)
    parser.add_argument("--score-file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    del args.cfg
    logging.debug(args)

    spoof_logits_to_trial_scores(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
