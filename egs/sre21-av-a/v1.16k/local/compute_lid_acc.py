#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""

import os
import logging
from jsonargparse import ArgumentParser, namespace_to_dict

import math
import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils import Utt2Info

if __name__ == "__main__":

    parser = ArgumentParser(description="Compute LID Acc in SRE21")

    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--gt-file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    u2p = Utt2Info.load(args.pred_file)
    u2g = Utt2Info.load(args.gt_file)

    u2g = u2g.filter_info(["OTHER"], keep=False)
    u2p = u2p.filter(u2g.key)
    n = 0
    c = 0
    for kp, p, kg, g in zip(u2p.key, u2p.info, u2g.key, u2g.info):
        assert kp == kg
        n += 1
        if p == g:
            c += 1

    logging.info("Acc: %f", c / n * 100)
