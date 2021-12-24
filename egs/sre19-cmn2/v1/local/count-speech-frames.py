#!/usr/bin/env python
"""
Trains Backend for SRE18 video condition
"""


import sys
import os
import argparse
import time

import numpy as np

from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.utils import Utt2Info


def count_speech_frames(vad_file, list_file, output_file):

    u2i = Utt2Info.load(list_file)
    r = DRF.create(vad_file)
    with open(output_file, "w") as f:
        for key in u2i.key:
            vad = r.read(key)
            nf = np.sum(vad)
            f.write("%s %d\n" % (key, nf))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Train Back-end for SRE18 video condition",
    )

    parser.add_argument("--vad-file", dest="vad_file", required=True)
    parser.add_argument("--list-file", dest="list_file", required=True)
    parser.add_argument("--output-file", dest="output_file", required=True)

    args = parser.parse_args()

    count_speech_frames(**vars(args))
