#!/usr/bin/env python
"""
 Copyright 2020 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import sys
import os
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
)
import time
import logging

import math
import numpy as np

from hyperion.hyp_defs import config_logger
from hyperion.io import SequentialAudioReader as AR
from hyperion.io import DataWriterFactory as DWF


def pack_wav_rirs(input_path, output_spec, **kwargs):

    writer = DWF.create(output_spec, scp_sep=" ", compress=False)
    t1 = time.time()
    with AR(input_path, wav_scale=1) as reader:
        for data in reader:
            key, h, fs = data
            if h.ndim == 2:
                h = h[:, 0]
            h_delay = np.argmax(np.abs(h))
            h_max = h[h_delay]
            h /= h_max
            h[h < 1e-3] = 0
            h = np.trim_zeros(h)
            logging.info(
                "Packing rir %s h_max=%f h_delay=%d h-length=%d"
                % (key, h_max, h_delay, len(h))
            )
            writer.write([key], [h])

    logging.info("Packed RIRS elapsed-time=%.f" % (time.time() - t1))


if __name__ == "__main__":

    parser = ArgumentParser(description="Packs RIRs in wave format to h5/ark files")

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--input", dest="input_path", required=True)
    parser.add_argument("--output", dest="output_spec", required=True)
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="Verbose level",
    )
    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    pack_wav_rirs(**namespace_to_dict(args))
