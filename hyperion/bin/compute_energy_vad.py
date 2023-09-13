#!/usr/bin/env python
"""
 Copyright 2018 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import logging
import os
import sys
import time

import numpy as np
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import SequentialAudioReader as AR
from hyperion.np.feats import EnergyVAD


def compute_vad(recordings_file, output_spec, write_num_frames, **kwargs):
    vad_args = EnergyVAD.filter_args(**kwargs)
    vad = EnergyVAD(**vad_args)

    input_args = AR.filter_args(**kwargs)
    reader = AR(recordings_file, **input_args)

    metadata_columns = [
        "frame_shift",
        "frame_length",
        "num_frames",
        "num_speech_frames",
        "prob_speech",
    ]

    writer = DWF.create(output_spec, metadata_columns=metadata_columns)

    if write_num_frames is not None:
        f_num_frames = open(write_num_frames, "w")

    for data in reader:
        key, x, fs = data
        logging.info("Extracting VAD for %s", key)
        t1 = time.time()
        y = vad.compute(x)
        dt = (time.time() - t1) * 1000
        rtf = vad.frame_shift * y.shape[0] / dt
        num_speech_frames = np.sum(y)
        prob_speech = num_speech_frames / y.shape[0] * 100

        logging.info(
            "Extracted VAD for %s detected %d/%d (%f %%) speech frames, elapsed-time=%.2f ms. real-time-factor=%.2f",
            key,
            num_speech_frames,
            y.shape[0],
            prob_speech,
            dt,
            rtf,
        )
        metadata = {
            "frame_shift": vad.frame_shift,
            "frame_length": vad.frame_length,
            "num_frames": y.shape[0],
            "num_speech_frames": num_speech_frames,
            "prob_speech": prob_speech,
        }
        writer.write([key], [y], metadata)
        if write_num_frames is not None:
            f_num_frames.write("%s %d\n" % (key, y.shape[0]))

        vad.reset()

    if write_num_frames is not None:
        f_num_frames.close()


def main():
    parser = ArgumentParser(description="Compute Kaldi Energy VAD")

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--recordings-file", required=True)
    parser.add_argument("--output-spec", required=True)
    parser.add_argument("--write-num-frames", default=None)
    parser.add_argument("--write-stats", default=None)

    AR.add_class_args(parser)
    EnergyVAD.add_class_args(parser)
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

    compute_vad(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
