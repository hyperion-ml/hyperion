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
from scipy import signal, ndimage

from hyperion.hyp_defs import config_logger
from hyperion.utils import Utt2Info
from hyperion.io import SequentialAudioReader as AR
from hyperion.io import AudioWriter as Writer
from hyperion.io import VADReaderFactory as VRF


def process_vad(vad, length, fs, dilation, erosion):
    vad = signal.resample(vad, length) > 0.5
    if dilation > 0:
        iters = int(dilation * fs)
        vad = ndimage.binary_dilation(vad, iterations=iters)

    if erosion > 0:
        iters = int(erosion * fs)
        vad = ndimage.binary_erosion(vad, iterations=iters, border_value=True)

    return vad


def process_audio_files(
    input_path,
    output_path,
    output_script,
    write_time_durs_spec,
    vad_spec,
    vad_path_prefix,
    vad_fs=100,
    vad_dilation=0,
    vad_erosion=0,
    remove_dc_offset=False,
    **kwargs
):

    input_args = AR.filter_args(**kwargs)
    output_args = Writer.filter_args(**kwargs)
    logging.info("input_args={}".format(input_args))
    logging.info("output_args={}".format(output_args))

    if write_time_durs_spec is not None:
        keys = []
        info = []

    with AR(input_path, **input_args) as reader:
        with Writer(output_path, output_script, **output_args) as writer:

            if vad_spec is not None:
                logging.info("opening VAD stream: %s" % (vad_spec))
                v_reader = VRF.create(vad_spec, path_prefix=vad_path_prefix)

            t1 = time.time()
            for data in reader:
                key, x, fs = data
                logging.info("Processing audio %s" % (key))
                t2 = time.time()

                tot_samples = x.shape[0]
                if vad_spec is not None:
                    num_vad_frames = int(round(tot_samples * vad_fs / fs))
                    vad = v_reader.read(key, num_frames=num_vad_frames)[0].astype(
                        "bool", copy=False
                    )
                    logging.info("vad=%d/%d" % (np.sum(vad == 1), len(vad)))
                    vad = process_vad(vad, tot_samples, fs, vad_dilation, vad_erosion)
                    logging.info("vad=%d/%d" % (np.sum(vad == 1), len(vad)))
                    x = x[vad]

                logging.info(
                    "utt %s detected %f/%f secs (%.2f %%) speech "
                    % (
                        key[0],
                        x.shape[0] / fs,
                        tot_samples / fs,
                        x.shape[0] / tot_samples * 100,
                    )
                )

                if x.shape[0] > 0:
                    if remove_dc_offset:
                        x -= np.mean(x)

                    writer.write([key], [x], [fs])
                    if write_time_durs_spec is not None:
                        keys.append(key)
                        info.append(x.shape[0] / fs)

                    xmax = np.max(x)
                    xmin = np.min(x)
                else:
                    xmax = 0
                    xmin = 0

                t3 = time.time()
                dt2 = (t2 - t1) * 1000
                dt3 = (t3 - t1) * 1000
                time_dur = len(x) / fs
                rtf = (time_dur * 1000) / dt3
                logging.info(
                    (
                        "Packed audio %s length=%0.3f secs "
                        "elapsed-time=%.2f ms. "
                        "read-time=%.2f ms. write-time=%.2f ms. "
                        "real-time-factor=%.2f"
                        "x-range=[%f-%f]"
                    )
                    % (key, time_dur, dt3, dt2, dt3 - dt2, rtf, xmin, xmax)
                )
                t1 = time.time()

    if write_time_durs_spec is not None:
        logging.info("writing time durations to %s" % (write_time_durs_spec))
        u2td = Utt2Info.create(keys, info)
        u2td.save(write_time_durs_spec)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Process pipes in wav.scp file, optionally applies vad and save all audios in the same format"
    )

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--input", dest="input_path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--output-script", required=True)
    parser.add_argument("--write-time-durs", dest="write_time_durs_spec", default=None)
    parser.add_argument("--vad", dest="vad_spec", default=None)
    parser.add_argument(
        "--vad-path-prefix", default=None, help=("scp file_path prefix for vad")
    )

    parser.add_argument(
        "--vad-fs", default=100, type=float, help=("vad sampling frequency")
    )

    parser.add_argument(
        "--vad-dilation",
        default=0,
        type=float,
        help=("applies dilation operation to vad, in secs"),
    )

    parser.add_argument(
        "--vad-erosion",
        default=0,
        type=float,
        help=("applies erosion operation to vad (after dilation), in secs"),
    )

    AR.add_class_args(parser)
    Writer.add_class_args(parser)
    parser.add_argument(
        "--remove-dc-offset",
        default=False,
        action="store_true",
        help="removes dc offset from file",
    )
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

    process_audio_files(**namespace_to_dict(args))
