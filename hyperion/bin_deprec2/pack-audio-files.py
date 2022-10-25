#!/usr/bin/env python
"""
 Copyright 2020 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import sys
import os
import argparse
import time
import logging

import math
import numpy as np
from scipy import signal, ndimage

from hyperion.hyp_defs import config_logger
from hyperion.io import SequentialAudioReader as AR
from hyperion.io import PackedAudioWriter as Writer
from hyperion.io import VADReaderFactory as VRF
from hyperion.io import WSpecifier as WS


def process_vad(vad, length, fs, dilation, erosion):
    vad = signal.resample(vad, length) > 0.5
    if dilation > 0:
        iters = int(dilation * fs)
        vad = ndimage.binary_dilation(vad, iterations=iters)

    if erosion > 0:
        iters = int(erosion * fs)
        vad = ndimage.binary_erosion(vad, iterations=iters, border_value=True)

    return vad


def pack_audio_files(
    input_path,
    output_spec,
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

    output_spec = WS.create(output_spec)
    with AR(input_path, **input_args) as reader:
        with Writer(output_spec.archive, output_spec.script, **output_args) as writer:

            if vad_spec is not None:
                logging.info("opening VAD stream: %s" % (vad_spec))
                v_reader = VRF.create(vad_spec, path_prefix=vad_path_prefix)

            t1 = time.time()
            for data in reader:
                key, x, fs_i = data
                assert writer.fs == fs_i
                logging.info("Packing audio %s" % (key))
                t2 = time.time()

                tot_samples = x.shape[0]
                if vad_spec is not None:
                    num_vad_frames = int(round(tot_samples * vad_fs / fs_i))
                    vad = v_reader.read(key, num_frames=num_vad_frames)[0].astype(
                        "bool", copy=False
                    )
                    logging.info("vad=%d/%d" % (np.sum(vad == 1), len(vad)))
                    vad = process_vad(vad, tot_samples, fs_i, vad_dilation, vad_erosion)
                    logging.info("vad=%d/%d" % (np.sum(vad == 1), len(vad)))
                    x = x[vad]

                logging.info(
                    "utt %s detected %f/%f secs (%.2f %%) speech "
                    % (
                        key[0],
                        x.shape[0] / fs_i,
                        tot_samples / fs_i,
                        x.shape[0] / tot_samples * 100,
                    )
                )

                if remove_dc_offset:
                    x -= np.mean(x)

                writer.write([key], [x])
                t3 = time.time()
                dt2 = (t2 - t1) * 1000
                dt3 = (t3 - t1) * 1000
                time_dur = len(x) / writer.fs
                rtf = (time_dur * 1000) / dt3
                logging.info(
                    (
                        "Packed audio %s length=%0.3f secs "
                        "elapsed-time=%.2f ms. "
                        "read-time=%.2f ms. write-time=%.2f ms. "
                        "real-time-factor=%.2f"
                        "x-range=[%f-%f]"
                    )
                    % (key, time_dur, dt3, dt2, dt3 - dt2, rtf, np.min(x), np.max(x))
                )
                t1 = time.time()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Packs multiple audio files into single audio file",
    )

    parser.add_argument("--input", dest="input_path", required=True)
    parser.add_argument("--output", dest="output_spec", required=True)
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

    AR.add_argparse_args(parser)
    Writer.add_argparse_args(parser)
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

    pack_audio_files(**vars(args))
