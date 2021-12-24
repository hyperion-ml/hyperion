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
from hyperion.io import RandomAccessAudioReader as AR
from hyperion.io import AudioWriter as Writer
from hyperion.io import VADReaderFactory as VRF


def make_noise(xs):

    lens = np.array([x.shape[0] for x in xs])
    max_len = np.max(lens)
    num_tiles = np.ceil(max_len / lens)
    for i in range(len(xs)):
        xs[i] = np.tile(xs[i], int(num_tiles[i]))[:max_len]

    for i in range(1, len(xs)):
        xs[0] += xs[i] - xs[i].mean()

    return xs[0]


def make_babble_noise_audio_files(
    input_path,
    output_path,
    output_script,
    write_time_durs_spec,
    min_spks=3,
    max_spks=7,
    num_reuses=5,
    random_seed=112358,
    **kwargs
):

    input_args = AR.filter_args(**kwargs)
    output_args = Writer.filter_args(**kwargs)
    logging.info("input_args={}".format(input_args))
    logging.info("output_args={}".format(output_args))

    rng = np.random.RandomState(seed=random_seed)

    if write_time_durs_spec is not None:
        okeys = []
        info = []

    count = 0
    t1 = time.time()
    with AR(input_path, **input_args) as reader:
        keys = reader.keys
        with Writer(output_path, output_script, **output_args) as writer:

            for iters in range(num_reuses):
                keys = rng.permutation(keys)

                cur_spks = min_spks
                utt_list = []
                for utt_idx in range(len(keys)):
                    if len(utt_list) < cur_spks:
                        utt_list.append(keys[utt_idx])
                        continue

                    x, fs = reader.read(utt_list)
                    fs = fs[0]
                    y = make_noise(x)
                    babble_id = "babble-%05d" % (count)
                    logging.info("writing file % s" % (babble_id))
                    writer.write([babble_id], [y], [fs])
                    if write_time_durs_spec is not None:
                        okeys.append(babble_id)
                        info.append(y.shape[0] / fs)

                    count += 1
                    utt_list = []
                    cur_spks += 1
                    if cur_spks > max_spks:
                        cur_spks = min_spks

    if write_time_durs_spec is not None:
        logging.info("writing time durations to %s" % (write_time_durs_spec))
        u2td = Utt2Info.create(okeys, info)
        u2td.save(write_time_durs_spec)

    logging.info("finished making babble files, elapsed-time=%f" % (time.time() - t1))


if __name__ == "__main__":

    parser = ArgumentParser(description="Creates babble noise by adding speech files")

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--input", dest="input_path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--output-script", required=True)
    parser.add_argument("--write-time-durs", dest="write_time_durs_spec", default=None)

    AR.add_class_args(parser)
    Writer.add_class_args(parser)

    parser.add_argument("--min-spks", default=3, type=int)
    parser.add_argument("--max-spks", default=10, type=int)
    parser.add_argument("--num-reuses", default=5, type=int)
    parser.add_argument("--random-seed", default=112358, type=int)
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

    make_babble_noise_audio_files(**namespace_to_dict(args))
