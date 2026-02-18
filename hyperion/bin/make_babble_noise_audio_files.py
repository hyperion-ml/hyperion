#!/usr/bin/env python
"""
 Copyright 2020 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import logging
import math
import os
import sys
import time
from typing import Any, List, Optional

import numpy as np
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.io import AudioWriter as Writer
from hyperion.io import RandomAccessAudioReader as AR
from hyperion.io import VADReaderFactory as VRF
from hyperion.utils import PathLike, Utt2Info


def make_noise(xs: List[np.ndarray], max_value: float) -> np.ndarray:
    """Create a babble-noise signal by summing mean-centered waveforms.

    Args:
        xs: List of 1D waveform arrays.
        max_value: Maximum absolute amplitude allowed in the output waveform.
    """
    lens = np.array([x.shape[0] for x in xs])
    max_len = np.max(lens)
    num_tiles = np.ceil(max_len / lens)
    for i in range(len(xs)):
        xs[i] = np.tile(xs[i], int(num_tiles[i]))[:max_len]

    xs[0] -= xs[0].mean()
    for i in range(1, len(xs)):
        xs[0] += xs[i] - xs[i].mean()

    max_x = np.max(np.abs(xs[0]))
    if max_x > max_value:
        xs[0] *= max_value / max_x

    return xs[0]


def make_babble_noise_audio_files(
    recordings_file: PathLike,
    output_path: PathLike,
    output_recordings_file: PathLike,
    write_time_durs: Optional[PathLike],
    min_spks: int = 3,
    max_spks: int = 7,
    num_reuses: int = 5,
    random_seed: int = 112358,
    **kwargs: Any,
) -> None:
    """Generate babble-noise recordings by mixing random speech utterances.

    Args:
        recordings_file: Input recordings file (SCP/list supported by audio reader).
        output_path: Output directory for generated babble audio files.
        output_recordings_file: Output recordings manifest file.
        write_time_durs: Optional file to store utterance durations in seconds.
        min_spks: Minimum number of utterances mixed per babble example.
        max_spks: Maximum number of utterances mixed per babble example.
        num_reuses: Number of shuffled passes over the input recordings.
        random_seed: Random seed for permutation reproducibility.
        **kwargs: Extra reader/writer options parsed from CLI.
    """
    input_args = AR.filter_args(**kwargs)
    output_args = Writer.filter_args(**kwargs)
    logging.info(f"input_args={input_args}")
    logging.info(f"output_args={output_args}")

    rng = np.random.default_rng(seed=random_seed)

    if write_time_durs is not None:
        okeys: List[str] = []
        info: List[float] = []

    count = 0
    t1 = time.time()
    with AR(recordings=recordings_file, **input_args) as reader, Writer(
        output_path, output_recordings_file, **output_args
    ) as writer:
        keys = reader.keys
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
                y = make_noise(x, reader.wav_scale)
                babble_id = "babble-%05d" % (count)
                logging.info("writing file %s", babble_id)
                writer.write([babble_id], [y], [fs])
                if write_time_durs is not None:
                    okeys.append(babble_id)
                    info.append(y.shape[0] / fs)

                count += 1
                utt_list = []
                cur_spks += 1
                if cur_spks > max_spks:
                    cur_spks = min_spks

    if write_time_durs is not None:
        logging.info("writing time durations to %s", write_time_durs)
        u2td = Utt2Info.create(okeys, info)
        u2td.save(write_time_durs)

    logging.info("finished making babble files, elapsed-time=%f", time.time() - t1)


def main() -> None:
    """Parse command-line arguments and create babble-noise audio files."""
    parser = ArgumentParser(description="Creates babble noise by adding speech files")

    parser.add_argument(
        "--cfg",
        action=ActionConfigFile,
        help="Path to a configuration file.",
    )
    parser.add_argument(
        "--recordings-file",
        required=True,
        help="Input recordings file used as source utterances.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output directory for generated babble audio files.",
    )
    parser.add_argument(
        "--output-recordings-file",
        required=True,
        help="Output recordings manifest for generated babble files.",
    )
    parser.add_argument(
        "--write-time-durs",
        default=None,
        help="Optional output file to store generated utterance durations.",
    )

    AR.add_class_args(parser)
    Writer.add_class_args(parser)

    parser.add_argument(
        "--min-spks",
        default=3,
        type=int,
        help="Minimum number of source utterances mixed per babble file.",
    )
    parser.add_argument(
        "--max-spks",
        default=10,
        type=int,
        help="Maximum number of source utterances mixed per babble file.",
    )
    parser.add_argument(
        "--num-reuses",
        default=5,
        type=int,
        help="Number of shuffled passes over the input recordings.",
    )
    parser.add_argument(
        "--random-seed",
        default=112358,
        type=int,
        help="Random seed for reproducible shuffling.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="Verbosity level: 0=error, 1=warning, 2=info, 3=debug.",
    )
    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    make_babble_noise_audio_files(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
