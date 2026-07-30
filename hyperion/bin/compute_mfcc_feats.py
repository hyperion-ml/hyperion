#!/usr/bin/env python
"""
Copyright 2018 Jesus Villalba (Johns Hopkins University)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
import sys
import time
from typing import Any, Optional

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
from hyperion.io import SequentialDataReaderFactory as DRF
from hyperion.io import compression_methods
from hyperion.np.feats import MFCC
from hyperion.utils.misc import PathLike


def compute_mfcc_feats(
    input_path: PathLike,
    output_path: PathLike,
    compress: bool,
    compression_method: str,
    write_num_frames: Optional[PathLike],
    **kwargs: Any,
) -> None:
    """Compute MFCC features from audio/waveform input and write them.

    Args:
        input_path: Input recordings/path specifier.
        output_path: Output feature wspecifier/path.
        compress: If ``True``, apply Kaldi-style feature compression.
        compression_method: Compression method used when ``compress`` is enabled.
        write_num_frames: Optional output file to store frame counts per utterance.
        **kwargs: Additional reader/MFCC configuration parsed from CLI.
    """
    mfcc_args = MFCC.filter_args(**kwargs)
    mfcc = MFCC(**mfcc_args)

    if mfcc.input_step == "wave":
        input_args = AR.filter_args(**kwargs)
        reader = AR(recordings=input_path, **input_args)
    else:
        input_args = DRF.filter_args(**kwargs)
        reader = DRF.create(input_path, **input_args)

    writer = DWF.create(
        output_path,
        compress=compress,
        compression_method=compression_method,
    )

    if write_num_frames is not None:
        f_num_frames = open(write_num_frames, "w")

    for data in reader:
        if mfcc.input_step == "wave":
            key, x, fs = data
        else:
            key, x = data
        logging.info("Extracting MFCC for %s num_samples=%d" % (key, len(x)))
        t1 = time.time()
        y = mfcc.compute(x)
        dt = (time.time() - t1) * 1000
        rtf = dt / (mfcc.frame_shift * y.shape[0])
        logging.info(
            "Extracted MFCC for %s num-frames=%d elapsed-time=%.2f ms. real-time-factor=%.2f",
            key,
            y.shape[0],
            dt,
            rtf,
        )
        writer.write([key], [y])

        if write_num_frames is not None:
            f_num_frames.write("%s %d\n" % (key, y.shape[0]))

        mfcc.reset()

    if write_num_frames is not None:
        f_num_frames.close()


def main() -> None:
    """Parse CLI arguments and run MFCC feature extraction."""
    parser = ArgumentParser(description="Compute MFCC features")

    parser.add_argument("--cfg", action=ActionConfigFile, help="configuration file")
    parser.add_argument(
        "--input",
        dest="input_path",
        required=True,
        help="input recordings rspecifier/path",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        required=True,
        help="output MFCC wspecifier/path",
    )
    parser.add_argument(
        "--write-num-frames",
        default=None,
        help="optional output file to write number of frames per utterance",
    )

    AR.add_class_args(parser)
    DRF.add_class_args(parser)
    MFCC.add_class_args(parser)
    parser.add_argument(
        "--compress",
        dest="compress",
        default=False,
        action="store_true",
        help="compress output features",
    )
    parser.add_argument(
        "--compression-method",
        dest="compression_method",
        default="auto",
        choices=compression_methods,
        help="compression method (used when --compress is set)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="verbosity level (0=error, 1=warning, 2=info, 3=debug)",
    )
    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    compute_mfcc_feats(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
