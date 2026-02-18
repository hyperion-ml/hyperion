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
from typing import Any

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
from hyperion.utils import PathLike


def pack_wav_rirs(input_path: PathLike, output_spec: PathLike, **kwargs: Any) -> None:
    """Pack RIR waveforms into feature archives after normalization and trimming.

    Args:
        input_path: Input recordings list/specifier for RIR wave files.
        output_spec: Output writer specifier (for example, ``ark,scp`` or ``h5``).
        **kwargs: Additional parsed CLI arguments.
    """
    writer = DWF.create(output_spec, compress=False)
    t1 = time.time()
    with AR(recordings=input_path, wav_scale=1) as reader:
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
                "Packing rir %s h_max=%f h_delay=%d h-length=%d",
                key,
                h_max,
                h_delay,
                len(h),
            )
            writer.write([key], [h])

    logging.info("Packed RIRS elapsed-time=%.f", time.time() - t1)


def main() -> None:
    """Parse CLI arguments and pack RIR waveforms into archive format."""
    parser = ArgumentParser(description="Packs RIRs in wave format to h5/ark files")

    parser.add_argument(
        "--cfg",
        action=ActionConfigFile,
        help="Path to a configuration file.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        required=True,
        help="Input recordings list/specifier with RIR wave files.",
    )
    parser.add_argument(
        "--output",
        dest="output_spec",
        required=True,
        help="Output data writer specifier (for example, ark/scp or h5 path).",
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

    pack_wav_rirs(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
