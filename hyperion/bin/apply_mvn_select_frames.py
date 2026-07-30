#!/usr/bin/env python
"""
Copyright 2019 Jesus Villalba (Johns Hopkins University)
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
from hyperion.io import RandomAccessDataReaderFactory as RDRF
from hyperion.io import SequentialDataReaderFactory as DRF
from hyperion.np.feats import FrameSelector as FSel
from hyperion.np.feats import MeanVarianceNorm as MVN
from hyperion.utils import Utt2Info
from hyperion.utils.kaldi_matrix import compression_methods
from hyperion.utils.misc import PathLike


def process_feats(
    input_spec: PathLike,
    output_spec: PathLike,
    vad_spec: Optional[PathLike],
    write_num_frames_spec: Optional[PathLike],
    path_prefix: Optional[PathLike],
    vad_path_prefix: Optional[PathLike],
    part_idx: int,
    num_parts: int,
    compress: bool,
    compression_method: str,
    **kwargs: Any,
) -> None:
    """Apply mean-variance normalization and optional VAD frame selection.

    Args:
        input_spec: Input feature rspecifier/path.
        output_spec: Output feature wspecifier/path.
        vad_spec: Optional VAD rspecifier/path with frame-level speech decisions.
        write_num_frames_spec: Optional output path to store kept-frame counts.
        path_prefix: Optional prefix for paths referenced by ``input_spec`` scp.
        vad_path_prefix: Optional prefix for paths referenced by ``vad_spec`` scp.
        part_idx: 1-based part index to process when splitting work.
        num_parts: Total number of parts used to split work.
        compress: If ``True``, apply lossy Kaldi-style feature compression.
        compression_method: Compression method used when ``compress`` is enabled.
        **kwargs: Extra MVN/frame-selector arguments parsed from the CLI.
    """
    logging.info("initializing")
    mvn_args = MVN.filter_args(**kwargs)
    mvn = MVN(**mvn_args)
    if vad_spec is not None:
        fs_args = FSel.filter_args(**kwargs)
        fs = FSel(**fs_args)

    if write_num_frames_spec is not None:
        keys = []
        info = []

    logging.info("opening output stream: %s" % (output_spec))
    with DWF.create(
        output_spec,
        compress=compress,
        compression_method=compression_method,
    ) as writer:
        logging.info("opening input stream: %s" % (output_spec))
        with DRF.create(
            input_spec,
            path_prefix=path_prefix,
            part_idx=part_idx,
            num_parts=num_parts,
        ) as reader:
            if vad_spec is not None:
                logging.info("opening VAD stream: %s" % (vad_spec))
                v_reader = RDRF.create(
                    vad_spec,
                    path_prefix=vad_path_prefix,
                )

            while not reader.eof():
                key, data = reader.read(1)
                if len(key) == 0:
                    break
                logging.info("processing feats at %s" % (key[0]))
                x = mvn.normalize(data[0])
                if vad_spec is not None:
                    vad = v_reader.read(key)[0].astype("bool")
                    tot_frames = x.shape[0]
                    x = fs.select(x, vad)
                    logging.info(
                        "for %s detected %d/%d (%.2f %%) speech frames"
                        % (
                            key[0],
                            x.shape[0],
                            tot_frames,
                            x.shape[0] / tot_frames * 100,
                        )
                    )
                if x.shape[0] > 0:
                    writer.write(key, [x])
                    if write_num_frames_spec is not None:
                        keys += key
                        info.append(x.shape[0])

    if write_num_frames_spec is not None:
        logging.info("writing num-frames to %s" % (write_num_frames_spec))
        u2nf = Utt2Info.create(keys, info)
        u2nf.save(write_num_frames_spec)


def main() -> None:
    """Parse CLI arguments and apply MVN with optional frame selection."""
    parser = ArgumentParser(
        description="Apply mean-variance normalization and optional VAD-based frame selection"
    )

    parser.add_argument(
        "--input",
        dest="input_spec",
        required=True,
        help="input feature rspecifier or file path",
    )
    parser.add_argument(
        "--output",
        dest="output_spec",
        required=True,
        help="output feature wspecifier or file path",
    )
    parser.add_argument(
        "--vad",
        dest="vad_spec",
        default=None,
        help="optional VAD rspecifier/path used to select speech frames",
    )
    parser.add_argument(
        "--write-num-frames",
        dest="write_num_frames_spec",
        default=None,
        help="optional output path to write number of selected frames per utterance",
    )
    parser.add_argument(
        "--path-prefix",
        dest="path_prefix",
        default=None,
        help="optional prefix prepended to file paths in input scp entries",
    )
    parser.add_argument(
        "--vad-path-prefix",
        default=None,
        help="optional prefix prepended to file paths in VAD scp entries",
    )
    parser.add_argument(
        "--part-idx",
        type=int,
        default=1,
        help="1-based part index to process when splitting input into --num-parts",
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        default=1,
        help="total number of parts used to split the input list",
    )

    parser.add_argument(
        "--compress",
        default=False,
        action="store_true",
        help="apply lossy compression when writing output features",
    )
    parser.add_argument(
        "--compression-method",
        default="auto",
        choices=compression_methods,
        help=(
            "compression method (used when --compress is set): "
            "{auto (default), speech_feat, "
            "2byte-auto, 2byte-signed-integer, "
            "1byte-auto, 1byte-unsigned-integer, 1byte-0-1}."
        ),
    )
    MVN.add_argparse_args(parser)
    FSel.add_argparse_args(parser)

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

    process_feats(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
