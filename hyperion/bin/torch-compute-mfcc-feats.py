#!/usr/bin/env python
"""
 Copyright 2018 Jesus Villalba (Johns Hopkins University)
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

import torch

from hyperion.hyp_defs import config_logger
from hyperion.io import SequentialAudioReader as AR
from hyperion.io import SequentialDataReaderFactory as DRF
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import compression_methods
from hyperion.torch.layers import AudioFeatsFactory as AFF


def compute_mfcc_feats(
    input_path, output_path, compress, compression_method, write_num_frames, **kwargs
):

    mfcc_args = AFF.filter_args(**kwargs)
    mfcc = AFF.create(**mfcc_args)
    input_args = AR.filter_args(**kwargs)
    reader = AR(input_path, **input_args)
    writer = DWF.create(
        output_path,
        scp_sep=" ",
        compress=compress,
        compression_method=compression_method,
    )

    if write_num_frames is not None:
        f_num_frames = open(write_num_frames, "w")

    for data in reader:
        key, x, fs = data
        logging.info("Extracting MFCC for %s" % (key))
        t1 = time.time()
        x = torch.tensor(x[None, :], dtype=torch.get_default_dtype())
        y = mfcc(x).squeeze(0).detach().numpy()

        dt = (time.time() - t1) * 1000
        rtf = mfcc.frame_shift * y.shape[0] / dt
        logging.info(
            "Extracted MFCC for %s num-frames=%d elapsed-time=%.2f ms. real-time-factor=%.2f"
            % (key, y.shape[0], dt, rtf)
        )
        writer.write([key], [y])

        if write_num_frames is not None:
            f_num_frames.write("%s %d\n" % (key, y.shape[0]))

    if write_num_frames is not None:
        f_num_frames.close()


if __name__ == "__main__":

    parser = ArgumentParser(description="Compute MFCC features in pytorch")

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--input", dest="input_path", required=True)
    parser.add_argument("--output", dest="output_path", required=True)
    parser.add_argument("--write-num-frames", dest="write_num_frames", default=None)

    AR.add_class_args(parser)
    AFF.add_class_args(parser)
    parser.add_argument(
        "--compress",
        dest="compress",
        default=False,
        action="store_true",
        help="Compress the features",
    )
    parser.add_argument(
        "--compression-method",
        dest="compression_method",
        default="auto",
        choices=compression_methods,
        help="Compression method",
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

    compute_mfcc_feats(**namespace_to_dict(args))
