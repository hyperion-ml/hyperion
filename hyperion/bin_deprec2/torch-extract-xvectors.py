#!/usr/bin/env python
"""
 Copyright 2019 Jesus Villalba (Johns Hopkins University)
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

import numpy as np

import torch

from hyperion.hyp_defs import config_logger, float_cpu, set_float_cpu
from hyperion.utils import Utt2Info
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import SequentialDataReaderFactory as DRF
from hyperion.io import VADReaderFactory as VRF
from hyperion.np.feats import MeanVarianceNorm as MVN

from hyperion.torch.utils import open_device
from hyperion.torch import TorchModelLoader as TML


def init_device(use_gpu):
    set_float_cpu("float32")
    num_gpus = 1 if use_gpu else 0
    logging.info("initializing devices num_gpus={}".format(num_gpus))
    device = open_device(num_gpus=num_gpus)
    return device


def init_mvn(device, **kwargs):
    mvn_args = MVN.filter_args(**kwargs["mvn"])
    logging.info("mvn args={}".format(mvn_args))
    mvn = MVN(**mvn_args)
    if mvn.norm_mean or mvn.norm_var:
        return mvn
    return None


def load_model(model_path, device):
    logging.info("loading model {}".format(model_path))
    model = TML.load(model_path)
    logging.info("xvector-model={}".format(model))
    model.to(device)
    model.eval()
    return model


def select_random_chunk(key, x, min_utt_length, max_utt_length, rng):
    utt_length = rng.randint(low=min_utt_length, high=max_utt_length + 1)
    if utt_length < x.shape[1]:
        first_frame = rng.randint(low=0, high=x.shape[1] - utt_length)
        x = x[:, first_frame : first_frame + utt_length]
        logging.info(
            "extract-random-utt %s of length=%d first-frame=%d"
            % (key, x.shape[1], first_frame)
        )
    return x


def extract_xvectors(
    input_spec,
    output_spec,
    vad_spec,
    write_num_frames_spec,
    vad_path_prefix,
    model_path,
    chunk_length,
    embed_layer,
    random_utt_length,
    min_utt_length,
    max_utt_length,
    use_gpu,
    **kwargs
):

    logging.info("initializing")
    rng = np.random.RandomState(seed=1123581321 + kwargs["part_idx"])
    device = init_device(use_gpu)
    mvn = init_mvn(device, **kwargs)
    model = load_model(model_path, device)

    if write_num_frames_spec is not None:
        keys = []
        info = []

    dr_args = DRF.filter_args(**kwargs)
    logging.info("opening output stream: %s" % (output_spec))
    with DWF.create(output_spec) as writer:

        logging.info("opening input stream: %s" % (input_spec))
        with DRF.create(input_spec, **dr_args) as reader:
            if vad_spec is not None:
                logging.info("opening VAD stream: %s" % (vad_spec))
                v_reader = VRF.create(vad_spec, path_prefix=vad_path_prefix)

            while not reader.eof():
                t1 = time.time()
                key, data = reader.read(1)
                if len(key) == 0:
                    break
                t2 = time.time()
                logging.info("processing utt %s" % (key[0]))
                x = data[0]
                if mvn is not None:
                    x = mvn.normalize(x)
                t3 = time.time()
                tot_frames = x.shape[0]
                if vad_spec is not None:
                    vad = v_reader.read(key, num_frames=x.shape[0])[0].astype(
                        "bool", copy=False
                    )
                    x = x[vad]

                logging.info(
                    "utt %s detected %d/%d (%.2f %%) speech frames"
                    % (key[0], x.shape[0], tot_frames, x.shape[0] / tot_frames * 100)
                )

                if random_utt_length:
                    x = select_random_chunk(key, x, min_utt_length, max_utt_length, rng)

                t4 = time.time()
                if x.shape[0] == 0:
                    y = np.zeros((model.embed_dim,), dtype=float_cpu())
                else:
                    xx = torch.tensor(x.T[None, :], dtype=torch.get_default_dtype())
                    with torch.no_grad():
                        y = (
                            model.extract_embed(
                                xx, chunk_length=chunk_length, embed_layer=embed_layer
                            )
                            .detach()
                            .cpu()
                            .numpy()[0]
                        )

                t5 = time.time()
                writer.write(key, [y])
                if write_num_frames_spec is not None:
                    keys.append(key[0])
                    info.append(str(x.shape[0]))
                t6 = time.time()
                logging.info(
                    (
                        "utt %s total-time=%.3f read-time=%.3f mvn-time=%.3f "
                        "vad-time=%.3f embed-time=%.3f write-time=%.3f "
                        "rt-factor=%.2f"
                    )
                    % (
                        key[0],
                        t6 - t1,
                        t2 - t1,
                        t3 - t2,
                        t4 - t3,
                        t5 - t4,
                        t6 - t5,
                        x.shape[0] * 1e-2 / (t6 - t1),
                    )
                )

    if write_num_frames_spec is not None:
        logging.info("writing num-frames to %s" % (write_num_frames_spec))
        u2nf = Utt2Info.create(keys, info)
        u2nf.save(write_num_frames_spec)


if __name__ == "__main__":

    parser = ArgumentParser(description="Extracts x-vectors from features")

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--input", dest="input_spec", required=True)
    DRF.add_class_args(parser)
    parser.add_argument("--vad", dest="vad_spec", default=None)
    parser.add_argument(
        "--write-num-frames", dest="write_num_frames_spec", default=None
    )
    # parser.add_argument('--scp-sep', default=' ',
    #                     help=('scp file field separator'))
    # parser.add_argument('--path-prefix', default=None,
    #                     help=('scp file_path prefix'))
    parser.add_argument(
        "--vad-path-prefix", default=None, help=("scp file_path prefix for vad")
    )

    MVN.add_class_args(parser, prefix="mvn")

    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=0,
        help=(
            "number of frames used in each forward pass of the x-vector encoder,"
            "if 0 the full utterance is used"
        ),
    )
    parser.add_argument(
        "--embed-layer",
        type=int,
        default=None,
        help=(
            "classifier layer to get the embedding from,"
            "if None the layer set in training phase is used"
        ),
    )

    parser.add_argument(
        "--random-utt-length",
        default=False,
        action="store_true",
        help="calculates x-vector from a random chunk of the utterance",
    )
    parser.add_argument(
        "--min-utt-length",
        type=int,
        default=500,
        help=("minimum utterance length when using random utt length"),
    )
    parser.add_argument(
        "--max-utt-length",
        type=int,
        default=12000,
        help=("maximum utterance length when using random utt length"),
    )

    parser.add_argument("--output", dest="output_spec", required=True)
    parser.add_argument(
        "--use-gpu", default=False, action="store_true", help="extract xvectors in gpu"
    )
    # parser.add_argument('--part-idx', dest='part_idx', type=int, default=1,
    #                     help=('splits the list of files in num-parts and process part_idx'))
    # parser.add_argument('--num-parts', dest='num_parts', type=int, default=1,
    #                     help=('splits the list of files in num-parts and process part_idx'))
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    extract_xvectors(**namespace_to_dict(args))
