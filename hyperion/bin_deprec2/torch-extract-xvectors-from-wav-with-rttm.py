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
import pandas as pd

import torch

from hyperion.hyp_defs import config_logger, float_cpu, set_float_cpu
from hyperion.utils import Utt2Info, RTTM
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import SequentialAudioReader as AR
from hyperion.io import VADReaderFactory as VRF
from hyperion.np.augment import SpeechAugment

from hyperion.torch.utils import open_device
from hyperion.torch.narchs import AudioFeatsMVN as AF
from hyperion.torch import TorchModelLoader as TML


def init_device(use_gpu):
    set_float_cpu("float32")
    num_gpus = 1 if use_gpu else 0
    logging.info("initializing devices num_gpus={}".format(num_gpus))
    device = open_device(num_gpus=num_gpus)
    return device


def init_feats(device, **kwargs):
    feat_args = AF.filter_args(**kwargs["feats"])
    logging.info("feat args={}".format(feat_args))
    logging.info("initializing feature extractor")
    feat_extractor = AF(trans=False, **feat_args)
    logging.info("feat-extractor={}".format(feat_extractor))
    feat_extractor.eval()
    feat_extractor.to(device)
    return feat_extractor


def load_model(model_path, device):
    logging.info("loading model {}".format(model_path))
    model = TML.load(model_path)
    logging.info("xvector-model={}".format(model))
    model.to(device)
    model.eval()
    return model


def augment(key0, x0, augmenter, aug_df, aug_id):
    if augmenter is None:
        x = x0
        key = key0
    else:
        x, aug_info = augmenter(x0)
        key = "%s-aug-%02d" % (key0, aug_id)
        aug_df_row = {
            "key_aug": key,
            "key_orig": key0,
            "noise_type": aug_info["noise"]["noise_type"],
            "snr": aug_info["noise"]["snr"],
            "rir_type": aug_info["reverb"]["rir_type"],
            "srr": aug_info["reverb"]["srr"],
            "sdr": aug_info["sdr"],
        }

        aug_df.append(pd.DataFrame(aug_df_row, index=[0]))

    return key, x


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
    rttm_file,
    scp_sep,
    model_path,
    chunk_length,
    embed_layer,
    random_utt_length,
    min_utt_length,
    max_utt_length,
    aug_cfg,
    num_augs,
    aug_info_path,
    use_gpu,
    **kwargs
):

    rng = np.random.RandomState(seed=1123581321 + kwargs["part_idx"])
    device = init_device(use_gpu)
    feat_extractor = init_feats(device, **kwargs)
    model = load_model(model_path, device)

    aug_df = []
    if aug_cfg is not None:
        augmenter = SpeechAugment.create(aug_cfg, rng=rng)
    else:
        augmenter = None
        num_augs = 1

    min_samples = int(feat_extractor.fs * feat_extractor.frame_length / 1000)

    ar_args = AR.filter_args(**kwargs)
    logging.info("opening output stream: %s" % (output_spec))
    with DWF.create(output_spec, scp_sep=scp_sep) as writer:

        logging.info(
            "opening input stream: {} with args={}".format(input_spec, ar_args)
        )
        with AR(input_spec, **ar_args) as reader:
            rttm = RTTM.load(rttm_file)
            rttm = rttm.filter(reader.scp.key)
            while not reader.eof():
                t1 = time.time()
                key, x0, fs = reader.read(1)
                if len(key) == 0:
                    break

                x0 = x0[0]
                key0 = key[0]
                t2 = time.time()

                spk_names = rttm.get_uniq_names_for_file(key0)
                num_spks = len(spk_names)
                logging.info("processing utt %s num-spks=%d" % (key0, num_spks))

                for aug_id in range(num_augs):
                    t3 = time.time()
                    key, x = augment(key0, x0, augmenter, aug_df, aug_id)

                    x_total = x
                    max_samples = x.shape[0]
                    y = np.zeros(
                        (
                            num_spks,
                            model.embed_dim,
                        ),
                        dtype=float_cpu(),
                    )
                    val_spks = np.ones((num_spks,), dtype=np.bool)
                    for spk_id in range(num_spks):
                        t4 = time.time()
                        spk_name = spk_names[spk_id]
                        mask = rttm.get_bin_sample_mask_for_spk(
                            key0, spk_name, feat_extractor.fs, max_samples=max_samples
                        )
                        x = x_total[mask]
                        num_speech_samples = x.shape[0]
                        logging.info(
                            "utt %s spk-name %s %d/%d (%.2f %%) speech samples"
                            % (
                                key,
                                spk_name,
                                num_speech_samples,
                                max_samples,
                                num_speech_samples / max_samples * 100,
                            )
                        )

                        if num_speech_samples < min_samples:
                            val_spks[spk_id] = False
                            logging.info(
                                "utt %s spk-name %s %d < %d  speech samples, skipping"
                                % (key, spk_name, num_speech_samples, min_samples)
                            )
                            continue

                        with torch.no_grad():
                            x = torch.tensor(
                                x[None, :], dtype=torch.get_default_dtype()
                            ).to(device)

                            x = feat_extractor(x)
                            t5 = time.time()
                            tot_frames = x.shape[1]
                            if random_utt_length:
                                x = select_random_chunk(
                                    key, x, min_utt_length, max_utt_length, rng
                                )

                            t6 = time.time()
                            if x.shape[1] > 0:
                                x = x.transpose(1, 2).contiguous()
                                y_i = (
                                    model.extract_embed(
                                        x,
                                        chunk_length=chunk_length,
                                        embed_layer=embed_layer,
                                    )
                                    .cpu()
                                    .numpy()[0]
                                )
                                y[spk_id, :] = y_i
                            t7 = time.time()
                            tot_time = t7 - t4
                            logging.info(
                                (
                                    "utt %s spk=%s total-time=%.3f feat-time=%.3f "
                                    "embed-time=%.3f write-time=%.3f "
                                    "rt-factor=%.2f"
                                )
                                % (
                                    key,
                                    spk_name,
                                    tot_time,
                                    t5 - t4,
                                    t6 - t5,
                                    t7 - t6,
                                    num_speech_samples / fs[0] / tot_time,
                                )
                            )

                    if not np.any(val_spks):
                        y = y[:1]  # if none are valid spks, we keep a 1xdim 0 vector
                    else:
                        y = y[val_spks]  # we keep speakers with at least 1 frame

                    writer.write([key], [y])

    if aug_info_path is not None:
        aug_df = pd.concat(aug_df, ignore_index=True)
        aug_df.to_csv(aug_info_path, index=False, na_rep="n/a")


if __name__ == "__main__":

    parser = ArgumentParser(
        description=(
            "Extract x-vectors from waveform computing " "acoustic features on the fly"
        )
    )

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--input", dest="input_spec", required=True)
    parser.add_argument("--scp-sep", default=" ", help=("scp file field separator"))
    parser.add_argument("--rttm-file", required=True, help=("RTTM file path"))

    AR.add_class_args(parser)

    parser.add_argument("--aug-cfg", default=None)
    parser.add_argument("--aug-info-path", default=None)
    parser.add_argument(
        "--num-augs", default=1, type=int, help="number of augmentations per utterance"
    )

    AF.add_class_args(parser, prefix="feats")

    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=0,
        help=(
            "number of frames used in each forward pass "
            "of the x-vector encoder,"
            "if 0 the full utterance is used"
        ),
    )
    parser.add_argument(
        "--embed-layer",
        type=int,
        default=None,
        help=(
            "classifier layer to get the embedding from, "
            "if None, it uses layer set in training phase"
        ),
    )

    parser.add_argument(
        "--random-utt-length",
        default=False,
        action="store_true",
        help="calculates x-vector from a random chunk",
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
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    extract_xvectors(**namespace_to_dict(args))
