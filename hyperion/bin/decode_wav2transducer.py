#!/usr/bin/env python
"""
 Copyright 2022 Johns Hopkins University  (Author: Yen-Ju Lu, Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""

import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
from hyperion.hyp_defs import config_logger, float_cpu, set_float_cpu
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import SequentialAudioReader as AR
from hyperion.np.augment import SpeechAugment
from hyperion.torch import TorchModelLoader as TML
from hyperion.torch.models.wav2transducer.beam_search import (beam_search,
                                                              greedy_search)
from hyperion.torch.narchs import AudioFeatsMVN as AF
from hyperion.torch.utils import open_device
from hyperion.utils import Utt2Info
from jsonargparse import (ActionConfigFile, ActionParser, ArgumentParser,
                          namespace_to_dict)


def init_device(use_gpu):
    set_float_cpu("float32")
    num_gpus = 1 if use_gpu else 0
    logging.info("initializing devices num_gpus={}".format(num_gpus))
    device = open_device(num_gpus=num_gpus)
    return device


def load_model(model_path, device):
    logging.info("loading model {}".format(model_path))
    model = TML.load(model_path)
    logging.info("transducer-model={}".format(model))
    model.to(device)
    model.eval()
    return model


def decode_one_batch(
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    x: torch.Tensor,
    decoding_method="beam_search",
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:
        - key: It indicates the setting used for decoding. For example,
               if greedy_search is used, it would be "greedy_search"
               If beam search with a beam size of 7 is used, it would be
               "beam_7"
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = model.device
    feature = x  # batch["inputs"]
    assert x.shape[0] == 1
    assert feature.ndim == 2

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    feature_lens = torch.Tensor([x.shape[1]]).int()

    encoder_out, hid_feats, encoder_out_lens = model.forward_feats(
        x=feature, x_lengths=feature_lens
    )

    hyps = []
    batch_size = encoder_out.size(0)

    encoder_out = encoder_out.permute(0, 2, 1)  # (N, C, T) ->(N, T, C)

    for i in range(batch_size):
        # fmt: off
        encoder_out_i = encoder_out[i:i + 1, :encoder_out_lens[i]]
        # fmt: on
        if decoding_method == "greedy_search":
            hyp = greedy_search(model=model, encoder_out=encoder_out_i)
        elif decoding_method == "beam_search":
            hyp = beam_search(model=model, encoder_out=encoder_out_i, beam=5)
        else:
            raise ValueError(f"Unsupported decoding method: {decoding_method}")
        hyps.append(sp.decode(hyp).split())

    logging.info("hyps:{}".format(" ".join(hyps[0])))

    if decoding_method == "greedy_search":
        return hyps[0]
    else:
        return hyps[0]


def decode_transducer(
    input_spec, output_spec, model_path, bpe_model, use_gpu, **kwargs
):

    device = init_device(use_gpu)
    model = load_model(model_path, device)

    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model)

    augmenter = None
    aug_df = None
    num_augs = 1

    ar_args = AR.filter_args(**kwargs)
    logging.info("opening output: %s" % (output_spec))
    with open(output_spec, "w") as writer:
        logging.info(
            "opening input stream: {} with args={}".format(input_spec, ar_args)
        )
        with AR(input_spec, **ar_args) as reader:
            while not reader.eof():
                t1 = time.time()
                key, x0, fs = reader.read(1)
                if len(key) == 0:
                    break

                x0 = x0[0]
                key0 = key[0]
                t2 = time.time()

                logging.info("processing utt %s" % (key0))
                for aug_id in range(num_augs):
                    t3 = time.time()
                    key, x = key0, x0  # augment(key0, x0, augmenter, aug_df, aug_id)
                    t4 = time.time()
                    with torch.no_grad():
                        x = torch.tensor(
                            x[None, :], dtype=torch.get_default_dtype()
                        ).to(device)

                        t5 = time.time()
                        tot_frames = x.shape[1]

                        logging.info(
                            "utt %s detected %d/%d (%.2f %%) speech frames"
                            % (
                                key,
                                x.shape[1],
                                tot_frames,
                                x.shape[1] / tot_frames * 100,
                            )
                        )

                        t6 = time.time()
                        if x.shape[1] == 0:
                            y = np.zeros((model.embed_dim,), dtype=float_cpu())
                        else:
                            y = decode_one_batch(model=model, sp=sp, x=x)

                    t7 = time.time()
                    writer.write(key + " " + " ".join(y) + "\n")

                    t8 = time.time()
                    read_time = t2 - t1
                    tot_time = read_time + t8 - t3
                    logging.info(
                        (
                            "utt %s total-time=%.3f read-time=%.3f "
                            "aug-time=%.3f feat-time=%.3f "
                            "vad-time=%.3f embed-time=%.3f write-time=%.3f "
                            "rt-factor=%.2f"
                        )
                        % (
                            key,
                            tot_time,
                            read_time,
                            t4 - t3,
                            t5 - t4,
                            t6 - t5,
                            t7 - t6,
                            t8 - t7,
                            x0.shape[0] / fs[0] / tot_time,
                        )
                    )


if __name__ == "__main__":

    parser = ArgumentParser(
        description=(
            "Extracts x-vectors from waveform computing " "acoustic features on the fly"
        )
    )

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--input", dest="input_spec", required=True)

    AR.add_class_args(parser)

    AF.add_class_args(parser, prefix="feats")

    parser.add_argument("--model-path", required=True)

    parser.add_argument("--bpe-model", required=True)

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

    decode_transducer(**namespace_to_dict(args))
