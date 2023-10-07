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
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger, float_cpu, set_float_cpu
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import SequentialAudioReader as AR
from hyperion.np.augment import SpeechAugment
from hyperion.torch import TorchModelLoader as TML
from hyperion.torch.data.char_piece import CharPieceProcessor
from hyperion.torch.models import HFWav2Vec2RNNTransducer
from hyperion.torch.models.wav2transducer.beam_search import beam_search, greedy_search
from hyperion.torch.narchs import AudioFeatsMVN as AF
from hyperion.torch.utils import open_device
from hyperion.utils import Utt2Info


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
    input_spec,
    output_spec,
    scp_sep,
    model_path,
    bpe_model,
    infer_args,
    use_gpu,
    **kwargs,
):
    device = init_device(use_gpu)
    model = load_model(model_path, device)



    if bpe_model.endswith(".txt"):
        logging.info("loading char piece file %s", bpe_model)
        sp = CharPieceProcessor()
        sp.load(open(bpe_model).read().split())    
    else:
        logging.info("bpe-model=%s", bpe_model)
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    infer_args = HFWav2Vec2RNNTransducer.filter_infer_args(**infer_args)
    logging.info(f"infer-args={infer_args}")

    ar_args = AR.filter_args(**kwargs)
    logging.info("opening output: %s", output_spec)
    with open(output_spec, "w") as writer:
        logging.info(f"opening input stream: {input_spec} with args={ar_args}")
        with AR(input_spec, **ar_args) as reader:
            while not reader.eof():
                t1 = time.time()
                key, x, fs = reader.read(1)
                if len(key) == 0:
                    break

                x, key, fs = x[0], key[0], fs[0]
                t2 = time.time()
                logging.info("processing utt %s", key)
                with torch.no_grad():
                    x = torch.tensor(x[None, :], dtype=torch.get_default_dtype()).to(
                        device
                    )

                    tot_frames = x.shape[1]
                    logging.info(
                        "utt %s detected %d/%d (%.2f %%) speech frames",
                        key,
                        x.shape[1],
                        tot_frames,
                        x.shape[1] / tot_frames * 100,
                    )

                    if x.shape[1] == 0:
                        y = [""]
                    else:
                        # y = decode_one_batch(model=model, sp=sp, x=x)
                        x_lengths = torch.tensor(
                            (x.shape[1],), dtype=torch.long, device=device
                        )
                        y = model.infer(x, x_lengths, **infer_args)

                    y = sp.decode(y[0])
                    logging.info(f"utt: {key} hyps: {y}")
                    t3 = time.time()
                    writer.write(f"{key} {y}\n")

                    t4 = time.time()
                    tot_time = t4 - t1
                    infer_time = t3 - t2
                    logging.info(
                        (
                            "utt %s total-time=%.3f read-time=%.3f "
                            "infer-time=%.3f "
                            "write-time=%.3f "
                            "infer-rt-factor=%.2f tot-rt-factor=%.2f"
                        ),
                        key,
                        tot_time,
                        t2 - t1,
                        infer_time,
                        t4 - t3,
                        x.shape[1] / fs / infer_time,
                        x.shape[1] / fs / tot_time,
                    )


def main():
    parser = ArgumentParser(
        description=("ASR decoding for RNN-T with Wav2vec features")
    )

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--input", dest="input_spec", required=True)
    parser.add_argument("--scp-sep", default=" ", help=("scp file field separator"))

    AR.add_class_args(parser)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--bpe-model", required=True)

    HFWav2Vec2RNNTransducer.add_infer_args(parser, "infer-args")
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


if __name__ == "__main__":
    main()
