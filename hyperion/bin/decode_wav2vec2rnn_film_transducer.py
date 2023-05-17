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
from hyperion.torch.models import HFWav2Vec2RNNFiLMTransducer
from hyperion.torch.models.wav2transducer.beam_search import (beam_search,
                                                              greedy_search)
from hyperion.torch.narchs import AudioFeatsMVN as AF
from hyperion.torch.utils import open_device
from hyperion.utils import Utt2Info
from hyperion.utils.class_info import ClassInfo
from hyperion.utils.segment_set import SegmentSet
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
    logging.info("transducer-film-model={}".format(model))
    model.to(device)
    model.eval()
    return model


def decode_transducer(input_spec, lang_input_spec, output_spec, scp_sep, model_path, bpe_model, lang_file,
                      infer_args, use_gpu, **kwargs):

    device = init_device(use_gpu)
    model = load_model(model_path, device)

    # load language dict form langfile by row number
    lang_info = ClassInfo.load(lang_file)
    utt2lang = SegmentSet.load(lang_input_spec)


    logging.info("bpe-model=%s", bpe_model)
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model)

    infer_args = HFWav2Vec2RNNFiLMTransducer.filter_infer_args(**infer_args)
    logging.info(f"infer-args={infer_args}")

    ar_args = AR.filter_args(**kwargs)
    logging.info("opening output: %s", output_spec)
    with open(output_spec, "w") as writer:
        logging.info(f"opening input stream: {input_spec} with args={ar_args}")
        with AR(input_spec, **ar_args) as reader:
            while not reader.eof():
                t1 = time.time()
                key, x, fs = reader.read(1)
                lang = utt2lang.loc[key, "class_id"]
                lang_id = torch.tensor([lang_info.loc[lang, "class_idx"]]).to(torch.int64)
                if len(key) == 0:
                    break

                x, key, fs = x[0], key[0], fs[0]
                t2 = time.time()
                logging.info("processing utt %s", key)
                with torch.no_grad():
                    x = torch.tensor(
                        x[None, :], dtype=torch.get_default_dtype()).to(device)

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
                        #y = decode_one_batch(model=model, sp=sp, x=x)
                        x_lengths = torch.tensor((x.shape[1], ),
                                                 dtype=torch.long,
                                                 device=device)
                                                 
                        y = model.infer(x=x, x_lengths=x_lengths, languageid=lang_id, **infer_args)

                    y = sp.decode(y[0])
                    logging.info(f"utt: {key} hyps: {y}")
                    t3 = time.time()
                    writer.write(f"{key} {y}\n")

                    t4 = time.time()
                    tot_time = t4 - t1
                    infer_time = t3 - t2
                    logging.info(
                        ("utt %s total-time=%.3f read-time=%.3f "
                         "infer-time=%.3f "
                         "write-time=%.3f "
                         "infer-rt-factor=%.2f tot-rt-factor=%.2f"),
                        key,
                        tot_time,
                        t2 - t1,
                        infer_time,
                        t4 - t3,
                        x.shape[1] / fs / infer_time,
                        x.shape[1] / fs / tot_time,
                    )


if __name__ == "__main__":

    parser = ArgumentParser(
        description=("ASR decoding for RNN-T with Wav2vec features"))

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--input", dest="input_spec", required=True)
    parser.add_argument("--lang_input", dest="lang_input_spec", required=True)
    parser.add_argument("--scp-sep",
                        default=" ",
                        help=("scp file field separator"))

    AR.add_class_args(parser)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--bpe-model", required=True)
    parser.add_argument("--lang-file", required=True)

    HFWav2Vec2RNNFiLMTransducer.add_infer_args(parser, "infer-args")
    parser.add_argument("--output", dest="output_spec", required=True)
    parser.add_argument("--use-gpu",
                        default=False,
                        action="store_true",
                        help="extract xvectors in gpu")
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    decode_transducer(**namespace_to_dict(args))
