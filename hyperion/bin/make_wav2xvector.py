#!/usr/bin/env python
"""
 Copyright 2023 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""

import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from hyperion.hyp_defs import config_logger

# from hyperion.torch import TorchModelLoader as TML
from hyperion.torch import TorchModel

# from hyperion.torch.models import SpineNetXVector as SpineXVec
# from hyperion.torch.models import TDNNXVector as TDXVec
# from hyperion.torch.models import TransformerXVectorV1 as TFXVec
# from hyperion.torch.models import EfficientNetXVector as EXVec
from hyperion.torch.models import ResNet1dXVector as R1dXVec
from hyperion.torch.models import ResNetXVector as RXVec
from hyperion.torch.models import Wav2ResNet1dXVector as W2R1dXVec
from hyperion.torch.models import Wav2ResNetXVector as W2RXVec
from hyperion.torch.narchs import AudioFeatsMVN as AF
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)


def init_feats(feats):
    feat_args = AF.filter_args(**feats)
    logging.info(f"feat args={feat_args}")
    logging.info("initializing feature extractor")
    feat_extractor = AF(trans=True, **feat_args)
    logging.info(f"feat-extractor={feat_extractor}")
    return feat_extractor


def load_model(model_path):
    logging.info("loading model %s", model_path)
    model = TorchModel.auto_load(model_path)
    logging.info(f"xvector-model={model}")
    return model


def make_wav2xvector(feats, xvector_path, output_path):

    feats = init_feats(feats)
    xvector_model = load_model(xvector_path)
    if isinstance(xvector_model, RXVec):
        model = W2RXVec(feats, xvector_model)
    elif isinstance(xvector_model, R1dXVec):
        model = W2R1dXVec(feats, xvector_model)
    else:
        TypeError(
            "Conversion of xvector class=%s not available", xvector_model.__class__
        )

    logging.info("saving model of class %s to %s", model.__class__, output_path)
    model.save(output_path)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Combines the feature extractor config with XVector model
        to produce a Wav2XVector model with integrated feature extraction"""
    )

    parser.add_argument("--cfg", action=ActionConfigFile)
    AF.add_class_args(parser, prefix="feats")
    parser.add_argument("--xvector-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    del args.cfg
    logging.debug(args)

    make_wav2xvector(**namespace_to_dict(args))
