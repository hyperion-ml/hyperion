#!/usr/bin/env python
"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import sys
import os
from pathlib import Path
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
)
import time
import logging
import multiprocessing

import numpy as np

import torch
import torch.nn as nn

from hyperion.hyp_defs import config_logger, set_float_cpu
from hyperion.torch.utils import open_device
from hyperion.torch.utils import ddp
from hyperion.torch.trainers import XVectorTrainerFromWav as Trainer
from hyperion.torch.data import AudioDataset as AD
from hyperion.torch.data import ClassWeightedSeqSampler as Sampler
from hyperion.torch.metrics import CategoricalAccuracy
from hyperion.torch.narchs import AudioFeatsMVN as AF
from hyperion.torch.models import ResNetXVector as RXVec
from hyperion.torch.models import EfficientNetXVector as EXVec
from hyperion.torch.models import TDNNXVector as TDXVec
from hyperion.torch.models import TransformerXVectorV1 as TFXVec

xvec_dict = {
    "resnet": RXVec,
    "efficientnet": EXVec,
    "tdnn": TDXVec,
    "transformer": TFXVec,
}


def init_data(
    audio_path,
    train_list,
    val_list,
    train_aug_cfg,
    val_aug_cfg,
    num_workers,
    num_gpus,
    rank,
    **kwargs
):

    ad_args = AD.filter_args(**kwargs)
    sampler_args = Sampler.filter_args(**kwargs)
    if rank == 0:
        logging.info("audio dataset args={}".format(ad_args))
        logging.info("sampler args={}".format(sampler_args))
        logging.info("init datasets")

    train_data = AD(audio_path, train_list, aug_cfg=train_aug_cfg, **ad_args)
    val_data = AD(audio_path, val_list, aug_cfg=val_aug_cfg, is_val=True, **ad_args)

    if rank == 0:
        logging.info("init samplers")
    train_sampler = Sampler(train_data, **sampler_args)
    val_sampler = Sampler(val_data, **sampler_args)

    num_workers_per_gpu = int((num_workers + num_gpus - 1) / num_gpus)
    largs = (
        {"num_workers": num_workers_per_gpu, "pin_memory": True} if num_gpus > 0 else {}
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_sampler=train_sampler, **largs
    )

    test_loader = torch.utils.data.DataLoader(
        val_data, batch_sampler=val_sampler, **largs
    )

    return train_loader, test_loader


def init_feats(rank, **kwargs):
    feat_args = AF.filter_args(**kwargs["feats"])
    if rank == 0:
        logging.info("feat args={}".format(feat_args))
        logging.info("initializing feature extractor")
    feat_extractor = AF(trans=True, **feat_args)
    if rank == 0:
        logging.info("feat-extractor={}".format(feat_extractor))
    return feat_extractor


def init_xvector(num_classes, rank, xvec_class, **kwargs):

    xvec_args = xvec_class.filter_args(**kwargs)
    if rank == 0:
        logging.info("xvector network args={}".format(xvec_args))
    xvec_args["num_classes"] = num_classes
    model = xvec_class(**xvec_args)
    if rank == 0:
        logging.info("x-vector-model={}".format(model))
    return model


def train_xvec(gpu_id, args):

    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    kwargs = namespace_to_dict(args)
    torch.manual_seed(args.seed)
    set_float_cpu("float32")

    ddp_args = ddp.filter_ddp_args(**kwargs)
    device, rank, world_size = ddp.ddp_init(gpu_id, **ddp_args)
    kwargs["rank"] = rank

    train_loader, test_loader = init_data(**kwargs)
    feat_extractor = init_feats(**kwargs)
    model = init_xvector(train_loader.dataset.num_classes, **kwargs)

    trn_args = Trainer.filter_args(**kwargs)
    if rank == 0:
        logging.info("trainer args={}".format(trn_args))
    metrics = {"acc": CategoricalAccuracy()}
    trainer = Trainer(
        model,
        feat_extractor,
        device=device,
        metrics=metrics,
        ddp=world_size > 1,
        **trn_args
    )
    if args.resume:
        trainer.load_last_checkpoint()
    trainer.fit(train_loader, test_loader)

    ddp.ddp_cleanup()


def make_parser(xvec_class):
    parser = ArgumentParser()

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)

    AD.add_class_args(parser)
    Sampler.add_class_args(parser)

    parser.add_argument("--train-aug-cfg", default=None)
    parser.add_argument("--val-aug-cfg", default=None)

    parser.add_argument(
        "--num-workers", type=int, default=5, help="num_workers of data loader"
    )

    AF.add_class_args(parser, prefix="feats")
    xvec_class.add_class_args(parser)
    Trainer.add_class_args(parser)
    ddp.add_ddp_args(parser)
    parser.add_argument("--seed", type=int, default=1123581321, help="random seed")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume training from checkpoint",
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    return parser


if __name__ == "__main__":

    parser = ArgumentParser(description="Train XVector from audio files")

    parser.add_argument("--cfg", action=ActionConfigFile)

    subcommands = parser.add_subcommands()

    for k, v in xvec_dict.items():
        parser_k = make_parser(v)
        subcommands.add_subcommand(k, parser_k)

    args = parser.parse_args()
    try:
        gpu_id = int(os.environ["LOCAL_RANK"])
    except:
        gpu_id = 0

    xvec_type = args.subcommand
    args_sc = vars(args)[xvec_type]

    if gpu_id == 0:
        try:
            config_file = Path(args_sc.exp_path) / "config.yaml"
            parser.save(args, str(config_file), format="yaml", overwrite=True)
        except:
            pass

    args_sc.xvec_class = xvec_dict[xvec_type]
    # torch docs recommend using forkserver
    multiprocessing.set_start_method("forkserver")
    train_xvec(gpu_id, args_sc)
