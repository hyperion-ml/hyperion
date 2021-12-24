#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import sys
import os
import argparse
import time
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from hyperion.hyp_defs import set_float_cpu, float_cpu, config_logger
from hyperion.torch.torch_defs import float_torch
from hyperion.torch.utils import open_device
from hyperion.torch.data import SeqDataset, ClassWeightedSeqSampler as Sampler
from hyperion.torch.helpers import TorchNALoader
from hyperion.torch.helpers import OptimizerFactory as OF
from hyperion.torch.lr_schedulers import LRSchedulerFactory as LRSF
from hyperion.torch.layers import GlobalPool1dFactory as PF
from hyperion.torch.seq_embed import XVector, XVectorTrainer
from hyperion.torch.metrics import CategoricalAccuracy


def train_xvector(
    data_path,
    train_list,
    val_list,
    encoder_net,
    classif_net,
    preproc_net,
    loss,
    exp_path,
    epochs,
    resume,
    resume_path,
    num_gpus,
    seed,
    **kwargs
):

    device = open_device(num_gpus=num_gpus)

    set_float_cpu(float_torch())
    torch.manual_seed(seed)

    opt_args = OF.filter_args(prefix="opt", **kwargs)
    lrsch_args = LRSF.filter_args(prefix="lrsch", **kwargs)
    pool_cfg = PF.filter_args(prefix="pool", **kwargs)
    dataset_args = SeqDataset.filter_args(prefix="data", **kwargs)
    sampler_args = Sampler.filter_args(prefix="data", **kwargs)

    train_data = SeqDataset(data_path, train_list, **dataset_args)
    train_sampler = Sampler(train_data, **sampler_args)
    val_data = SeqDataset(data_path, val_list, **dataset_args)
    val_sampler = Sampler(val_data, **sampler_args)
    train_loader = DataLoader(
        train_data, batch_sampler=train_sampler, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_data, batch_sampler=val_sampler, num_workers=num_workers
    )

    optimizer = OF.create(model.parameters(), **opt_args)
    lr_sch = LRSF.create(optimizer, **lrsch_args)
    loss = nn.CrossEntropyLoss()
    metrics = {"acc": CategoricalAccuracy()}

    if preproc_net is not None:
        preproc_net = TorchNALoader.load(preproc_net)
    encoder_net = TorchNALoader.load(encoder_net)
    classif_net = TorchNALoader.load(classif_net)

    model = XVector(encoder_net, pool_cfg, classif_net, preproc_net=preproc_net)
    trainer = XVectorTrainer(
        model,
        optimizer,
        loss,
        epochs,
        exp_path,
        device=device,
        metrics=metrics,
        lr_scheduler=lr_sch,
    )

    if resume:
        if resume_path is not None:
            trainer.load_checkpoint(resume_path)
        else:
            trainer.load_last_checkpoint()

    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Train x-vectors",
    )

    parser.add_argument("--data-path", dest="data_path", required=True)
    parser.add_argument("--train-list", dest="train_list", required=True)
    parser.add_argument("--val-list", dest="val_list", default=None)
    SeqDataset.add_argparse_args(parser, prefix="data")
    Sampler.add_argparse_args(parser, prefix="data")

    parser.add_argument("--encoder-net", dest="encoder_net", required=True)
    parser.add_argument("--classif-net", dest="classif_net", required=True)
    parser.add_argument("--preproc-net", dest="preproc_net", required=True)
    PF.add_argparse_args(parser, prefix="pool")

    OF.add_argparse_args(parser, prefix="opt")
    LRSF.add_argparse_args(parser, prefix="lrsch")

    parser.add_argument(
        "--num-gpus", action="num_gpus", default=1, help="number of gpus, if 0 use cpu"
    )
    parser.add_argument("--seed", type=int, default=1024, help="random seed")

    # parser.add_argument('--log-interval', type=int, default=10,
    #                     help='how many batches to wait before logging training status')

    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume training from checkpoint",
    )
    parser.add_argument(
        "--resume-path",
        default=None,
        help="checkpoint path, if none it uses the last checkpoint in exp_path",
    )

    parser.add_argument("--exp-path", help="experiment path")

    parser.add_argument("--epochs", dest="epochs", default=1000, type=int)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    train_xvectors(**vars(args))
