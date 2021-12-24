#!/usr/bin/env python
"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import sys
import os
import argparse
import time
import logging

import numpy as np

import torch
import torch.nn as nn

from hyperion.hyp_defs import config_logger, set_float_cpu
from hyperion.torch.utils import open_device
from hyperion.torch.helpers import OptimizerFactory as OF
from hyperion.torch.lr_schedulers import LRSchedulerFactory as LRSF
from hyperion.torch.narchs.dc1d_encoder import DC1dEncoder as Encoder
from hyperion.torch.narchs.dc1d_decoder import DC1dDecoder as Decoder
from hyperion.torch.models import AE
from hyperion.torch.trainers import AETrainer as Trainer
from hyperion.torch.data import SeqDataset as SD
from hyperion.torch.data import ClassWeightedSeqSampler as Sampler


def train_ae(
    data_rspec,
    train_list,
    val_list,
    exp_path,
    in_feats,
    latent_dim,
    loss,
    epochs,
    num_gpus,
    log_interval,
    resume,
    num_workers,
    grad_acc_steps,
    use_amp,
    **kwargs
):

    set_float_cpu("float32")
    logging.info("initializing devices num_gpus={}".format(num_gpus))
    device = open_device(num_gpus=num_gpus)

    sd_args = SD.filter_args(**kwargs)
    sampler_args = Sampler.filter_args(**kwargs)
    enc_args = Encoder.filter_args(prefix="enc", **kwargs)
    dec_args = Decoder.filter_args(prefix="dec", **kwargs)
    opt_args = OF.filter_args(prefix="opt", **kwargs)
    lrsch_args = LRSF.filter_args(prefix="lrsch", **kwargs)
    logging.info("seq dataset args={}".format(sd_args))
    logging.info("sampler args={}".format(sampler_args))
    logging.info("encoder args={}".format(enc_args))
    logging.info("decoder args={}".format(dec_args))
    logging.info("optimizer args={}".format(opt_args))
    logging.info("lr scheduler args={}".format(lrsch_args))

    logging.info("init datasets")
    train_data = SD(data_rspec, train_list, **sd_args)
    val_data = SD(data_rspec, val_list, is_val=True, **sd_args)

    logging.info("init samplers")
    train_sampler = Sampler(train_data, **sampler_args)
    val_sampler = Sampler(val_data, **sampler_args)

    largs = {"num_workers": num_workers, "pin_memory": True} if num_gpus > 0 else {}

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_sampler=train_sampler, **largs
    )

    test_loader = torch.utils.data.DataLoader(
        val_data, batch_sampler=val_sampler, **largs
    )

    encoder = Encoder(in_feats, head_channels=latent_dim, **enc_args)
    decoder = Decoder(latent_dim, head_channels=in_feats, **dec_args)
    model = AE(encoder, decoder)
    logging.info(str(model))

    optimizer = OF.create(model.parameters(), **opt_args)
    lr_sch = LRSF.create(optimizer, **lrsch_args)
    losses = {"mse": nn.MSELoss, "l1": nn.L1Loss, "smooth-l1": nn.SmoothL1Loss}
    metrics = {"mse": nn.MSELoss(), "L1": nn.L1Loss()}
    loss = losses[loss]()
    trainer = Trainer(
        model,
        optimizer,
        loss,
        epochs,
        exp_path,
        grad_acc_steps=grad_acc_steps,
        device=device,
        metrics=metrics,
        lr_scheduler=lr_sch,
        data_parallel=(num_gpus > 1),
        use_amp=use_amp,
    )
    if resume:
        trainer.load_last_checkpoint()
    trainer.fit(train_loader, test_loader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Train AE with Deep Conv1d Encoder-Decoder",
    )

    parser.add_argument("--data-rspec", dest="data_rspec", required=True)
    parser.add_argument("--train-list", dest="train_list", required=True)
    parser.add_argument("--val-list", dest="val_list", required=True)

    SD.add_argparse_args(parser)
    Sampler.add_argparse_args(parser)

    parser.add_argument(
        "--num-workers", type=int, default=5, help="num_workers of data loader"
    )

    parser.add_argument(
        "--grad-acc-steps",
        type=int,
        default=1,
        help="gradient accumulation batches before weigth update",
    )

    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")

    parser.add_argument(
        "--in-feats", type=int, required=True, help="input features dimension"
    )
    parser.add_argument(
        "--latent-dim", type=int, required=True, help="latent representation dimension"
    )

    Encoder.add_argparse_args(parser, prefix="enc")
    Decoder.add_argparse_args(parser, prefix="dec")

    OF.add_argparse_args(parser, prefix="opt")
    LRSF.add_argparse_args(parser, prefix="lrsch")

    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus, if 0 it uses cpu"
    )
    parser.add_argument("--seed", type=int, default=1123581321, help="random seed")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume training from checkpoint",
    )

    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=False,
        help="use mixed precision training",
    )

    parser.add_argument("--exp-path", help="experiment path")

    parser.add_argument("--loss", default="mse", choices=["mse", "l1", "smooth-l1"])
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    torch.manual_seed(args.seed)
    del args.seed

    train_ae(**vars(args))
