#!/usr/bin/env python
"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
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
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from hyperion.hyp_defs import config_logger
from hyperion.torch.utils import open_device
from hyperion.torch.narchs import (
    FCNetV1,
    TDNNV1,
    ETDNNV1,
    ResETDNNV1,
    LResNet18,
    LResNet50,
    LResNext50_4x4d,
)
from hyperion.torch.transforms import Reshape
from hyperion.torch.helpers import OptimizerFactory as OF
from hyperion.torch.lr_schedulers import LRSchedulerFactory as LRSF
from hyperion.torch.trainers import TorchTrainer
from hyperion.torch.metrics import CategoricalAccuracy

input_width = 28
input_height = 28


def create_net(net_type):
    if net_type == "fcnet":
        return FCNetV1(2, input_width * input_height, 1000, 10, dropout_rate=0.5)
    if net_type == "tdnn":
        return TDNNV1(2, input_height, 1000, 10, dropout_rate=0.5, pooling="mean")
    if net_type == "etdnn":
        return ETDNNV1(2, input_height, 1000, 10, dropout_rate=0.5, pooling="mean")
    if net_type == "resetdnn":
        return ResETDNNV1(
            3, input_height, 1000, 1000, 10, dropout_rate=0.5, pooling="mean"
        )
    if net_type == "lresnet18":
        return LResNet18(1, out_units=10, dropout_rate=0.5)
    if net_type == "lresnet50":
        return LResNet50(1, out_units=10, dropout_rate=0.5)
    if net_type == "lresnext50":
        return LResNext50_4x4d(1, out_units=10, dropout_rate=0.5)


def main(
    net_type,
    batch_size,
    test_batch_size,
    exp_path,
    epochs,
    num_gpus,
    log_interval,
    resume,
    **kwargs
):

    opt_args = OF.filter_args(prefix="opt", **kwargs)
    lrsch_args = LRSF.filter_args(prefix="lrsch", **kwargs)

    device = open_device(num_gpus=num_gpus)

    transform_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    if net_type == "fcnet":
        transform_list.append(Reshape((-1,)))
    elif net_type == "tdnn" or net_type == "etdnn" or net_type == "resetdnn":
        transform_list.append(Reshape((input_height, input_width)))
    transform = transforms.Compose(transform_list)

    largs = {"num_workers": 1, "pin_memory": True} if num_gpus > 0 else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./exp/data", train=True, download=True, transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
        **largs
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./exp/data", train=False, transform=transform),
        batch_size=args.test_batch_size,
        shuffle=False,
        **largs
    )

    model = create_net(net_type)
    # model.to(device)

    print(opt_args)
    print(lrsch_args)
    optimizer = OF.create(model.parameters(), **opt_args)
    lr_sch = LRSF.create(optimizer, **lrsch_args)
    loss = nn.CrossEntropyLoss()
    metrics = {"acc": CategoricalAccuracy()}

    trainer = TorchTrainer(
        model,
        optimizer,
        loss,
        epochs,
        exp_path,
        device=device,
        metrics=metrics,
        lr_scheduler=lr_sch,
        data_parallel=(num_gpus > 1),
    )
    if resume:
        trainer.load_last_checkpoint()
    trainer.fit(train_loader, test_loader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="PyTorch MNIST",
    )
    parser.add_argument(
        "--net-type", default="fcnet", metavar="N", help="Type of network architecture"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    OF.add_argparse_args(parser, prefix="opt")
    LRSF.add_argparse_args(parser, prefix="lrsch")
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus, if 0 it uses cpu"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
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

    parser.add_argument("--exp-path", help="experiment path")

    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    torch.manual_seed(args.seed)
    del args.seed

    main(**vars(args))
