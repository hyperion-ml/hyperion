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
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from hyperion.hyp_defs import config_logger
from hyperion.torch.utils import open_device
from hyperion.torch.narchs import ResNetFactory as RNF

from hyperion.torch.transforms import Reshape
from hyperion.torch.helpers import OptimizerFactory as OF
from hyperion.torch.lr_schedulers import LRSchedulerFactory as LRSF
from hyperion.torch.trainers import TorchTrainer
from hyperion.torch.metrics import CategoricalAccuracy


def main(
    batch_size,
    test_batch_size,
    exp_path,
    epochs,
    num_gpus,
    log_interval,
    resume,
    cifar_vers,
    **kwargs
):

    logging.info("initializing devices num_gpus={}".format(num_gpus))
    device = open_device(num_gpus=num_gpus)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    largs = {"num_workers": 2, "pin_memory": True} if num_gpus > 0 else {}

    if cifar_vers == 10:
        Dataset = datasets.CIFAR10
    else:
        Dataset = datasets.CIFAR100

    trainset = Dataset(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, **largs
    )

    testset = Dataset(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, **largs
    )

    model_args = RNF.filter_args(**kwargs)
    model_args["in_channels"] = 3
    model_args["out_units"] = cifar_vers
    logging.info("model-args={}".format(model_args))
    model = RNF.create(**model_args)
    logging.info("model={}".format(model))

    # classes cifar-10
    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #        'dog', 'frog', 'horse', 'ship', 'truck')

    opt_args = OF.filter_args(prefix="opt", **kwargs)
    logging.info("optim-args={}".format(opt_args))
    lrsch_args = LRSF.filter_args(prefix="lrsch", **kwargs)
    logging.info("lr-sched-args={}".format(lrsch_args))

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
        description="PyTorch CIFAR",
    )

    parser.add_argument(
        "--batch-size", type=int, default=128, help="input batch size for training"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=100, help="input batch size for testing"
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="number of epochs to train"
    )

    RNF.add_argparse_args(parser)
    OF.add_argparse_args(parser, prefix="opt")
    LRSF.add_argparse_args(parser, prefix="lrsch")

    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus, if 0 it uses cpu"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
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
        "--cifar-vers", default=10, type=int, choices=[10, 100], help="CIFAR version"
    )
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
