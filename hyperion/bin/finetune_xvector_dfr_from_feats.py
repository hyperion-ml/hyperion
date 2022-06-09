#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
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
from hyperion.torch.utils import open_device, ddp
from hyperion.torch.models import XVector as XVec
from hyperion.torch.trainers import XVectorTrainerDeepFeatReg as Trainer
from hyperion.torch.data import FeatSeqDataset as SD
from hyperion.torch.data import ClassWeightedSeqSampler as Sampler
from hyperion.torch.metrics import CategoricalAccuracy
from hyperion.torch import TorchModelLoader as TML


def init_data(data_rspec, train_list, val_list, num_workers, num_gpus, rank, **kwargs):
    sd_args = SD.filter_args(**kwargs)
    sampler_args = Sampler.filter_args(**kwargs)
    if rank == 0:
        logging.info("audio dataset args={}".format(sd_args))
        logging.info("sampler args={}".format(sampler_args))
        logging.info("init datasets")

    train_data = SD(data_rspec, train_list, **sd_args)
    val_data = SD(data_rspec, val_list, is_val=True, **sd_args)
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


def init_xvector(
    num_classes, in_model_path, prior_model_path, rank, train_mode, **kwargs
):

    xvec_args = XVec.filter_finetune_args(**kwargs)
    if rank == 0:
        logging.info("xvector network ft args={}".format(xvec_args))
    xvec_args["num_classes"] = num_classes
    model = TML.load(in_model_path)
    model.rebuild_output_layer(**xvec_args)
    if prior_model_path:
        prior_model = TML.load(prior_model_path)
    else:
        prior_model = model.copy()
    prior_model.freeze()
    prior_model.eval()
    if train_mode == "ft-embed-affine":
        model.freeze_preembed_layers()
    if rank == 0:
        logging.info("x-vector-model={}".format(model))
    return model, prior_model


def train_xvec(gpu_id, args):
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    kwargs = namespace_to_dict(args)
    torch.manual_seed(args.seed)
    set_float_cpu("float32")

    train_mode = kwargs["train_mode"]

    ddp_args = ddp.filter_ddp_args(**kwargs)
    device, rank, world_size = ddp.ddp_init(gpu_id, **ddp_args)
    kwargs["rank"] = rank
    train_loader, test_loader = init_data(**kwargs)
    model, prior_model = init_xvector(train_loader.dataset.num_classes, **kwargs)

    trn_args = Trainer.filter_args(**kwargs)
    if rank == 0:
        logging.info("trainer args={}".format(trn_args))
    metrics = {"acc": CategoricalAccuracy()}

    trainer = Trainer(
        model,
        prior_model,
        device=device,
        metrics=metrics,
        ddp=world_size > 1,
        train_mode=train_mode,
        **trn_args
    )
    if args.resume:
        trainer.load_last_checkpoint()
    trainer.fit(train_loader, test_loader)

    ddp.ddp_cleanup()


# def train_xvec(data_rspec, train_list, val_list, in_model_path,
#                prior_model_path,
#                reg_layers_enc, reg_layers_classif,
#                reg_weight_enc, reg_weight_classif, reg_loss,
#                num_gpus, resume, num_workers,
#                train_mode, **kwargs):

#     set_float_cpu('float32')
#     logging.info('initializing devices num_gpus={}'.format(num_gpus))
#     device = open_device(num_gpus=num_gpus)

#     sd_args = SD.filter_args(**kwargs)
#     sampler_args = Sampler.filter_args(**kwargs)
#     xvec_args = XVec.filter_finetune_args(**kwargs)
#     opt_args = OF.filter_args(prefix='opt', **kwargs)
#     lrsch_args = LRSF.filter_args(prefix='lrsch', **kwargs)
#     trn_args = Trainer.filter_args(**kwargs)
#     logging.info('seq dataset args={}'.format(sd_args))
#     logging.info('sampler args={}'.format(sampler_args))
#     logging.info('xvector finetune args={}'.format(xvec_args))
#     logging.info('optimizer args={}'.format(opt_args))
#     logging.info('lr scheduler args={}'.format(lrsch_args))
#     logging.info('trainer args={}'.format(trn_args))

#     logging.info('init datasets')
#     train_data = SD(data_rspec, train_list, **sd_args)
#     val_data = SD(data_rspec, val_list, is_val=True, **sd_args)

#     logging.info('init samplers')
#     train_sampler = Sampler(train_data, **sampler_args)
#     val_sampler = Sampler(val_data, **sampler_args)

#     largs = {'num_workers': num_workers, 'pin_memory': True} if num_gpus>0 else {}

#     train_loader = torch.utils.data.DataLoader(
#         train_data, batch_sampler = train_sampler, **largs)

#     test_loader = torch.utils.data.DataLoader(
#         val_data, batch_sampler = val_sampler, **largs)

#     xvec_args['num_classes'] = train_data.num_classes
#     model = TML.load(in_model_path)
#     model.rebuild_output_layer(**xvec_args)
#     if prior_model_path:
#         prior_model = TML.load(prior_model_path)
#     else:
#         prior_model = model.copy()
#     prior_model.freeze()
#     prior_model.eval()
#     if train_mode == 'ft-embed-affine':
#         model.freeze_preembed_layers()
#     logging.info(str(model))

#     optimizer = OF.create(model.parameters(), **opt_args)
#     lr_sch = LRSF.create(optimizer, **lrsch_args)
#     metrics = { 'acc': CategoricalAccuracy() }

#     if reg_loss == 'l1':
#         reg_loss = nn.L1Loss()
#     else:
#         reg_loss = nn.MSELoss()

#     trainer = Trainer(model, prior_model, optimizer,
#                       reg_layers_enc=reg_layers_enc,
#                       reg_layers_classif=reg_layers_classif,
#                       reg_weight_enc=reg_weight_enc,
#                       reg_weight_classif=reg_weight_classif,
#                       reg_loss=reg_loss,
#                       device=device, metrics=metrics, lr_scheduler=lr_sch,
#                       data_parallel=(num_gpus>1), train_mode=train_mode,
#                       **trn_args)
#     if resume:
#         trainer.load_last_checkpoint()
#     trainer.fit(train_loader, test_loader)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Fine-tune x-vector model with deep feature loss regularization"
    )

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--data-rspec", dest="data_rspec", required=True)
    parser.add_argument("--train-list", dest="train_list", required=True)
    parser.add_argument("--val-list", dest="val_list", required=True)

    SD.add_argparse_args(parser)
    Sampler.add_argparse_args(parser)
    parser.add_argument(
        "--num-workers", type=int, default=5, help="num_workers of data loader"
    )

    # parser.add_argument('--reg-layers-enc', type=int, default=None, nargs='+',
    #                     help='list of layers from the encoder nnet to use for regularization ')
    # parser.add_argument('--reg-layers-classif', type=int, default=None, nargs='+',
    #                     help='list of layers from the classif nnet to use for regularization ')
    # parser.add_argument('--reg-weight-enc', type=float, default=0.1,
    #                     help='weight for regularization from enc layers')
    # parser.add_argument('--reg-weight-classif', type=float, default=0.1,
    #                     help='weight for regularization from classif layers')
    # parser.add_argument('--reg-loss', default='l1',
    #                     choices=['l1', 'mse'],
    #                     help=('type of regularization loss'))

    parser.add_argument("--in-model-path", required=True)
    parser.add_argument("--prior-model-path")
    XVec.add_finetune_args(parser)
    Trainer.add_class_args(parser)
    ddp.add_ddp_args(parser)

    # parser.add_argument('--num-gpus', type=int, default=1,
    #                     help='number of gpus, if 0 it uses cpu')
    parser.add_argument(
        "--seed", type=int, default=1123581321, help="random seed (default: 1)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume training from checkpoint",
    )
    parser.add_argument(
        "--train-mode",
        default="ft-embed-affine",
        choices=["ft-full", "ft-embed-affine"],
        help=(
            "ft-full: adapt full x-vector network"
            "ft-embed-affine: adapt affine transform before embedding"
        ),
    )

    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()
    gpu_id = args.local_rank
    del args.local_rank

    if gpu_id == 0:
        try:
            config_file = Path(args.exp_path) / "config.yaml"
            parser.save(args, str(config_file), format="yaml", overwrite=True)
        except:
            pass

    # torch docs recommend using forkserver
    multiprocessing.set_start_method("forkserver")
    train_xvec(gpu_id, args)

    # config_logger(args.verbose)
    # del args.verbose
    # logging.debug(args)

    # torch.manual_seed(args.seed)
    # del args.seed

    # train_xvec(**vars(args))
