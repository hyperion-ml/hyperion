#!/usr/bin/env python
"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger, set_float_cpu
from hyperion.torch import HyperTorchModel
from hyperion.torch.data import ClassWeightedRandomSegChunkSampler as Sampler
from hyperion.torch.data import FeatSeqDataset as SD
from hyperion.torch.metrics import CategoricalAccuracy
from hyperion.torch.models import XVector as XVec
from hyperion.torch.trainers import XVectorTrainer as Trainer
from hyperion.torch.utils import ddp, open_device
from hyperion.utils.misc import PathLike


def init_data(
    data_rspec: PathLike,
    train_list: PathLike,
    val_list: PathLike,
    num_workers: int,
    num_gpus: int,
    rank: int,
    **kwargs: Any,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Initialize training and validation dataloaders.

    Args:
        data_rspec: Input feature specifier/archive.
        train_list: Training utterance/class list path.
        val_list: Validation utterance/class list path.
        num_workers: Total number of dataloader worker processes.
        num_gpus: Number of GPUs used by the process group.
        rank: Distributed rank of the current process.
        **kwargs: Additional parsed args for dataset and sampler configuration.
    """
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
    num_classes: int,
    in_model_path: PathLike,
    rank: int,
    train_mode: str,
    **kwargs: Any,
) -> torch.nn.Module:
    """Load and reconfigure x-vector model checkpoint for fine-tuning.

    Args:
        num_classes: Number of target classes in current training data.
        in_model_path: Input model checkpoint path used as fine-tuning start.
        rank: Distributed rank of the current process.
        train_mode: Fine-tuning mode.
        **kwargs: Additional parsed args containing model configuration.
    """
    xvec_args = XVec.filter_finetune_args(**kwargs)
    if rank == 0:
        logging.info("xvector network ft args={}".format(xvec_args))
    xvec_args["num_classes"] = num_classes
    model = HyperTorchModel.auto_load(in_model_path)
    model.rebuild_output_layer(**xvec_args)
    if train_mode == "ft-embed-affine":
        model.freeze_preembed_layers()
    if rank == 0:
        logging.info("x-vector-model={}".format(model))
    return model


def train_xvec(gpu_id: int, args: Any) -> None:
    """Run distributed x-vector fine-tuning from features.

    Args:
        gpu_id: Local GPU index used by this process.
        args: Parsed command-line namespace.
    """
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    kwargs = namespace_to_dict(args)
    torch.manual_seed(args.seed)
    set_float_cpu("float32")

    train_mode = kwargs["train_mode"]

    ddp_args = ddp.filter_ddp_args(**kwargs)
    device, rank, world_size = ddp.ddp_init(**ddp_args)
    kwargs["rank"] = rank
    train_loader, test_loader = init_data(**kwargs)
    model = init_xvector(train_loader.dataset.num_classes, **kwargs)

    trn_args = Trainer.filter_args(**kwargs)
    if rank == 0:
        logging.info("trainer args={}".format(trn_args))
    metrics = {"acc": CategoricalAccuracy()}
    trainer = Trainer(
        model,
        device=device,
        metrics=metrics,
        ddp=world_size > 1,
        train_mode=train_mode,
        **trn_args,
    )
    if args.resume:
        trainer.load_last_checkpoint()
    trainer.fit(train_loader, test_loader)

    ddp.ddp_cleanup()


# (data_rspec, train_list, val_list, in_model_path,
#                num_gpus, resume, num_workers, train_mode, **kwargs):

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
#     model = HyperTorchModel.auto_load(in_model_path)
#     model.rebuild_output_layer(**xvec_args)
#     if train_mode == 'ft-embed-affine':
#         model.freeze_preembed_layers()
#     logging.info(str(model))

#     optimizer = OF.create(model.parameters(), **opt_args)
#     lr_sch = LRSF.create(optimizer, **lrsch_args)
#     metrics = { 'acc': CategoricalAccuracy() }

#     trainer = Trainer(model, optimizer,
#                       device=device, metrics=metrics, lr_scheduler=lr_sch,
#                       data_parallel=(num_gpus>1), train_mode=train_mode,
#                       **trn_args)
#     if resume:
#         trainer.load_last_checkpoint()
#     trainer.fit(train_loader, test_loader)


def main() -> None:
    """Parse CLI arguments and start x-vector fine-tuning from features.

    Args:
        None.
    """
    parser = ArgumentParser(description="Fine-tune x-vector model")

    parser.add_argument("--cfg", action=ActionConfigFile, help="configuration file")
    parser.add_argument(
        "--data-rspec", required=True, help="input feature archive/specifier"
    )
    parser.add_argument(
        "--train-list", required=True, help="training utterance/class list"
    )
    parser.add_argument(
        "--val-list", required=True, help="validation utterance/class list"
    )

    SD.add_argparse_args(parser)
    Sampler.add_argparse_args(parser)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=5,
        help="number of dataloader worker processes",
    )
    parser.add_argument(
        "--in-model-path",
        required=True,
        help="input model checkpoint path used as fine-tuning starting point",
    )
    XVec.add_finetune_args(parser)
    Trainer.add_class_args(parser)
    ddp.add_ddp_args(parser)

    # parser.add_argument('--num-gpus', type=int, default=1,
    #                     help='number of gpus, if 0 it uses cpu')
    parser.add_argument("--seed", type=int, default=1123581321, help="random seed")
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
            "ft-full: adapt full x-vector network; "
            "ft-embed-affine: adapt affine transform before embedding"
        ),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="verbosity level (0=warning, 1=info, 2=debug, 3=trace)",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="local rank assigned by distributed launcher",
    )

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

    # args = parser.parse_args()
    # config_logger(args.verbose)
    # del args.verbose
    # logging.debug(args)

    # torch.manual_seed(args.seed)
    # del args.seed

    # train_xvec(**vars(args))


if __name__ == "__main__":
    main()
