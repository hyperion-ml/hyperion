#!/usr/bin/env python
"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
import multiprocessing
import os
from pathlib import Path

import numpy
import torch
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger, set_float_cpu
from hyperion.torch.data import AudioDataset as AD
from hyperion.torch.data import SegSamplerFactory
from hyperion.torch.metrics import CategoricalAccuracy
from hyperion.torch.models import Wav2ConformerV1XVector as CXVec
from hyperion.torch.models import Wav2ConvNext1dXVector as ConvNext1dXVec
from hyperion.torch.models import Wav2ConvNext2dXVector as ConvNext2dXVec
from hyperion.torch.models import Wav2ResNet1dXVector as R1dXVec
from hyperion.torch.models import Wav2ResNetXVector as RXVec
from hyperion.torch.trainers import XVectorTrainer as Trainer
from hyperion.torch.utils import ddp

xvec_dict = {
    "resnet": RXVec,
    "resnet1d": R1dXVec,
    "conformer": CXVec,
    "convnext1d": ConvNext1dXVec,
    "convnext2d": ConvNext2dXVec,
    # "efficientnet": EXVec,
    # "tdnn": TDXVec,
    # "transformer": TFXVec,
    # "spinenet": SpineXVec,
}


def init_data(partition, rank, num_gpus, **kwargs):
    kwargs = kwargs["data"][partition]
    ad_args = AD.filter_args(**kwargs["dataset"])
    sampler_args = kwargs["sampler"]
    if rank == 0:
        logging.info("{} audio dataset args={}".format(partition, ad_args))
        logging.info("{} sampler args={}".format(partition, sampler_args))
        logging.info("init %s dataset", partition)

    is_val = partition == "val"
    ad_args["is_val"] = is_val
    sampler_args["shuffle"] = not is_val
    dataset = AD(**ad_args)

    if rank == 0:
        logging.info("init %s samplers", partition)

    sampler = SegSamplerFactory.create(dataset, **sampler_args)

    if rank == 0:
        logging.info("init %s dataloader", partition)

    num_workers = kwargs["data_loader"]["num_workers"]
    num_workers_per_gpu = int((num_workers + num_gpus - 1) / num_gpus)
    largs = (
        {"num_workers": num_workers_per_gpu, "pin_memory": True} if num_gpus > 0 else {}
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=dataset.get_collator(), **largs
    )
    return data_loader


def init_xvector(num_classes, rank, xvec_class, **kwargs):
    xvec_args = xvec_class.filter_args(**kwargs["model"])
    if rank == 0:
        logging.info("xvector network args={}".format(xvec_args))
    xvec_args["xvector"]["num_classes"] = num_classes
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

    train_loader = init_data(partition="train", **kwargs)
    val_loader = init_data(partition="val", **kwargs)

    model = init_xvector(list(train_loader.dataset.num_classes.values())[0], **kwargs)

    trn_args = Trainer.filter_args(**kwargs["trainer"])
    if rank == 0:
        logging.info("trainer args={}".format(trn_args))
    metrics = {"acc": CategoricalAccuracy()}
    trainer = Trainer(
        model,
        device=device,
        metrics=metrics,
        ddp=world_size > 1,
        **trn_args,
    )
    trainer.load_last_checkpoint()
    trainer.fit(train_loader, val_loader)

    ddp.ddp_cleanup()


def make_parser(xvec_class):
    parser = ArgumentParser()

    parser.add_argument("--cfg", action=ActionConfigFile)

    train_parser = ArgumentParser(prog="")

    AD.add_class_args(train_parser, prefix="dataset")
    SegSamplerFactory.add_class_args(train_parser, prefix="sampler")
    train_parser.add_argument(
        "--data_loader.num-workers",
        type=int,
        default=5,
        help="num_workers of data loader",
    )

    val_parser = ArgumentParser(prog="")
    AD.add_class_args(val_parser, prefix="dataset")
    SegSamplerFactory.add_class_args(val_parser, prefix="sampler")
    val_parser.add_argument(
        "--data_loader.num-workers",
        type=int,
        default=5,
        help="num_workers of data loader",
    )
    data_parser = ArgumentParser(prog="")
    data_parser.add_argument("--train", action=ActionParser(parser=train_parser))
    data_parser.add_argument("--val", action=ActionParser(parser=val_parser))
    parser.add_argument("--data", action=ActionParser(parser=data_parser))
    parser.link_arguments(
        "data.train.dataset.class_files", "data.val.dataset.class_files"
    )
    parser.link_arguments(
        "data.train.data_loader.num_workers", "data.val.data_loader.num_workers"
    )

    xvec_class.add_class_args(parser, prefix="model")
    Trainer.add_class_args(
        parser, prefix="trainer", train_modes=xvec_class.valid_train_modes()
    )
    ddp.add_ddp_args(parser)
    parser.add_argument("--seed", type=int, default=1123581321, help="random seed")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    return parser


def main():
    parser = ArgumentParser(description="Train Wav2XVector from audio files")
    parser.add_argument("--cfg", action=ActionConfigFile)

    subcommands = parser.add_subcommands()
    for k, v in xvec_dict.items():
        parser_k = make_parser(v)
        subcommands.add_subcommand(k, parser_k)

    # os.environ["MKL_THREADING_LAYER"] = "GNU"
    # os MKL_SERVICE_FORCE_INTEL=1
    args = parser.parse_args()
    try:
        gpu_id = int(os.environ["LOCAL_RANK"])
    except:
        gpu_id = 0

    xvec_type = args.subcommand
    args_sc = vars(args)[xvec_type]

    if gpu_id == 0:
        try:
            config_file = Path(args_sc.trainer.exp_path) / "config.yaml"
            parser.save(args, str(config_file), format="yaml", overwrite=True)
        except:
            logging.warning(f"failed saving {args} to {config_file}")

    args_sc.xvec_class = xvec_dict[xvec_type]
    # torch docs recommend using forkserver
    multiprocessing.set_start_method("forkserver")
    train_xvec(gpu_id, args_sc)


if __name__ == "__main__":
    main()
