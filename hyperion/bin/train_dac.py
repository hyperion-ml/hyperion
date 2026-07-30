#!/usr/bin/env python
"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import multiprocessing

# import sys
import os
import time
from pathlib import Path
from typing import Any, Dict, Type

import numpy as np
import torch
import torch.nn as nn
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger, set_float_cpu
from hyperion.torch import HyperTorchModel
from hyperion.torch.data import AudioDataset as AD
from hyperion.torch.data import SegSamplerFactory
from hyperion.torch.models.audio_discrimitator import AudioMultiDiscriminator
from hyperion.torch.models.dac import DAC, StreamingDAC
from hyperion.torch.narchs import AudioFeatsMVN
from hyperion.torch.trainers.dac_trainer import DACTrainer as Trainer
from hyperion.torch.utils import ddp

model_dict = {
    "dac": DAC,
    "streaming_dac": StreamingDAC,
}


def init_data(
    partition: str, rank: int, num_gpus: int, **kwargs: Any
) -> torch.utils.data.DataLoader:
    """Initialize dataset, sampler, and dataloader for one partition.

    Args:
        partition: Dataset split name (``"train"`` or ``"val"``).
        rank: Process rank in distributed training.
        num_gpus: Number of GPUs available to this job.
        **kwargs: Parsed configuration dictionary containing data settings.
    """
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

    num_workers_per_gpu = kwargs["data_loader"]["num_workers"]
    largs = (
        {
            "num_workers": num_workers_per_gpu,
            "pin_memory": True,
            "persistent_workers": True,
        }
        if num_gpus > 0
        else {}
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=dataset.get_collator(), **largs
    )
    return data_loader


def init_dac_model(
    rank: int, model_class: Type[HyperTorchModel], model_args: Dict[str, Any]
) -> HyperTorchModel:
    """Initialize DAC model from configuration.

    Args:
        rank: Process rank in distributed training.
        model_class: DAC model class to instantiate.
        model_args: DAC model initialization arguments.
    """
    if rank == 0:
        logging.info(f"dac_model network args={model_args}")

    model = model_class(**model_args)
    if rank == 0:
        logging.info(f"dac_model={model}")
        logging.info(f"dac_model frame_shift={model.frame_shift} samples")
        logging.info(f"dac_model frame_length={model.frame_length} samples")
        logging.info(
            f"dac_model frame_shift={model.frame_shift / model.input_sample_frequency} seconds"
        )
        logging.info(
            f"dac_model frame_length={model.frame_length / model.input_sample_frequency} seconds"
        )
        logging.info(f"dac_model in_context={model.in_context()}")
        logging.info(f"dac model delay={model.delay}")
        logging.info(
            f"dac_model encoder_frame_length={model.encoder_frame_length} samples"
        )
        logging.info(
            f"dac_model encoder_frame_length={model.encoder_frame_length / model.input_sample_frequency} seconds"
        )
        logging.info(f"dac_model encoder_in_context={model.encoder_in_context()}")

    return model


def init_discrim_model(
    rank: int, model_args: Dict[str, Any]
) -> AudioMultiDiscriminator:
    """Initialize audio discriminator model.

    Args:
        rank: Process rank in distributed training.
        model_args: Discriminator model initialization arguments.
    """
    if rank == 0:
        logging.info("discrim_model network args={}".format(model_args))

    model = AudioMultiDiscriminator(**model_args)
    if rank == 0:
        logging.info("discrim_model={}".format(model))
    return model


def train_model(gpu_id: int, args: Any) -> None:
    """Run distributed training for a DAC model.

    Args:
        gpu_id: Local GPU id used by this process.
        args: Parsed subcommand arguments.
    """
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    kwargs = namespace_to_dict(args)
    torch.manual_seed(args.seed)
    set_float_cpu("float32")

    ddp_args = ddp.filter_ddp_args(**kwargs)
    device, rank, world_size = ddp.ddp_init(**ddp_args)
    kwargs["rank"] = rank

    train_loader = init_data(partition="train", **kwargs)
    val_loader = init_data(partition="val", **kwargs)
    dac_model = init_dac_model(rank, kwargs["model_class"], kwargs["dac_model"])
    discrim_model = init_discrim_model(rank, kwargs["discrim_model"])

    trn_args = Trainer.filter_args(**kwargs["trainer"])
    if rank == 0:
        logging.info(f"trainer args={trn_args}")

    trainer = Trainer(
        dac_model=dac_model,
        discrim_model=discrim_model,
        device=device,
        ddp=world_size > 1,
        **trn_args,
    )
    trainer.load_last_checkpoint()
    trainer.fit(train_loader, val_loader)

    ddp.ddp_cleanup()


def make_parser(model_class: Type[HyperTorchModel]) -> ArgumentParser:
    """Create parser for one DAC model subcommand.

    Args:
        model_class: DAC model class whose args should be exposed.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--cfg",
        action=ActionConfigFile,
        help="Path to a configuration file.",
    )

    train_parser = ArgumentParser(prog="")
    AD.add_class_args(train_parser, prefix="dataset")
    SegSamplerFactory.add_class_args(train_parser, prefix="sampler")
    train_parser.add_argument(
        "--data_loader.num-workers",
        type=int,
        default=5,
        help="Number of workers for the training dataloader.",
    )

    val_parser = ArgumentParser(prog="")
    AD.add_class_args(val_parser, prefix="dataset")
    SegSamplerFactory.add_class_args(val_parser, prefix="sampler")
    val_parser.add_argument(
        "--data_loader.num-workers",
        type=int,
        default=5,
        help="Number of workers for the validation dataloader.",
    )
    data_parser = ArgumentParser(prog="")
    data_parser.add_argument(
        "--train",
        action=ActionParser(parser=train_parser),
        help="Training data configuration block.",
    )
    data_parser.add_argument(
        "--val",
        action=ActionParser(parser=val_parser),
        help="Validation data configuration block.",
    )
    parser.add_argument(
        "--data",
        action=ActionParser(parser=data_parser),
        help="Data configuration block containing train/val settings.",
    )
    model_class.add_class_args(parser, prefix="dac_model")
    AudioMultiDiscriminator.add_class_args(parser, prefix="discrim_model")
    Trainer.add_class_args(
        parser,
        prefix="trainer",
    )
    ddp.add_ddp_args(parser)
    parser.add_argument("--seed", type=int, default=1123581321, help="random seed")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="Verbosity level: 0=error, 1=warning, 2=info, 3=debug.",
    )

    return parser


def main() -> None:
    """Parse CLI arguments and launch DAC training."""
    parser = ArgumentParser(description="Train DAC model")
    parser.add_argument(
        "--cfg",
        action=ActionConfigFile,
        help="Path to a configuration file.",
    )

    subcommands = parser.add_subcommands()

    for k, v in model_dict.items():
        parser_k = make_parser(v)
        subcommands.add_subcommand(k, parser_k)

    args = parser.parse_args()
    try:
        gpu_id = int(os.environ["LOCAL_RANK"])
    except:
        gpu_id = 0

    model_type = args.subcommand
    args_sc = vars(args)[model_type]

    if gpu_id == 0:
        config_file = Path(args_sc.trainer.exp_path) / "config.yaml"
        parser.save(args, str(config_file), format="yaml", overwrite=True)

    args_sc.model_class = model_dict[model_type]
    # torch docs recommend using forkserver
    multiprocessing.set_start_method("forkserver")
    train_model(gpu_id, args_sc)


if __name__ == "__main__":
    main()
