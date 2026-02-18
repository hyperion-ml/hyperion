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
from hyperion.torch.models.freevc import HFWavLMFreeVC
from hyperion.torch.narchs import AudioFeatsMVN
from hyperion.torch.trainers.vi_anonymizer_trainer import VIAnonymizerTrainer as Trainer
from hyperion.torch.utils import ddp
from hyperion.utils.misc import PathLike

model_dict = {
    "hf_wavlm_freevc": HFWavLMFreeVC,
}


def init_data(
    partition: str, rank: int, num_gpus: int, **kwargs: Any
) -> torch.utils.data.DataLoader:
    """Initialize dataset, sampler, and dataloader for a partition.

    Args:
        partition: Dataset split name, e.g. ``"train"`` or ``"val"``.
        rank: Distributed rank of the current process.
        num_gpus: Number of GPUs used by the process group.
        **kwargs: Parsed configuration dictionary containing ``data`` settings.
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

    # num_workers = kwargs["data_loader"]["num_workers"]
    # num_workers_per_gpu = int((num_workers + num_gpus - 1) / num_gpus)
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


def init_vc_model(
    rank: int, model_class: Type[HyperTorchModel], model_args: Dict[str, Any]
) -> HyperTorchModel:
    """Initialize the VC model instance.

    Args:
        rank: Distributed rank of the current process.
        model_class: VC model class to instantiate.
        model_args: Configuration dictionary for the VC model.
    """
    if rank == 0:
        logging.info("vc_model network args={}".format(model_args))

    model = model_class(**model_args)
    if rank == 0:
        logging.info("vc_model={}".format(model))
    return model


def init_discrim_model(
    rank: int, model_args: Dict[str, Any]
) -> AudioMultiDiscriminator:
    """Initialize discriminator model.

    Args:
        rank: Distributed rank of the current process.
        model_args: Configuration dictionary for discriminator initialization.
    """
    if rank == 0:
        logging.info("discrim_model network args={}".format(model_args))

    model = AudioMultiDiscriminator(**model_args)
    if rank == 0:
        logging.info("discrim_model={}".format(model))
    return model


def init_audio_feats(rank: int, model_args: Dict[str, Any]) -> AudioFeatsMVN:
    """Initialize audio-feature network used by loss functions.

    Args:
        rank: Distributed rank of the current process.
        model_args: Configuration dictionary for feature extractor initialization.
    """
    if rank == 0:
        logging.info("audio_feats network args={}".format(model_args))

    model = AudioFeatsMVN(**model_args)
    if rank == 0:
        logging.info("audio_feats={}".format(model))
    return model


def init_xvector(model_file: PathLike, rank: int) -> HyperTorchModel:
    """Load x-vector model checkpoint.

    Args:
        model_file: Path to x-vector model checkpoint.
        rank: Distributed rank of the current process.
    """
    if rank == 0:
        logging.info("loading xvector_model: %s", model_file)
    model = HyperTorchModel.auto_load(model_file)
    if rank == 0:
        logging.info("x-vector_model={}".format(model))
    return model


def train_model(gpu_id: int, args: Any) -> None:
    """Run distributed VI anonymizer fine-tuning for one process/GPU.

    Args:
        gpu_id: Local GPU index used by this process.
        args: Parsed model-specific command-line namespace.
    """
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
    vc_model = init_vc_model(rank, kwargs["model_class"], kwargs["vc_model"])
    discrim_model = init_discrim_model(rank, kwargs["discrim_model"])
    audio_feats = init_audio_feats(rank, kwargs["loss_audio_feats"])
    xvector_model = init_xvector(kwargs["xvector_model_file"], rank)

    trn_args = Trainer.filter_args(**kwargs["trainer"])
    if rank == 0:
        logging.info(f"trainer args={trn_args}")

    trainer = Trainer(
        vc_model=vc_model,
        discrim_model=discrim_model,
        audio_feats=audio_feats,
        xvector_model=xvector_model,
        device=device,
        ddp=world_size > 1,
        **trn_args,
    )
    trainer.load_last_checkpoint()
    trainer.fit(train_loader, val_loader)

    ddp.ddp_cleanup()


def make_parser(model_class: Type[HyperTorchModel]) -> ArgumentParser:
    """Build parser for a specific anonymizer model subcommand.

    Args:
        model_class: Model class used to register model-specific arguments.
    """
    parser = ArgumentParser()

    parser.add_argument("--cfg", action=ActionConfigFile, help="configuration file")

    train_parser = ArgumentParser(prog="")
    AD.add_class_args(train_parser, prefix="dataset")
    SegSamplerFactory.add_class_args(train_parser, prefix="sampler")
    train_parser.add_argument(
        "--data_loader.num-workers",
        type=int,
        default=5,
        help="number of worker processes for the training dataloader",
    )

    val_parser = ArgumentParser(prog="")
    AD.add_class_args(val_parser, prefix="dataset")
    SegSamplerFactory.add_class_args(val_parser, prefix="sampler")
    val_parser.add_argument(
        "--data_loader.num-workers",
        type=int,
        default=5,
        help="number of worker processes for the validation dataloader",
    )
    data_parser = ArgumentParser(prog="")
    data_parser.add_argument(
        "--train",
        action=ActionParser(parser=train_parser),
        help="training data configuration block",
    )
    data_parser.add_argument(
        "--val",
        action=ActionParser(parser=val_parser),
        help="validation data configuration block",
    )
    parser.add_argument(
        "--data", action=ActionParser(parser=data_parser), help="data configuration"
    )
    model_class.add_class_args(parser, prefix="vc_model")
    AudioMultiDiscriminator.add_class_args(parser, prefix="discrim_model")
    AudioFeatsMVN.add_class_args(parser, prefix="loss_audio_feats")
    parser.add_argument(
        "--xvector-model-file",
        type=str,
        required=True,
        help="input x-vector model checkpoint path",
    )
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
        help="verbosity level (0=warning, 1=info, 2=debug, 3=trace)",
    )

    return parser


def main() -> None:
    """Parse CLI arguments and start VI anonymizer fine-tuning.

    Args:
        None.
    """
    parser = ArgumentParser(description="Train VI anonymizer model")
    parser.add_argument("--cfg", action=ActionConfigFile, help="configuration file")

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
        # try:
        #     config_file = Path(args_sc.trainer.exp_path) / "config.yaml"
        #     parser.save(args, str(config_file), format="yaml", overwrite=True)
        # except:
        #     pass

    args_sc.model_class = model_dict[model_type]
    # torch docs recommend using forkserver
    multiprocessing.set_start_method("forkserver")
    train_model(gpu_id, args_sc)


if __name__ == "__main__":
    main()
