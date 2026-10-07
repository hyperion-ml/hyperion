#!/usr/bin/env python
"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any, Type

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
from hyperion.torch.hyper_torch_model import HyperTorchModel
from hyperion.torch.models import HFWav2Vec2XVectorP as W2V2XVecP
from hyperion.torch.models import ResNetXVectorP as RXVecP
from hyperion.torch.models import XVectorP
from hyperion.torch.narchs import HydraHeadType
from hyperion.torch.trainers import XVectorPTrainer as Trainer
from hyperion.torch.utils import ddp

xvecp_dict = {
    "resnet": RXVecP,
    "wav2vec2": W2V2XVecP,
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
        logging.info(f"{partition} audio dataset args={ad_args}")
        logging.info(f"{partition} sampler args={sampler_args}")
        logging.info(f"init {partition} dataset")

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
    num_workers_per_gpu = (
        (num_workers + num_gpus - 1) // num_gpus if num_gpus > 0 else num_workers
    )
    largs = {
        "num_workers": num_workers_per_gpu,
        "pin_memory": num_gpus > 0,
        "persistent_workers": num_workers_per_gpu > 0,
    }
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=dataset.get_collator(), **largs
    )
    return data_loader


def init_xvectorp(
    num_classes: int, rank: int, xvecp_class: Type[XVectorP], **kwargs: Any
) -> XVectorP:
    """Initialize XVectorP model.

    Args:
        num_classes: Number of classes for classification head (if applicable).
        rank: Process rank in distributed training.
        xvecp_class: XVectorP model class to instantiate.
        **kwargs: Parsed configuration dictionary.
    """
    xvecp_args = xvecp_class.filter_args(**kwargs["model"])
    if rank == 0:
        logging.info("XVectorP network args=%s", xvecp_args)

    head_args = xvecp_args.get("head")
    if (
        isinstance(head_args, dict)
        and head_args.get("head_type") == HydraHeadType.CLASSIF
    ):
        xvecp_args["head"]["num_classes"] = num_classes
    model = xvecp_class(**xvecp_args)
    if model.head is None:
        raise ValueError(
            "XVectorP training requires a classification or regression head"
        )
    if rank == 0:
        logging.info("XVectorP model=%s", model)
    return model


def train_xvectorp(args: Any) -> None:
    """Run distributed XVectorP training.

    Args:
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
    try:
        train_loader = init_data(partition="train", **kwargs)
        val_loader = init_data(partition="val", **kwargs)

        num_classes = list(train_loader.dataset.num_classes.values())[0]
        model = init_xvectorp(num_classes, **kwargs)
        if kwargs["init_from_xvector_model_file"] is not None:
            xvec_path = kwargs["init_from_xvector_model_file"]
            if rank == 0:
                logging.info(
                    "Initializing XVectorP model from x-vector model %s", xvec_path
                )

            xvector_model = HyperTorchModel.auto_load(xvec_path)
            model.init_from_xvector(xvector_model)

        trn_args = Trainer.filter_args(**kwargs["trainer"])
        if rank == 0:
            logging.info("trainer args=%s", trn_args)

        trainer = Trainer(
            model,
            device=device,
            ddp=world_size > 1,
            **trn_args,
        )
        trainer.load_last_checkpoint()
        trainer.fit(train_loader, val_loader)
    finally:
        ddp.ddp_cleanup()


def make_parser(xvecp_class: Type[XVectorP]) -> ArgumentParser:
    """Create parser for one XVectorP model subcommand.

    Args:
        xvecp_class: XVectorP model class whose args should be exposed.
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
        help="Number of worker processes for this dataloader.",
    )

    val_parser = ArgumentParser(prog="")
    AD.add_class_args(val_parser, prefix="dataset")
    SegSamplerFactory.add_class_args(val_parser, prefix="sampler")
    val_parser.add_argument(
        "--data_loader.num-workers",
        type=int,
        default=5,
        help="Number of worker processes for this dataloader.",
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
    # parser.link_arguments(
    #     "data.train.dataset.class_files", "data.val.dataset.class_files"
    # )
    # parser.link_arguments(
    #     "data.train.data_loader.num_workers", "data.val.data_loader.num_workers"
    # )

    xvecp_class.add_class_args(parser, prefix="model")
    parser.add_argument(
        "--init-from-xvector-model-file",
        type=str,
        default=None,
        help="Optional x-vector checkpoint used to initialize the XVectorP backbone.",
    )
    Trainer.add_class_args(
        parser,
        prefix="trainer",
    )
    ddp.add_ddp_args(parser)
    parser.add_argument(
        "--seed", type=int, default=1123581321, help="Random seed for training."
    )
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
    """Parse CLI arguments and launch XVectorP training."""
    parser = ArgumentParser(description="Train an XVectorP model from audio files")
    parser.add_argument(
        "--cfg",
        action=ActionConfigFile,
        help="Path to a configuration file.",
    )

    subcommands = parser.add_subcommands()
    for k, v in xvecp_dict.items():
        parser_k = make_parser(v)
        subcommands.add_subcommand(k, parser_k)

    # os.environ["MKL_THREADING_LAYER"] = "GNU"
    # os MKL_SERVICE_FORCE_INTEL=1
    args = parser.parse_args()
    try:
        gpu_id = int(os.environ["LOCAL_RANK"])
    except (KeyError, ValueError):
        gpu_id = 0

    xvecp_type = args.subcommand
    args_sc = vars(args)[xvecp_type]

    if gpu_id == 0:
        try:
            config_file = Path(args_sc.trainer.exp_path) / "config.yaml"
            parser.save(args, str(config_file), format="yaml", overwrite=True)
        except Exception:
            logging.warning("Failed saving training configuration", exc_info=True)

    args_sc.xvecp_class = xvecp_dict[xvecp_type]
    # torch docs recommend using forkserver
    multiprocessing.set_start_method("forkserver")
    train_xvectorp(args_sc)


if __name__ == "__main__":
    main()
