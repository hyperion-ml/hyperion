#!/usr/bin/env python
"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import sys
import pdb
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
from hyperion.torch.utils import ddp
from hyperion.torch.trainers import TransducerTrainer as Trainer
from hyperion.torch.data import AudioDataset as AD
from hyperion.torch.data import SegSamplerFactory
from hyperion.torch.metrics import CategoricalAccuracy
from hyperion.torch.models import HFWav2Vec2Transducer
from hyperion.torch.models.transducer import Conformer
from hyperion.torch.models.transducer import Decoder
from hyperion.torch.models.transducer import Joiner



model_dict = {
    "hf_wav2vec2transducer": HFWav2Vec2Transducer,
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
    print("ad_args", ad_args)
    dataset = AD(**ad_args)

    if rank == 0:
        logging.info("init %s samplers", partition)
    print("sampler_args", sampler_args)
    sampler = SegSamplerFactory.create(dataset, **sampler_args)

    if rank == 0:
        logging.info("init %s dataloader", partition)

    num_workers = kwargs["data_loader"]["num_workers"]
    num_workers_per_gpu = int((num_workers + num_gpus - 1) / num_gpus)
    largs = (
        {"num_workers": num_workers_per_gpu, "pin_memory": True} if num_gpus > 0 else {}
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, **largs)
    return data_loader


def init_model(num_classes, rank, model_class, **kwargs):
    model_args = model_class.filter_args(**kwargs["model"])
    if rank == 0:
        logging.info("model network args={}".format(model_args))
    # TODO: check model_args 
    model_args["num_classes"] = num_classes
    model = model_class(**model_args)
    if rank == 0:
        logging.info("model={}".format(model))
    return model




def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - attention_dim: Hidden dim for multi-head attention model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - warm_step: The warm_step for Noam optimizer.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,  # For the 100h subset, use 800
            # parameters for conformer
            "feature_dim": 80,
            "encoder_out_dim": 512,
            "subsampling_factor": 4,
            "attention_dim": 512,
            "nhead": 8,
            "dim_feedforward": 2048,
            "num_encoder_layers": 12,
            "vgg_frontend": False,
            # decoder params
            "decoder_embedding_dim": 1024,
            "num_decoder_layers": 2,
            "decoder_hidden_dim": 512,
            # parameters for Noam
            "warm_step": 80000,  # For the 100h subset, use 8k
            "env_info": get_env_info(),
        }
    )

    return params


def get_encoder_model(params: AttributeDict):
    # TODO: We can add an option to switch between Conformer and Transformer
    encoder = Conformer(
        num_features=params.feature_dim,
        output_dim=params.encoder_out_dim,
        subsampling_factor=params.subsampling_factor,
        d_model=params.attention_dim,
        nhead=params.nhead,
        dim_feedforward=params.dim_feedforward,
        num_encoder_layers=params.num_encoder_layers,
        vgg_frontend=params.vgg_frontend,
    )
    return encoder


def get_decoder_model(params: AttributeDict):
    decoder = Decoder(
        vocab_size=params.vocab_size,
        embedding_dim=params.decoder_embedding_dim,
        blank_id=params.blank_id,
        num_layers=params.num_decoder_layers,
        hidden_dim=params.decoder_hidden_dim,
        output_dim=params.encoder_out_dim,
    )
    return decoder


def get_joiner_model(params: AttributeDict):
    joiner = Joiner(
        input_dim=params.encoder_out_dim,
        output_dim=params.vocab_size,
    )
    return joiner


def get_transducer_model(params: AttributeDict):
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)

    model = Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
    )
    return model



def train_model(gpu_id, args):

    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    kwargs = namespace_to_dict(args)
    torch.manual_seed(args.seed)
    set_float_cpu("float32")

    ddp_args = ddp.filter_ddp_args(**kwargs)
    device, rank, world_size = ddp.ddp_init(gpu_id, **ddp_args)
    kwargs["rank"] = rank

    # # for Debug
    # rank = 0
    # kwargs["rank"] = 0
    # device = "cpu"
    # world_size=1
    
    train_loader = init_data(partition="train", **kwargs)
    val_loader = init_data(partition="val", **kwargs)
    # model = init_model(train_loader.dataset.num_classes.values())[0], **kwargs)
    model = init_model(train_loader.dataset.num_classes, **kwargs)

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


def make_parser(model_class):
    parser = ArgumentParser()

    parser.add_argument("--cfg", action=ActionConfigFile)
    train_parser = ArgumentParser(prog="")
    AD.add_class_args(train_parser, prefix="dataset", skip={})
    SegSamplerFactory.add_class_args(train_parser, prefix="sampler")
    # Sampler.add_class_args(train_parser, prefix="sampler")
    train_parser.add_argument(
        "--data_loader.num-workers",
        type=int,
        default=5,
        help="num_workers of data loader",
    )

    val_parser = ArgumentParser(prog="")
    AD.add_class_args(val_parser, prefix="dataset", skip={})
    SegSamplerFactory.add_class_args(val_parser, prefix="sampler")
    # Sampler.add_class_args(val_parser, prefix="sampler")
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


    parser.add_argument(
        "--data.train.dataset.text_file",
        type=str, 
    )

    parser.add_argument(
        "--data.train.dataset.bpe_model",
        type=str, 
    )

    parser.add_argument("--data.val.dataset.text_file", type=str) 
    
    parser.link_arguments(
        "data.train.data_loader.num_workers", "data.val.data_loader.num_workers"
    )

    parser.link_arguments(
        "data.train.dataset.bpe_model", "data.val.dataset.bpe_model"
    )

    # parser.link_arguments(
    #     "data.train.dataset.class_file", "data.val.dataset.class_file"
    # )
    # parser.link_arguments(
    #     "data.train.data_loader.num_workers", "data.val.data_loader.num_workers"
    # )
    # parser.link_arguments(
    #     "data.train.sampler.batch_size", "data.val.sampler.batch_size"
    # )

    model_class.add_class_args(parser, prefix="model")
    Trainer.add_class_args(
        parser, prefix="trainer", train_modes=model_class.valid_train_modes()
    )
    ddp.add_ddp_args(parser)
    parser.add_argument("--seed", type=int, default=1123581321, help="random seed")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    return parser


if __name__ == "__main__":

    parser = ArgumentParser(description="Train Wav2Vec2Transducer model from audio files")
    parser.add_argument("--cfg", action=ActionConfigFile)

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
        try:
            config_file = Path(args_sc.trainer.exp_path) / "config.yaml"
            parser.save(args, str(config_file), format="yaml", overwrite=True)
        except:
            pass

    args_sc.model_class = model_dict[model_type]
    # torch docs recommend using forkserver
    # multiprocessing.set_start_method("forkserver")
    train_model(gpu_id, args_sc)
