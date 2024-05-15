#!/usr/bin/env python
"""
 Copyright 2022 Johns Hopkins University  (Author: Yen-Ju Lu, Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

import k2
import numpy as np
import torch
import torch.nn as nn
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)
from torch.nn.utils.rnn import pad_sequence

from hyperion.hyp_defs import config_logger, set_float_cpu
from hyperion.torch.data import AudioDataset as AD
from hyperion.torch.data import SegSamplerFactory
from hyperion.torch.models import Wav2ConformerV1RNNTransducer, Wav2RNNRNNTransducer
from hyperion.torch.trainers import TransducerTrainer as Trainer
from hyperion.torch.utils import ddp

model_dict = {
    "rnn_rnn_transducer": Wav2RNNRNNTransducer,
    "conformer_v1_rnn_transducer": Wav2ConformerV1RNNTransducer,
}


def transducer_collate(batch):
    audio = []
    audio_length = []
    target = []
    for record in batch:
        audio_length.append(record["x"].shape[0])
    audio_length = torch.as_tensor(audio_length)
    if not torch.all(audio_length[:-1] >= audio_length[1:]):
        sort_idx = torch.argsort(audio_length, descending=True)
        batch = [batch[i] for i in sort_idx]

    audio_length = []
    for record in batch:
        wav = torch.as_tensor(record["x"])
        audio.append(wav)
        audio_length.append(wav.shape[0])
        target.append(record["text"])
    audio = pad_sequence(audio)
    audio_length = torch.as_tensor(audio_length)
    target = k2.RaggedTensor(target)
    batch = {
        "x": torch.transpose(audio, 0, 1),
        "x_lengths": audio_length,
        "text": target,
    }
    return batch


def init_data(partition, rank, num_gpus, **kwargs):
    data_kwargs = kwargs["data"][partition]
    ad_args = AD.filter_args(**data_kwargs["dataset"])
    sampler_args = data_kwargs["sampler"]
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

    num_workers = data_kwargs["data_loader"]["num_workers"]
    num_workers_per_gpu = int((num_workers + num_gpus - 1) / num_gpus)
    largs = (
        {"num_workers": num_workers_per_gpu, "pin_memory": True} if num_gpus > 0 else {}
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        **largs,
        collate_fn=dataset.get_collator(),  # collate_fn=transducer_collate
    )
    return data_loader


# def init_model(blank_id, vocab_size, rank, model_class, **kwargs):
#     model_args = model_class.filter_args(**kwargs["model"])
#     if rank == 0:
#         logging.info("model network args={}".format(model_args))
#     # TODO: check model_args
#     model_args["transducer"]["decoder"]["blank_id"] = blank_id
#     model_args["transducer"]["decoder"]["vocab_size"] = vocab_size
#     model = model_class(**model_args)
#     if rank == 0:
#         logging.info("model={}".format(model))
#     return model


def init_model(rank, model_class, tokenizers, **kwargs):
    model_args = model_class.filter_args(**kwargs["model"])
    if rank == 0:
        logging.info("model network args={}".format(model_args))

    tokenizer = list(tokenizers.items())[0][1]
    model_args["transducer"]["decoder"]["blank_id"] = tokenizer.blank_id
    model_args["transducer"]["decoder"]["vocab_size"] = tokenizer.vocab_size
    model = model_class(**model_args)
    if rank == 0:
        logging.info("model={}".format(model))
    return model


def train_model(gpu_id, args):
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    kwargs = namespace_to_dict(args)
    torch.manual_seed(args.seed)
    set_float_cpu("float32")
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False

    ddp_args = ddp.filter_ddp_args(**kwargs)
    device, rank, world_size = ddp.ddp_init(gpu_id, **ddp_args)
    kwargs["rank"] = rank

    train_loader = init_data(partition="train", **kwargs)
    val_loader = init_data(partition="val", **kwargs)
    # model = init_model(
    #     train_loader.dataset.sp.piece_to_id("<blk>"),
    #     train_loader.dataset.sp.get_piece_size(),
    #     **kwargs,
    # )

    model = init_model(
        tokenizers=train_loader.dataset.tokenizers,
        **kwargs,
    )

    trn_args = Trainer.filter_args(**kwargs["trainer"])
    if rank == 0:
        logging.info("trainer args={}".format(trn_args))
    metrics = {}
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

    # parser.add_argument(
    #     "--data.train.dataset.text_file",
    #     type=str,
    # )

    # parser.add_argument("--data.val.dataset.text_file", type=str)

    # parser.add_argument(
    #     "--data.train.dataset.bpe_model",
    #     type=str,
    # )

    parser.link_arguments(
        "data.train.data_loader.num_workers", "data.val.data_loader.num_workers"
    )

    parser.link_arguments(
        "data.train.dataset.tokenizer_mappings", "data.val.dataset.tokenizer_mappings"
    )
    parser.link_arguments(
        "data.train.dataset.tokenizer_files", "data.val.dataset.tokenizer_files"
    )
    parser.link_arguments("data.train.dataset.bpe_model", "data.val.dataset.bpe_model")

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


def main():
    parser = ArgumentParser(description="Train Transducer model from audio files")
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
        except Exception as err:
            logging.warning(f"{err}")

    args_sc.model_class = model_dict[model_type]
    # torch docs recommend using forkserver
    # multiprocessing.set_start_method("forkserver")
    train_model(gpu_id, args_sc)


if __name__ == "__main__":
    main()
