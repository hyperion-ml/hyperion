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
import gc
import k2
import numpy as np
import torch
import torch.nn as nn
from hyperion.hyp_defs import config_logger, set_float_cpu
from hyperion.torch import TorchModelLoader as TML
from hyperion.torch.data import AudioDataset as AD
from hyperion.torch.data import SegSamplerFactory
from hyperion.torch.metrics import CategoricalAccuracy
from hyperion.torch.models import (HFWav2Vec2RNNRNNTransducer,
                                   HFWav2Vec2RNNTransducer,
                                   HFWav2Vec2RNNFiLMTransducer,
                                   HFWav2Vec2RNNTransducerResnet1D,
                                   HFWav2Vec2RNNFiLMTransducerResnet1D)
from hyperion.torch.trainers import TransducerLanguageIDTrainer as Trainer
from hyperion.torch.utils import ddp
from jsonargparse import (ActionConfigFile, ActionParser, ArgumentParser,
                          namespace_to_dict)
from torch.nn.utils.rnn import pad_sequence


import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='torch.distributed.distributed_c10d')

model_dict = {
    "hf_wav2vec2rnn_transducer_resnet1d": HFWav2Vec2RNNTransducerResnet1D,
    "hf_wav2vec2rnn_film_transducer_resnet1d": HFWav2Vec2RNNFiLMTransducerResnet1D,
    
}


def transducer_language_collate(batch):
    audio = []
    audio_length = []
    target = []
    language = []
    for record in batch:
        wav = torch.as_tensor(record["x"])
        audio.append(wav)
        audio_length.append(wav.shape[0])
        target.append(record["text"])
        language.append(record["language"])
    audio = pad_sequence(audio).transpose(0, 1)
    audio_length = torch.as_tensor(audio_length)

    # sort audios by length
    sort_idx = torch.argsort(audio_length, descending=True)
    audio = audio[sort_idx]
    audio_length = audio_length[sort_idx]
    target = [target[k] for k in sort_idx]
    target = k2.RaggedTensor(target)

    language = [language[k] for k in sort_idx]
    language = torch.as_tensor(language)

    # FiLM: add language ID to the input
    batch = {
        "x": audio,
        "x_lengths": audio_length,
        "text": target,
        "language": language,
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
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, **largs, collate_fn=transducer_language_collate)
    return data_loader

def init_model(num_classes, loss_class_weight, in_model_file, rank, model_class, **kwargs):
    model_args = model_class.filter_finetune_args(**kwargs["model"])
    # model_args = model_class.filter_args(**kwargs["model"])
    if rank == 0:
        logging.info("model network ft args={}".format(model_args))
    model_args["languageid"]["num_classes"] = num_classes
    # model_args["loss_class_weight"] = loss_class_weight
    model = TML.load(in_model_file)
    logging.info(model_args)
    model.change_config(**model_args)
    if rank == 0:
        logging.info("model={}".format(model))
    return model
# def init_model(in_model_transducer, in_model_lid, rank, model_class, **kwargs):
#     # load pretrained models
#     model_wav2transducer = torch.load(in_model_transducer)
#     model_wav2lid = torch.load(in_model_lid) 
#     if rank == 0:
#         logging.info("init joint model")
#         logging.info("hf_feats network ft args={}".format(model_wav2transducer["model_cfg"]["hf_feats"]))
#         logging.info("transducer network ft args={}".format(model_wav2transducer["model_cfg"]["transducer"]))
#         logging.info("languageid network ft args={}".format(model_wav2lid["model_cfg"]["languageid"]))
#         logging.info("feat_fusion_start={}".format(model_wav2transducer["model_cfg"]["feat_fusion_start"]))
#         logging.info("feat_fusion_method_transducer={}".format(model_wav2transducer["model_cfg"]["feat_fusion_method"]))
#         logging.info("feat_fusion_method_languageid={}".format(model_wav2lid["model_cfg"]["feat_fusion_method"]))

#     # init joint model
#     model = model_class(hf_feats=model_wav2transducer["model_cfg"]["hf_feats"], 
#                         transducer=model_wav2transducer["model_cfg"]["transducer"], 
#                         languageid=model_wav2lid["model_cfg"]["languageid"],
#                         feat_fusion_start=model_wav2transducer["model_cfg"]["feat_fusion_start"],
#                         feat_fusion_method_transducer=model_wav2transducer["model_cfg"]["feat_fusion_method"],
#                         feat_fusion_method_languageid=model_wav2lid["model_cfg"]["feat_fusion_method"],
#                         loss_weight_transducer=kwargs["model"]["loss_weight_transducer"],
#                         loss_weight_lid=kwargs["model"]["loss_weight_lid"],
#                         lid_length=kwargs["model"]["lid_length"],
#                         )

#     copy_model_parameters(model, model_wav2transducer["model_state_dict"], model_wav2lid["model_state_dict"], rank)


#     # add finetune args
#     model_args = model_class.filter_finetune_args(**kwargs["model"])

#     # model_args = model_class.filter_args(**kwargs["model"])
#     if rank == 0:
#         logging.info("model network ft args={}".format(model_args))
#     model_args["languageid"]["num_classes"] = model_wav2lid["model_cfg"]["languageid"]["num_classes"]
#     model.change_config(**model_args)
#     if rank == 0:
#         logging.info("model={}".format(model))

#     model_wav2transducer = None
#     model_wav2lid = None
#     gc.collect()
#     torch.cuda.empty_cache()
#     return model


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
    # model = init_model(list(train_loader.dataset.num_classes.values())[0], **kwargs)
    model = init_model(list(train_loader.dataset.num_classes.values())[0],
                       train_loader.batch_sampler.class_info["weights"],
                        **kwargs)

    trn_args = Trainer.filter_args(**kwargs["trainer"])
    if rank == 0:
        logging.info("trainer args={}".format(trn_args))
    metrics = {} #{"acc": CategoricalAccuracy()}
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
    train_parser.add_argument(
        "--data_loader.num-workers",
        type=int,
        default=5,
        help="num_workers of data loader",
    )

    val_parser = ArgumentParser(prog="")
    AD.add_class_args(val_parser, prefix="dataset", skip={})
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


    parser.add_argument(
        "--data.train.dataset.text_file",
        type=str, 
    )
    
    parser.add_argument("--data.val.dataset.text_file", type=str) 
    
    parser.add_argument(
        "--data.train.dataset.bpe_model",
        type=str, 
    )

    parser.link_arguments(
        "data.train.data_loader.num_workers", "data.val.data_loader.num_workers"
    )

    parser.link_arguments(
        "data.train.dataset.bpe_model", "data.val.dataset.bpe_model"
    )


    # parser.add_argument("--in-model-transducer", required=True)
    # parser.add_argument("--in-model-lid", required=True)
    parser.add_argument("--in-model-file", required=True)

    model_class.add_finetune_args(parser, prefix="model")
    # model_class.add_class_args(parser, prefix="model")
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
    parser = ArgumentParser(description="Fine-tune  Wav2Vec2Transducer model from audio files")
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
    multiprocessing.set_start_method("forkserver")
    train_model(gpu_id, args_sc)
