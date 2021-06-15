#!/usr/bin/env python
"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
# Edited from hyperion/bin/torch-train-resnet-xvec-from-wav.py to run dino ssl
# (NOTE) changes in var names from original dino repo to this: student --> model & teacher --> model_teacher
import sys
import os
from pathlib import Path
from jsonargparse import ArgumentParser, ActionConfigFile, ActionParser, namespace_to_dict
import time
import logging
import multiprocessing 
import copy 

import numpy as np

import torch
import torch.nn as nn
# import torch.multiprocessing as mp

from hyperion.hyp_defs import config_logger, set_float_cpu
from hyperion.torch.utils import open_device
from hyperion.torch.utils import ddp
from hyperion.torch.utils import dinossl
# from hyperion.torch.helpers import OptimizerFactory as OF
# from hyperion.torch.lr_schedulers import LRSchedulerFactory as LRSF
from hyperion.torch.trainers import DINOSSLXVectorTrainerFromWav as Trainer
from hyperion.torch.models import ResNetXVector as XVec
from hyperion.torch.data import AudioDataset as AD
from hyperion.torch.data import ClassWeightedSeqSampler as Sampler
from hyperion.torch.metrics import CategoricalAccuracy

#from hyperion.torch.layers import AudioFeatsFactory as AFF
#from hyperion.torch.layers import MeanVarianceNorm as MVN
from hyperion.torch.narchs import AudioFeatsMVN as AF

# from torch.utils.data import dataloader
# from torch.multiprocessing import reductions
# from multiprocessing.reduction import ForkingPickler

# default_collate_func = dataloader.default_collate
# def default_collate_override(batch):
#   dataloader._use_shared_memory = False
#   return default_collate_func(batch)

# setattr(dataloader, 'default_collate', default_collate_override)

# for t in torch._storage_classes:
#     if t in ForkingPickler._extra_reducers:
#         del ForkingPickler._extra_reducers[t]

# class FeatExtractor(nn.Module):

#     def __init__(self, feat_extractor, mvn=None):
#         super().__init__()

#         self.feat_extractor = feat_extractor
#         self.mvn = mvn

#     def forward(self, x):
#         f = self.feat_extractor(x)
#         if self.mvn is not None:
#             f = self.mvn(f)

#         f = f.transpose(1,2).contiguous()
#         return f



# def init_device(num_gpus):
#     set_float_cpu('float32')
#     logging.info('initializing devices num_gpus={}'.format(num_gpus))
#     device = open_device(num_gpus=num_gpus)
#     return device


def init_data(audio_path, train_list, val_list, 
              train_aug_cfg, val_aug_cfg, num_workers, 
              num_gpus, rank, **kwargs):

    ad_args = AD.filter_args(**kwargs)
    sampler_args = Sampler.filter_args(**kwargs)
    if rank == 0:
        logging.info('audio dataset args={}'.format(ad_args))
        logging.info('sampler args={}'.format(sampler_args))
        logging.info('init datasets')

    train_data = AD(audio_path, train_list, aug_cfg=train_aug_cfg, **ad_args, dinossl_chunk_len_mult=kwargs['dinossl_chunk_len_mult'], dinossl_n_chunks=kwargs['dinossl_local_crops_number'] + 2)
    val_data = AD(audio_path, val_list, aug_cfg=val_aug_cfg, is_val=True, **ad_args)

    if rank == 0:
        logging.info('init samplers')
    train_sampler = Sampler(train_data, **sampler_args)
    val_sampler = Sampler(val_data, **sampler_args)

    num_workers_per_gpu = int((num_workers + num_gpus - 1) / num_gpus)
    largs = {'num_workers': num_workers_per_gpu, 'pin_memory': True} if num_gpus > 0 else {}

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_sampler = train_sampler, **largs)

    test_loader = torch.utils.data.DataLoader(
        val_data, batch_sampler = val_sampler, **largs)

    return train_loader, test_loader


# def init_feats1(**kwargs):
#     kwargs = kwargs['feats']
#     feat_args = AFF.filter_args(**kwargs['audio_feats'])
#     mvn_args = MVN.filter_args(**kwargs['mvn'])
#     logging.info('feat args={}'.format(feat_args))
#     logging.info('mvn args={}'.format(mvn_args))
#     logging.info('initializing feature extractor')
#     feat_extractor = AFF.create(**feat_args)
#     mvn = None
#     if mvn_args['norm_mean'] or mvn_args['norm_var']:
#         logging.info('initializing short-time mvn')
#         mvn = MVN(**mvn_args)

#     feat_extractor = FeatExtractor(feat_extractor, mvn)
#     logging.info('feat-extractor={}'.format(feat_extractor))
#     return feat_extractor


def init_feats(rank, **kwargs):
    feat_args = AF.filter_args(**kwargs['feats'])
    if rank == 0:
        logging.info('feat args={}'.format(feat_args))
        logging.info('initializing feature extractor')
    feat_extractor = AF(trans=True, **feat_args)
    if rank == 0:
        logging.info('feat-extractor={}'.format(feat_extractor))
    return feat_extractor


def init_xvector(num_classes, rank, **kwargs):
    xvec_args = XVec.filter_args(**kwargs)
    if rank == 0:
        logging.info('xvector network args={}'.format(xvec_args))
    xvec_args['num_classes'] = num_classes
    model = XVec(**xvec_args)
    loss = None
    if kwargs['dinossl']:
        dinossl_args = dinossl.filter_args(**kwargs)
        if rank == 0:
            logging.info('dinossl args={}'.format(dinossl_args))
        embed_dim = model.classif_net.in_feats
        # model
        model_teacher = copy.deepcopy(model) # teacher and student start with the same weights. copying values of parameters from model also happens in this line differently from original dino repo
        model = dinossl.MultiCropWrapper(model, dinossl.DINOHead(embed_dim, dinossl_args['dinossl_out_dim'], use_bn=dinossl_args['dinossl_use_bn_in_head'], norm_last_layer=dinossl_args['dinossl_norm_last_layer'])) # multi-crop wrapper handles forward with inputs of different chunk lengths
        model_teacher = dinossl.MultiCropWrapper(model_teacher, dinossl.DINOHead(embed_dim, dinossl_args['dinossl_out_dim'], use_bn=dinossl_args['dinossl_use_bn_in_head']))
        model = [model, model_teacher]
        # loss
        loss = dinossl.DINOLoss(dinossl_args['dinossl_out_dim'],
        dinossl_args['dinossl_local_crops_number'] + 2,  # total number of crops = 2 global crops + local_crops_number
        dinossl_args['dinossl_warmup_teacher_temp'],
        dinossl_args['dinossl_teacher_temp'],
        dinossl_args['dinossl_warmup_teacher_temp_epochs'],
        kwargs['epochs'],) # to(device) will happen in torch_trainer_dinossl.py
    if rank == 0:
        logging.info('x-vector-model={}'.format(model))
    return model, loss


# def init_opt(model, rank, **kwargs):

#     opt_args = OF.filter_args(**kwargs['opt'])
#     if kwargs['num_gpus'] > 0 and kwargs['ddp_type'] != 'ddp':
#         opt_args['oss'] = True
#     lrsch_args = LRSF.filter_args(**kwargs['lrsch'])
#     if rank == 0:
#         logging.info('optimizer args={}'.format(opt_args))
#         logging.info('lr scheduler args={}'.format(lrsch_args))
#     optimizer = OF.create(model.parameters(), **opt_args)
#     lr_sch = LRSF.create(optimizer, **lrsch_args)
#     return optimizer, lr_sch


def train_xvec(gpu_id, args):
        
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    kwargs = namespace_to_dict(args)
    torch.manual_seed(args.seed)
    set_float_cpu('float32')

    ddp_args = ddp.filter_ddp_args(**kwargs)
    device, rank, world_size = ddp.ddp_init(gpu_id, **ddp_args)
    #use_gpu = ddp_args['num_gpus'] > 0
    #kwargs['use_gpu'] = use_gpu
    kwargs['rank'] = rank
    # train_loader, test_loader = init_data(
    #     args.audio_path, args.train_list, args.val_list, 
    #     args.train_aug_cfg, args.val_aug_cfg, args.num_workers, 
    #     use_gpu, **kwargs)

    train_loader, test_loader = init_data(**kwargs)
    feat_extractor = init_feats(**kwargs)
    model, dino_loss = init_xvector(train_loader.dataset.num_classes, **kwargs)
    #model.to(device)
    #optimizer, lr_sch = init_opt(model, **kwargs)
    trn_args = Trainer.filter_args(**kwargs)
    if rank == 0:
        logging.info('trainer args={}'.format(trn_args))
    metrics = {}
    # trainer = Trainer(model, feat_extractor, optimizer, 
    #                   device=device, metrics=metrics, lr_scheduler=lr_sch,
    #                   ddp=world_size>1, **trn_args)
    trn_args['niter_per_ep'] = len(train_loader) # (JJ: TODO - currently this does NOT consider grad_acc_steps
    trn_args['batch_size'] = kwargs['batch_size']
    trainer = Trainer(model, feat_extractor,
                      device=device, metrics=metrics, 
                      ddp=world_size>1, loss=dino_loss, **trn_args)
    if args.resume:
        trainer.load_last_checkpoint()
    trainer.fit(train_loader, val_data=None) # (JJ: TODO - We don't use test_loader in SSL but still run the process (better to remove it). )

    ddp.ddp_cleanup()


if __name__ == '__main__':

    parser = ArgumentParser(
        description='Train XVector with ResNet encoder from audio files in a self-supervised fashion: Currently it follows DINO')

    parser.add_argument('--cfg', action=ActionConfigFile)
    parser.add_argument('--audio-path', required=True)
    parser.add_argument('--train-list', required=True)
    parser.add_argument('--val-list', required=True)

    AD.add_class_args(parser)
    Sampler.add_class_args(parser)

    parser.add_argument('--train-aug-cfg', default=None)
    parser.add_argument('--val-aug-cfg', default=None)

    parser.add_argument('--num-workers', type=int, default=5, 
                        help='num_workers of data loader')

    AF.add_class_args(parser, prefix='feats')
    # feats_parser = ArgumentParser()
    # AFF.add_class_args(feats_parser, prefix='audio_feats')
    # MVN.add_class_args(feats_parser, prefix='mvn')
    # parser.add_argument('--feats', action=ActionParser(parser=feats_parser), help='feat extraction opts')

    XVec.add_class_args(parser)
    # OF.add_class_args(parser, prefix='opt')
    # LRSF.add_class_args(parser, prefix='lrsch')
    Trainer.add_class_args(parser)
    dinossl.add_dinossl_args(parser)

    ddp.add_ddp_args(parser)
    parser.add_argument('--seed', type=int, default=1123581321, 
                        help='random seed')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training from checkpoint')
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, 
                        choices=[0, 1, 2, 3], type=int)

    parser.add_argument('--local_rank', default=0, type=int)

    args = parser.parse_args()

    #device = init_device(args.num_gpus)
    #mp.spawn(train_xvec, nprocs=args.num_gpus, args=(args,))
    gpu_id = args.local_rank
    del args.local_rank

    if gpu_id == 0:
        try:
            config_file = Path(args.exp_path) / 'config.yaml'
            parser.save(args, str(config_file), format='yaml', overwrite=True)
        except:
            pass

    # torch docs recommend using forkserver
    multiprocessing.set_start_method('forkserver')
    train_xvec(gpu_id, args)

