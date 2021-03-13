#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import os
import argparse
import time
import logging

import numpy as np

import torch

from hyperion.hyp_defs import config_logger, set_float_cpu
from hyperion.torch.utils import open_device
from hyperion.torch.helpers import OptimizerFactory as OF
from hyperion.torch.lr_schedulers import LRSchedulerFactory as LRSF
from hyperion.torch.seq_embed import XVector as XVec
from hyperion.torch.trainers import XVectorAdvTrainer as Trainer
from hyperion.torch.data import SeqDataset as SD
from hyperion.torch.data import ClassWeightedSeqSampler as Sampler
from hyperion.torch.metrics import CategoricalAccuracy
from hyperion.torch.adv_attacks import PGDAttack
from hyperion.torch import TorchModelLoader as TML

def train_xvec(data_rspec, train_list, val_list, in_model_path,
               attack_eps, attack_eps_step, attack_random_eps, 
               attack_max_iter, p_attack,
               num_gpus, resume, num_workers, 
               train_mode, **kwargs):

    set_float_cpu('float32')
    logging.info('initializing devices num_gpus={}'.format(num_gpus))
    device = open_device(num_gpus=num_gpus)

    sd_args = SD.filter_args(**kwargs)
    sampler_args = Sampler.filter_args(**kwargs)
    xvec_args = XVec.filter_finetune_args(**kwargs)
    opt_args = OF.filter_args(prefix='opt', **kwargs)
    lrsch_args = LRSF.filter_args(prefix='lrsch', **kwargs)
    trn_args = Trainer.filter_args(**kwargs)
    logging.info('seq dataset args={}'.format(sd_args))
    logging.info('sampler args={}'.format(sampler_args))
    logging.info('xvector finetune args={}'.format(xvec_args))
    logging.info('optimizer args={}'.format(opt_args))
    logging.info('lr scheduler args={}'.format(lrsch_args))
    logging.info('trainer args={}'.format(trn_args))

    logging.info('init datasets')
    train_data = SD(data_rspec, train_list, **sd_args)
    val_data = SD(data_rspec, val_list, is_val=True, **sd_args)

    logging.info('init samplers')
    train_sampler = Sampler(train_data, **sampler_args)
    val_sampler = Sampler(val_data, **sampler_args)

    largs = {'num_workers': num_workers, 'pin_memory': True} if num_gpus>0 else {}

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_sampler = train_sampler, **largs)

    test_loader = torch.utils.data.DataLoader(
        val_data, batch_sampler = val_sampler, **largs)

    xvec_args['num_classes'] = train_data.num_classes
    model = TML.load(in_model_path)
    model.rebuild_output_layer(**xvec_args)
    if train_mode == 'ft-embed-affine':
        model.freeze_preembed_layers()
    logging.info(str(model))

    optimizer = OF.create(model.parameters(), **opt_args)
    lr_sch = LRSF.create(optimizer, **lrsch_args)
    metrics = { 'acc': CategoricalAccuracy() }

    attack = PGDAttack(model, eps=attack_eps, alpha=attack_eps_step,
                       norm=float('inf'), max_iter=attack_max_iter,
                       random_eps=attack_random_eps)
    
    trainer = Trainer(model, optimizer, attack, p_attack=p_attack,
                      device=device, metrics=metrics, lr_scheduler=lr_sch,
                      data_parallel=(num_gpus>1), train_mode=train_mode,
                      **trn_args)
    if resume:
        trainer.load_last_checkpoint()
    trainer.fit(train_loader, test_loader)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Fine-tune x-vector model with adv training')

    parser.add_argument('--data-rspec', dest='data_rspec', required=True)
    parser.add_argument('--train-list', dest='train_list', required=True)
    parser.add_argument('--val-list', dest='val_list', required=True)

    SD.add_argparse_args(parser)
    Sampler.add_argparse_args(parser)

    parser.add_argument('--num-workers', type=int, default=5, 
                        help='num_workers of data loader')
    parser.add_argument('--in-model-path', required=True)
    XVec.add_argparse_finetune_args(parser)
    OF.add_argparse_args(parser, prefix='opt')
    LRSF.add_argparse_args(parser, prefix='lrsch')
    Trainer.add_argparse_args(parser)

    parser.add_argument('--num-gpus', type=int, default=1,
                        help='number of gpus, if 0 it uses cpu')
    parser.add_argument('--seed', type=int, default=1123581321, 
                        help='random seed (default: 1)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training from checkpoint')
    parser.add_argument('--train-mode', default='ft-embed-affine',
                        choices=['ft-full', 'ft-embed-affine'],
                        help=('ft-full: adapt full x-vector network'
                              'ft-embed-affine: adapt affine transform before embedding'))

    parser.add_argument('--attack-eps', required=True, type=float,
                       help='epsilon adversarial attack')
    parser.add_argument('--attack-eps-step', required=True, type=float,
                       help='eps step adversarial attack')
    parser.add_argument('--attack-random-eps', default=False, 
                       action='store_true',
                       help='use random eps in adv. attack')
    parser.add_argument('--attack-max-iter', default=10, type=int, 
                       help='number of iterations for adversarial optimization')
    parser.add_argument('--p-attack', default=0.5, type=float, 
                       help='ratio of batches with adv attack')
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, 
                        choices=[0, 1, 2, 3], type=int)

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    torch.manual_seed(args.seed)
    del args.seed

    train_xvec(**vars(args))

