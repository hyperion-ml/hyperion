#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
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
from hyperion.torch.seq_embed import TDNNXVector as XVec
from hyperion.torch.trainers import XVectorTrainer as Trainer
from hyperion.torch.data import SeqDataset as SD
from hyperion.torch.data import ClassWeightedSeqSampler as Sampler
from hyperion.torch.metrics import CategoricalAccuracy

def train_xvec(data_rspec, train_list, val_list, exp_path,
               epochs, num_gpus, log_interval, resume, num_workers, 
               grad_acc_steps, use_amp, **kwargs):

    set_float_cpu('float32')
    logging.info('initializing devices num_gpus={}'.format(num_gpus))
    device = open_device(num_gpus=num_gpus)

    sd_args = SD.filter_args(**kwargs)
    sampler_args = Sampler.filter_args(**kwargs)
    xvec_args = XVec.filter_args(**kwargs)
    opt_args = OF.filter_args(prefix='opt', **kwargs)
    lrsch_args = LRSF.filter_args(prefix='lrsch', **kwargs)
    logging.info('seq dataset args={}'.format(sd_args))
    logging.info('sampler args={}'.format(sampler_args))
    logging.info('xvector network args={}'.format(xvec_args))
    logging.info('optimizer args={}'.format(opt_args))
    logging.info('lr scheduler args={}'.format(lrsch_args))

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
    model = XVec(**xvec_args)
    logging.info(str(model))

    optimizer = OF.create(model.parameters(), **opt_args)
    lr_sch = LRSF.create(optimizer, **lrsch_args)
    metrics = { 'acc': CategoricalAccuracy() }
    
    trainer = Trainer(model, optimizer, epochs, exp_path, 
                      grad_acc_steps=grad_acc_steps,
                      device=device, metrics=metrics, lr_scheduler=lr_sch,
                      data_parallel=(num_gpus>1), use_amp=use_amp)
    if resume:
        trainer.load_last_checkpoint()
    trainer.fit(train_loader, test_loader)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Train XVector with TDNN encoder')

    parser.add_argument('--data-rspec', dest='data_rspec', required=True)
    parser.add_argument('--train-list', dest='train_list', required=True)
    parser.add_argument('--val-list', dest='val_list', required=True)

    SD.add_argparse_args(parser)
    Sampler.add_argparse_args(parser)


    # parser.add_argument('--batch-size', type=int, default=64,
    #                     help='input batch size for training')
    # parser.add_argument('--test-batch-size', type=int, default=64,
    #                    help='input batch size for testing')
    parser.add_argument('--num-workers', type=int, default=5, 
                        help='num_workers of data loader')

    parser.add_argument('--grad-acc-steps', type=int, default=1, 
                        help='gradient accumulation batches before weigth update')

    parser.add_argument('--epochs', type=int, default=200, 
                        help='number of epochs')

    XVec.add_argparse_args(parser)
    OF.add_argparse_args(parser, prefix='opt')
    LRSF.add_argparse_args(parser, prefix='lrsch')

    parser.add_argument('--num-gpus', type=int, default=1,
                        help='number of gpus, if 0 it uses cpu')
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, 
                        help='how many batches to wait before logging training status')

    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training from checkpoint')

    parser.add_argument('--use-amp', action='store_true', default=False,
                        help='use mixed precision training')

    parser.add_argument('--exp-path', help='experiment path')

    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    torch.manual_seed(args.seed)
    del args.seed

    train_xvec(**vars(args))

