#!/usr/bin/env python
"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import sys
import os
import argparse
import time
import logging

import numpy as np

import torch
import torch.nn as nn

from hyperion.hyp_defs import config_logger, set_float_cpu
from hyperion.torch.utils import open_device
from hyperion.torch.helpers import OptimizerFactory as OF
from hyperion.torch.lr_schedulers import LRSchedulerFactory as LRSF
from hyperion.torch.narchs import ResNet2dEncoder as Encoder
from hyperion.torch.narchs import ResNet2dDecoder as Decoder
from hyperion.torch.models import VQVAE as VAE
from hyperion.torch.trainers import VQVAETrainer as Trainer
from hyperion.torch.data import SeqDataset as SD
from hyperion.torch.data import ClassWeightedSeqSampler as Sampler


def train_vae(data_rspec, train_list, val_list, 
              exp_path, epochs, num_gpus, log_interval, resume, num_workers, 
              grad_acc_steps, use_amp, **kwargs):

    set_float_cpu('float32')
    logging.info('initializing devices num_gpus={}'.format(num_gpus))
    device = open_device(num_gpus=num_gpus)

    sd_args = SD.filter_args(**kwargs)
    sampler_args = Sampler.filter_args(**kwargs)
    enc_args = Encoder.filter_args(prefix='enc', **kwargs)
    dec_args = Decoder.filter_args(prefix='dec', **kwargs)
    vae_args = VAE.filter_args(**kwargs)
    opt_args = OF.filter_args(prefix='opt', **kwargs)
    lrsch_args = LRSF.filter_args(prefix='lrsch', **kwargs)
    logging.info('seq dataset args={}'.format(sd_args))
    logging.info('sampler args={}'.format(sampler_args))
    logging.info('encoder args={}'.format(enc_args))
    logging.info('decoder args={}'.format(dec_args))
    logging.info('vae args={}'.format(vae_args))
    logging.info('optimizer args={}'.format(opt_args))
    logging.info('lr scheduler args={}'.format(lrsch_args))

    logging.info('init datasets')
    train_data = SD(data_rspec, train_list, 
                    return_class=False, **sd_args)
    val_data = SD(data_rspec, val_list, 
                  return_class=False, is_val=True, **sd_args)

    logging.info('init samplers')
    train_sampler = Sampler(train_data, **sampler_args)
    val_sampler = Sampler(val_data, **sampler_args)

    largs = {'num_workers': num_workers, 'pin_memory': True} if num_gpus > 0 else {}

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_sampler = train_sampler, **largs)

    test_loader = torch.utils.data.DataLoader(
        val_data, batch_sampler = val_sampler, **largs)

    encoder = Encoder(**enc_args)
    decoder = Decoder(**dec_args)
    model = VAE(encoder, decoder, **vae_args)
    logging.info(str(model))

    optimizer = OF.create(model.parameters(), **opt_args)
    lr_sch = LRSF.create(optimizer, **lrsch_args)
    metrics = { 'mse': nn.MSELoss(), 'L1': nn.L1Loss() }

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
        description='Train VQ-VAE with ResNet2d Encoder-Decoder')

    parser.add_argument('--data-rspec', required=True)
    parser.add_argument('--train-list', required=True)
    parser.add_argument('--val-list', required=True)

    SD.add_argparse_args(parser)
    Sampler.add_argparse_args(parser)

    parser.add_argument('--num-workers', type=int, default=5, 
                        help='num_workers of data loader')

    parser.add_argument('--grad-acc-steps', type=int, default=1, 
                        help='gradient accumulation batches before weight update')

    parser.add_argument('--epochs', type=int, default=200, 
                        help='number of epochs')

    Encoder.add_argparse_args(parser, prefix='enc')
    Decoder.add_argparse_args(parser, prefix='dec')
    VAE.add_argparse_args(parser)

    OF.add_argparse_args(parser, prefix='opt')
    LRSF.add_argparse_args(parser, prefix='lrsch')

    parser.add_argument('--num-gpus', type=int, default=1,
                        help='number of gpus, if 0 it uses cpu')
    parser.add_argument('--seed', type=int, default=1123581321, 
                        help='random seed')
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

    train_vae(**vars(args))

