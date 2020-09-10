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
from hyperion.torch.seq_embed import TDNNXVector as XVec
from hyperion.torch.trainers import XVectorTrainerFromWav as Trainer
from hyperion.torch.data import AudioDataset as AD
from hyperion.torch.data import ClassWeightedSeqSampler as Sampler
from hyperion.torch.metrics import CategoricalAccuracy

from hyperion.torch.layers import AudioFeatsFactory as AFF
from hyperion.torch.layers import MeanVarianceNorm as MVN

from apex import amp

class FeatExtractor(nn.Module):

    def __init__(self, feat_extractor, mvn=None):
        super().__init__()

        self.feat_extractor = feat_extractor
        self.mvn = mvn

    @amp.float_function
    def forward(self, x):
        #logging.info('x={}'.format(x))
        f = self.feat_extractor(x)
        #logging.info('f={}'.format(f))
        if self.mvn is not None:
            f = self.mvn(f)
        #logging.info('f={}'.format(f))
        f = f.transpose(1,2).contiguous()
        return f


def train_xvec(audio_path, train_list, val_list, 
               train_aug_cfg, val_aug_cfg,
               exp_path,
               epochs, num_gpus, log_interval, resume, num_workers, 
               mvn_no_norm_mean, mvn_norm_var, mvn_context,
               grad_acc_steps, use_amp, **kwargs):

    set_float_cpu('float32')
    logging.info('initializing devices num_gpus={}'.format(num_gpus))
    device = open_device(num_gpus=num_gpus)

    ad_args = AD.filter_args(**kwargs)
    sampler_args = Sampler.filter_args(**kwargs)
    feat_args = AFF.filter_args(prefix='feats', **kwargs)
    xvec_args = XVec.filter_args(**kwargs)
    opt_args = OF.filter_args(prefix='opt', **kwargs)
    lrsch_args = LRSF.filter_args(prefix='lrsch', **kwargs)
    logging.info('audio dataset args={}'.format(ad_args))
    logging.info('sampler args={}'.format(sampler_args))
    logging.info('feat args={}'.format(feat_args))
    #logging.info('mvn args={}'.format(mvn_args))
    logging.info('xvector network args={}'.format(xvec_args))
    logging.info('optimizer args={}'.format(opt_args))
    logging.info('lr scheduler args={}'.format(lrsch_args))

    logging.info('initializing feature extractor args={}'.format(feat_args))
    feat_extractor = AFF.create(**feat_args)
    do_mvn = False
    if not mvn_no_norm_mean or mvn_norm_var:
        do_mvn = True

    mvn = None
    if do_mvn:
        logging.info('initializing short-time mvn')
        mvn = MVN(
            norm_mean=(not mvn_no_norm_mean), norm_var=mvn_norm_var,
            left_context=mvn_context, right_context=mvn_context)

    feat_extractor = FeatExtractor(feat_extractor, mvn)

    logging.info('init datasets')
    train_data = AD(audio_path, train_list, aug_cfg=train_aug_cfg, **ad_args)
    val_data = AD(audio_path, val_list, aug_cfg=val_aug_cfg, is_val=True, **ad_args)

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

    logging.info('feat-extractor={}'.format(feat_extractor))
    logging.info('x-vector-model={}'.format(model))

    optimizer = OF.create(model.parameters(), **opt_args)
    lr_sch = LRSF.create(optimizer, **lrsch_args)
    metrics = { 'acc': CategoricalAccuracy() }
    
    trainer = Trainer(model, feat_extractor, optimizer, epochs, exp_path, 
                      grad_acc_steps=grad_acc_steps,
                      device=device, metrics=metrics, lr_scheduler=lr_sch,
                      data_parallel=(num_gpus>1), use_amp=use_amp, 
                      log_interval=log_interval)
    if resume:
        trainer.load_last_checkpoint()
    trainer.fit(train_loader, test_loader)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Train XVector with TDNN/E-TDNN/ResE-TDNN encoder from audio files')

    parser.add_argument('--audio-path', required=True)
    parser.add_argument('--train-list', required=True)
    parser.add_argument('--val-list', required=True)

    AD.add_argparse_args(parser)
    Sampler.add_argparse_args(parser)

    parser.add_argument('--train-aug-cfg', default=None)
    parser.add_argument('--val-aug-cfg', default=None)

    parser.add_argument('--num-workers', type=int, default=5, 
                        help='num_workers of data loader')
    parser.add_argument('--grad-acc-steps', type=int, default=1, 
                        help='gradient accumulation batches before weigth update')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='number of epochs')

    AFF.add_argparse_args(parser, prefix='feats')
    parser.add_argument('--mvn-no-norm-mean', 
                        default=False, action='store_true',
                        help='don\'t center the features')
    parser.add_argument('--mvn-norm-var', 
                        default=False, action='store_true',
                        help='normalize the variance of the features')
    parser.add_argument('--mvn-context', type=int,
                        default=150,
                        help='short-time mvn context in number of frames')

    XVec.add_argparse_args(parser)
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

    train_xvec(**vars(args))

