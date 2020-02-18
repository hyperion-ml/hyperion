#!/usr/bin/env python
"""
 Copyright 2019 Jesus Villalba (Johns Hopkins University)
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

from hyperion.hyp_defs import config_logger, float_cpu
from hyperion.utils import Utt2Info
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import SequentialDataReaderFactory as DRF
from hyperion.io import VADReaderFactory as VRF
from hyperion.feats import MeanVarianceNorm as MVN

from hyperion.torch.utils import open_device
from hyperion.torch.helpers import TorchModelLoader as TML


def extract_xvectors(input_spec, output_spec, vad_spec, write_num_frames_spec,
                     scp_sep, path_prefix, vad_path_prefix, 
                     model_path, chunk_length, embed_layer, 
                     random_utt_length, min_utt_length, max_utt_length,
                     use_gpu, part_idx, num_parts, **kwargs):
    
    logging.info('initializing')
    mvn_args = MVN.filter_args(**kwargs)
    mvn = MVN(**mvn_args)
    do_mvn = True
    if mvn.norm_mean or mvn.norm_var:
        do_mvn = True

    if write_num_frames_spec is not None:
        keys = []
        info = []

    if random_utt_length:
        rng = np.random.RandomState(seed=1123581321+part_idx)
    
    num_gpus = 1 if use_gpu else 0
    logging.info('initializing devices num_gpus={}'.format(num_gpus))
    device = open_device(num_gpus=num_gpus)
    logging.info('loading model {}'.format(model_path))
    model = TML.load(model_path)
    model.to(device)
    model.eval()
        
    logging.info('opening output stream: %s' % (output_spec))
    with DWF.create(output_spec, scp_sep=scp_sep) as writer:

        logging.info('opening input stream: %s' % (output_spec))
        with DRF.create(input_spec, path_prefix=path_prefix, scp_sep=scp_sep,
                        part_idx=part_idx, num_parts=num_parts) as reader:
            if vad_spec is not None:
                logging.info('opening VAD stream: %s' % (vad_spec))
                v_reader = VRF.create(vad_spec, path_prefix=vad_path_prefix, scp_sep=scp_sep)
    
            while not reader.eof():
                t1 = time.time()
                key, data = reader.read(1)
                if len(key) == 0:
                    break
                t2 = time.time()
                logging.info('processing utt %s' % (key[0]))
                x = data[0]
                if do_mvn:
                    x = mvn.normalize(x)
                t3 = time.time()
                tot_frames = x.shape[0]
                if vad_spec is not None:
                    vad = v_reader.read(
                        key, num_frames=x.shape[0])[0].astype(
                            'bool', copy=False)
                    x = x[vad]

                logging.info('utt %s detected %d/%d (%.2f %%) speech frames' % (
                        key[0], x.shape[0], tot_frames, x.shape[0]/tot_frames*100))
                
                if random_utt_length:
                    utt_length = rng.randint(low=min_utt_length, high=max_utt_length+1)
                    if utt_length < x.shape[0]:
                        first_frame = rng.randint(low=0, high=x.shape[0]-utt_length)
                        x = x[first_frame:first_frame+utt_length]
                        logging.info('extract-random-utt %s of length=%d first-frame=%d' % (
                            key[0], x.shape[0], first_frame))

                t4 = time.time()
                if x.shape[0] == 0:
                    y = np.zeros((model.embed_dim,), dtype=float_cpu())
                else:
                    xx = torch.tensor(x.T[None,:], dtype=torch.get_default_dtype())
                    with torch.no_grad():
                        y = model.extract_embed(
                            xx, chunk_length=chunk_length, 
                            embed_layer=embed_layer, device=device).cpu().numpy()[0]

                t5 = time.time()
                writer.write(key, [y])
                if write_num_frames_spec is not None:
                    keys.append(key[0])
                    info.append(str(x.shape[0]))
                t6 = time.time()
                logging.info((
                    'utt %s total-time=%.3f read-time=%.3f mvn-time=%.3f '
                    'vad-time=%.3f embed-time=%.3f write-time=%.3f '
                    'rt-factor=%.2f') % (
                        key[0], t6-t1, t2-t1, t3-t2, t4-t3, 
                        t5-t4, t6-t5, x.shape[0]*1e-2/(t6-t1)))

    if write_num_frames_spec is not None:
        logging.info('writing num-frames to %s' % (write_num_frames_spec))
        u2nf = Utt2Info.create(keys, info)
        u2nf.save(write_num_frames_spec)
    
    
if __name__ == "__main__":
    
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Extract x-vectors with pytorch model')

    parser.add_argument('--input', dest='input_spec', required=True)
    parser.add_argument('--vad', dest='vad_spec', default=None)
    parser.add_argument('--write-num-frames', dest='write_num_frames_spec', default=None)
    parser.add_argument('--scp-sep', dest='scp_sep', default=' ',
                        help=('scp file field separator'))
    parser.add_argument('--path-prefix', dest='path_prefix', default=None,
                        help=('scp file_path prefix'))
    parser.add_argument('--vad-path-prefix', dest='vad_path_prefix', default=None,
                        help=('scp file_path prefix for vad'))

    MVN.add_argparse_args(parser)

    parser.add_argument('--model-path', required=True)
    parser.add_argument('--chunk-length', type=int, default=0, 
                        help=('number of frames used in each forward pass of the x-vector encoder,'
                              'if 0 the full utterance is used'))
    parser.add_argument('--embed-layer', type=int, default=None, 
                        help=('classifier layer to get the embedding from,' 
                              'if None the layer set in training phase is used'))

    parser.add_argument('--random-utt-length', default=False, action='store_true',
                        help='calculates x-vector from a random chunk of the utterance')
    parser.add_argument('--min-utt-length', type=int, default=500, 
                        help=('minimum utterance length when using random utt length'))
    parser.add_argument('--max-utt-length', type=int, default=12000, 
                        help=('maximum utterance length when using random utt length'))

    parser.add_argument('--output', dest='output_spec', required=True)
    parser.add_argument('--use-gpu', default=False, action='store_true',
                        help='extract xvectors in gpu')
    parser.add_argument('--part-idx', dest='part_idx', type=int, default=1,
                        help=('splits the list of files in num-parts and process part_idx'))
    parser.add_argument('--num-parts', dest='num_parts', type=int, default=1,
                        help=('splits the list of files in num-parts and process part_idx'))
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)

    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    extract_xvectors(**vars(args))
    
