#!/usr/bin/env python
"""
  Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
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

from hyperion.hyp_defs import config_logger
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.io import RandomAccessAudioReader as AR
from hyperion.utils import Utt2Info, TrialNdx, TrialKey, TrialScores
from hyperion.utils.list_utils import ismember
from hyperion.io import VADReaderFactory as VRF
#from hyperion.feats import MeanVarianceNorm as MVN2

from hyperion.torch.utils import open_device
from hyperion.torch.helpers import TorchModelLoader as TML
from hyperion.torch.layers import AudioFeatsFactory as AFF
from hyperion.torch.layers import MeanVarianceNorm as MVN
from hyperion.torch.utils.misc import l2_norm

def read_data(v_file, ndx_file, enroll_file, seg_part_idx, num_seg_parts):

    r = DRF.create(v_file)
    enroll = Utt2Info.load(enroll_file)
    try:
        ndx = TrialNdx.load(ndx_file)
    except:
        ndx = TrialKey.load(ndx_file).to_ndx()
        
    if num_seg_parts > 1:
        ndx = ndx.split(1, 1, seg_part_idx, num_seg_parts)

    x_e = r.read(enroll.key, squeeze=True)

    f, idx = ismember(ndx.model_set, enroll.info)
    
    assert np.all(f)
    x_e = x_e[idx]

    return ndx, x_e



def eval_cosine_scoring(v_file, ndx_file, enroll_file, test_wav_file,
                        mvn_no_norm_mean, mvn_norm_var, mvn_context,
                        vad_spec, vad_path_prefix, model_path, embed_layer,
                        score_file,
                        use_gpu, seg_part_idx, num_seg_parts, **kwargs):

    num_gpus = 1 if use_gpu else 0
    logging.info('initializing devices num_gpus={}'.format(num_gpus))
    device = open_device(num_gpus=num_gpus)

    feat_args = AFF.filter_args(**kwargs)
    logging.info('initializing feature extractor args={}'.format(feat_args))
    feat_extractor = AFF.create(**feat_args)
    feat_extractor.to(device)
    feat_extractor.eval()

    do_mvn = False
    if not mvn_no_norm_mean or mvn_norm_var:
        do_mvn = True

    if do_mvn:
        logging.info('initializing short-time mvn')
        # mvn = MVN2(
        #     norm_mean=(not mvn_no_norm_mean), norm_var=mvn_norm_var,
        #     left_context=mvn_context, right_context=mvn_context)
        mvn = MVN(
            norm_mean=(not mvn_no_norm_mean), norm_var=mvn_norm_var,
            left_context=mvn_context, right_context=mvn_context)
        mvn.to(device)


    logging.info('loading model {}'.format(model_path))
    model = TML.load(model_path)
    model.to(device)
    model.eval()

    logging.info('loading ndx and enrollment x-vectors')
    ndx, y_e = read_data(v_file, ndx_file, enroll_file, seg_part_idx, num_seg_parts)

    audio_reader = AR(test_wav_file)

    if vad_spec is not None:
        logging.info('opening VAD stream: %s' % (vad_spec))
        v_reader = VRF.create(vad_spec, path_prefix=vad_path_prefix, scp_sep=' ')

    scores = np.zeros((ndx.num_models, ndx.num_tests), dtype='float32')
    with torch.no_grad():
        for j in range(ndx.num_tests):
            t1 = time.time()
            logging.info('scoring test utt %s' % (ndx.seg_set[j]))
            s, fs = audio_reader.read([ndx.seg_set[j]])
            s = s[0]
            fs = fs[0]
            t2 = time.time()
            s = torch.as_tensor(s[None,:], dtype=torch.get_default_dtype()).to(device)
            x_t = feat_extractor(s)
            t3 = time.time()            

            if do_mvn:
                # x_t = x_t.cpu().numpy()[0]
                # x_t = mvn.normalize(x_t)
                # x_t = torch.as_tensor(x_t[None,:], dtype=torch.get_default_dtype()).to(device)
                x_t = mvn(x_t)
            t4 = time.time()            
            tot_frames = x_t.shape[1]
            if vad_spec is not None:
                vad = torch.as_tensor(
                    v_reader.read(
                        [ndx.seg_set[j]], num_frames=x_t.shape[1])[0].astype(
                            np.uint8, copy=False), dtype=torch.uint8).to(device)
                x_t = x_t[:,vad]
                logging.info('utt %s detected %d/%d (%.2f %%) speech frames' % (
                        ndx.seg_set[j], x_t.shape[1], tot_frames, x_t.shape[1]/tot_frames*100))

            t5 = time.time()            
            x_t = x_t.transpose(1,2).contiguous()
            y_t = model.extract_embed(x_t, embed_layer=embed_layer)
            y_t = l2_norm(y_t)
            t6 = time.time()                        

            for i in range(ndx.num_models):
                if ndx.trial_mask[i,j]:
                    y_e_i = torch.as_tensor(y_e[i], dtype=torch.get_default_dtype()).to(device)
                    y_e_i = l2_norm(y_e_i)
                    scores[i,j] = torch.sum(y_e_i * y_t, dim=-1)

            t7 = time.time()
            logging.info((
                    'utt %s total-time=%.3f read-time=%.3f feat-time=%.3f mvn-time=%.3f '
                    'vad-time=%.3f embed-time=%.3f trial-time=%.3f n_trials=%d '
                    'rt-factor=%.2f') % (
                        ndx.seg_set[j], t7-t1, t2-t1, t3-t2, t4-t3, 
                        t5-t4, t6-t5, t7-t6, np.sum(ndx.trial_mask[:,j]), 
                        x_t.shape[-1]*1e-2/(t7-t1)))

    if num_seg_parts > 1:
        score_file = '%s-%03d-%03d' % (score_file, 1, seg_part_idx)
    logging.info('saving scores to %s' % (score_file))
    s = TrialScores(ndx.model_set, ndx.seg_set, scores, score_mask=ndx.trial_mask)
    s.save_txt(score_file)





if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Eval cosine-scoring given enroll x-vector and test wave')

    parser.add_argument('--v-file', dest='v_file', required=True)
    parser.add_argument('--ndx-file', dest='ndx_file', default=None)
    parser.add_argument('--enroll-file', dest='enroll_file', required=True)
    parser.add_argument('--test-wav-file', required=True)

    AFF.add_argparse_args(parser)

    parser.add_argument('--mvn-no-norm-mean', 
                        default=False, action='store_true',
                        help='don\'t center the features')

    parser.add_argument('--mvn-norm-var', 
                        default=False, action='store_true',
                        help='normalize the variance of the features')
        
    parser.add_argument('--mvn-context', type=int,
                        default=300,
                        help='short-time mvn context in number of frames')

    parser.add_argument('--vad', dest='vad_spec', default=None)
    parser.add_argument('--vad-path-prefix', dest='vad_path_prefix', default=None,
                        help=('scp file_path prefix for vad'))

    parser.add_argument('--model-path', required=True)
    parser.add_argument('--embed-layer', type=int, default=None, 
                        help=('classifier layer to get the embedding from,' 
                              'if None the layer set in training phase is used'))

    parser.add_argument('--use-gpu', default=False, action='store_true',
                        help='extract xvectors in gpu')

    parser.add_argument('--seg-part-idx', default=1, type=int,
                        help=('test part index'))
    parser.add_argument('--num-seg-parts', default=1, type=int,
                        help=('number of parts in which we divide the test list '
                              'to run evaluation in parallel'))

    parser.add_argument('--score-file', dest='score_file', required=True)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1,
                        choices=[0, 1, 2, 3], type=int)
    
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_cosine_scoring(**vars(args))
