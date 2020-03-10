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
import torch.nn as nn

from hyperion.hyp_defs import config_logger
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.io import RandomAccessAudioReader as AR
from hyperion.io import AudioWriter as AW
from hyperion.utils import Utt2Info, TrialNdx, TrialKey, TrialScores
from hyperion.utils.list_utils import ismember
from hyperion.io import VADReaderFactory as VRF

from hyperion.torch.utils import open_device
from hyperion.torch.helpers import TorchModelLoader as TML
from hyperion.torch.layers import AudioFeatsFactory as AFF
from hyperion.torch.layers import MeanVarianceNorm as MVN
from hyperion.torch.utils.misc import l2_norm, compute_snr

from hyperion.torch.adv_attacks import AttackFactory

def read_data(v_file, key_file, enroll_file, seg_part_idx, num_seg_parts):

    r = DRF.create(v_file)
    enroll = Utt2Info.load(enroll_file)
    key = TrialKey.load(key_file)
        
    if num_seg_parts > 1:
        key = key.split(1, 1, seg_part_idx, num_seg_parts)

    x_e = r.read(enroll.key, squeeze=True)

    f, idx = ismember(key.model_set, enroll.info)
    
    assert np.all(f)
    x_e = x_e[idx]

    return key, x_e



class MyModel(nn.Module):

    def __init__(self, feat_extractor, xvector_model, mvn=None, embed_layer=None):
        super(MyModel, self).__init__()
        self.feat_extractor = feat_extractor
        self.xvector_model = xvector_model
        self.mvn = mvn
        self.x_e = None
        self.vad_t = None
        self.embed_layer = embed_layer


    def forward(self, s_t):
        f_t = s_t
        f_t = self.feat_extractor(s_t)
        if self.mvn is not None:
            f_t = self.mvn(f_t)

        if self.vad_t is not None:
            n_vad_frames = len(self.vad_t)
            n_feat_frames = f_t.shape[1]
            if n_vad_frames > n_feat_frames:
                self.vad_t = self.vad_t[:n_feat_frames]
            elif n_vad_frames < n_feat_frames:
                f_t = f_t[:,:n_vad_frames]

            f_t = f_t[:,self.vad_t]

        f_t = f_t.transpose(1,2).contiguous()
        x_t = self.xvector_model.extract_embed(f_t, embed_layer=self.embed_layer)
        x_t = l2_norm(x_t)
        x_e = l2_norm(self.x_e)
        score = torch.sum(x_e * x_t, dim=-1)
        return score


def eval_cosine_scoring(v_file, key_file, enroll_file, test_wav_file,
                        mvn_no_norm_mean, mvn_norm_var, mvn_context,
                        vad_spec, vad_path_prefix, model_path, embed_layer,
                        score_file, snr_file,
                        save_adv_wav, save_adv_wav_tar_thr, save_adv_wav_non_thr, save_adv_wav_path,
                        use_gpu, seg_part_idx, num_seg_parts, 
                        **kwargs):

    num_gpus = 1 if use_gpu else 0
    logging.info('initializing devices num_gpus={}'.format(num_gpus))
    device = open_device(num_gpus=num_gpus)

    feat_args = AFF.filter_args(**kwargs)
    logging.info('initializing feature extractor args={}'.format(feat_args))
    feat_extractor = AFF.create(**feat_args)

    do_mvn = False
    if not mvn_no_norm_mean or mvn_norm_var:
        do_mvn = True

    if do_mvn:
        logging.info('initializing short-time mvn')
        mvn = MVN(
            norm_mean=(not mvn_no_norm_mean), norm_var=mvn_norm_var,
            left_context=mvn_context, right_context=mvn_context)


    logging.info('loading model {}'.format(model_path))
    xvector_model = TML.load(model_path)
    xvector_model.freeze()

    model = MyModel(feat_extractor, xvector_model, mvn, embed_layer)
    model.to(device)
    model.eval()

    tar = torch.as_tensor([1], dtype=torch.float).to(device)
    non = torch.as_tensor([0], dtype=torch.float).to(device)

    logging.info('loading key and enrollment x-vectors')
    key, x_e = read_data(v_file, key_file, enroll_file, seg_part_idx, num_seg_parts)
    x_e = torch.as_tensor(x_e, dtype=torch.get_default_dtype())

    audio_args = AR.filter_args(**kwargs)
    audio_reader = AR(test_wav_file)
    wav_scale = audio_reader.scale

    if save_adv_wav:
        tar_audio_writer = AW(save_adv_wav_path + '/tar2non')
        non_audio_writer = AW(save_adv_wav_path + '/non2tar')

    attack_args = AttackFactory.filter_args(**kwargs)
    attack_type = attack_args['attack_type']
    del attack_args['attack_type']
    attack_args['attack_eps'] *= wav_scale
    attack_args['attack_alpha'] *= wav_scale
    attack = AttackFactory.create(
        attack_type, model, loss=nn.functional.binary_cross_entropy_with_logits, 
        range_min=-wav_scale, range_max=wav_scale, **attack_args)

    if vad_spec is not None:
        logging.info('opening VAD stream: %s' % (vad_spec))
        v_reader = VRF.create(vad_spec, path_prefix=vad_path_prefix, scp_sep=' ')

    scores = np.zeros((key.num_models, key.num_tests), dtype='float32')
    snr = np.zeros((key.num_models, key.num_tests), dtype='float32')
    for j in range(key.num_tests):
        t1 = time.time()
        logging.info('scoring test utt %s' % (key.seg_set[j]))
        s, fs = audio_reader.read([key.seg_set[j]])
        s = s[0]
        fs = fs[0]

        s = torch.as_tensor(s[None,:], dtype=torch.get_default_dtype()).to(device)
        
        if vad_spec is not None:
            vad = v_reader.read([key.seg_set[j]])[0]
            tot_frames = len(vad)
            speech_frames = np.sum(vad)
            vad = torch.as_tensor(vad.astype(np.uint8, copy=False), dtype=torch.uint8).to(device)
            model.vad_t = vad
            logging.info('utt %s detected %d/%d (%.2f %%) speech frames' % (
                key.seg_set[j], speech_frames, tot_frames, speech_frames/tot_frames*100))

        t2 = time.time()

        trial_time = 0
        num_trials = 0
        for i in range(key.num_models):
            if key.tar[i,j] or key.non[i,j]:
                t3 = time.time()
                model.x_e = x_e[i].to(device)
                if key.tar[i,j]:
                    if attack.targeted:
                        t = non
                    else:
                        t = tar
                else:
                    if attack.targeted:
                        t = tar
                    else:
                        t = non

                s_adv = attack.generate(s, t)
                with torch.no_grad():
                    scores[i,j] = model(s_adv)

                t4 = time.time()
                trial_time += (t4 - t3)
                num_trials += 1

                s_adv = s_adv.detach()
                snr[i,j] = compute_snr(s, s_adv - s)
                logging.info('min-max %f %f %f %f' % (torch.min(s), torch.max(s), torch.min(s_adv-s), torch.max(s_adv-s)))
                if save_adv_wav:
                    s_adv = s_adv.cpu().numpy()[0]
                    trial_name = '%s-%s' % (key.model_set[i], key.seg_set[j])
                    if key.tar[i,j] and scores[i,j] < save_adv_wav_non_thr:
                        tar_audio_writer.write(trial_name, s_adv, fs)
                    elif key.non[i,j] and scores[i,j] > save_adv_wav_tar_thr:
                        non_audio_writer.write(trial_name, s_adv, fs)

        trial_time /= num_trials
        t7 = time.time()
        logging.info((
            'utt %s total-time=%.3f read-time=%.3f trial-time=%.3f n_trials=%d '
            'rt-factor=%.2f') % (
                key.seg_set[j], t7-t1, t2-t1, trial_time, num_trials,
                num_trials*len(s)/fs/(t7-t1)))
        
    if num_seg_parts > 1:
        score_file = '%s-%03d-%03d' % (score_file, 1, seg_part_idx)
        snr_file = '%s-%03d-%03d' % (snr_file, 1, seg_part_idx)
    logging.info('saving scores to %s' % (score_file))
    s = TrialScores(key.model_set, key.seg_set, scores, score_mask=np.logical_or(key.tar,key.non))
    s.save_txt(score_file)

    logging.info('saving snr to %s' % (snr_file))
    s.scores = snr
    s.save_txt(snr_file)





if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Eval cosine-scoring given enroll x-vector and test wave')

    parser.add_argument('--v-file', dest='v_file', required=True)
    parser.add_argument('--key-file', dest='key_file', default=None)
    parser.add_argument('--enroll-file', dest='enroll_file', required=True)
    parser.add_argument('--test-wav-file', required=True)

    AR.add_argparse_args(parser)
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

    AttackFactory.add_argparse_args(parser)

    parser.add_argument('--seg-part-idx', default=1, type=int,
                        help=('test part index'))
    parser.add_argument('--num-seg-parts', default=1, type=int,
                        help=('number of parts in which we divide the test list '
                              'to run evaluation in parallel'))

    parser.add_argument('--score-file', dest='score_file', required=True)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1,
                        choices=[0, 1, 2, 3], type=int)

    parser.add_argument('--save-adv-wav', 
                        default=False, action='store_true',
                        help='save adversarial signals to disk')

    parser.add_argument('--save-adv-wav-path', default=None, 
                        help='output path of adv signals')

    parser.add_argument('--save-adv-wav-tar-thr', 
                        default=0.75, type=float,
                        help='min score to save signal from attack that makes non-tar into tar')

    parser.add_argument('--save-adv-wav-non-thr', 
                        default=-0.75, type=float,
                        help='max score to save signal from attack that makes tar into non-tar')

    parser.add_argument('--snr-file', default=None, 
                        help='output path of to save snr of signals')
    
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_cosine_scoring(**vars(args))
