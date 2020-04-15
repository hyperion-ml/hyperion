#!/usr/bin/env python
"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""
from __future__ import absolute_import

import os

import numpy as np
import pandas as pd

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.metrics.verification_evaluator import VerificationAdvAttackEvaluator as Eval

filenames = ['voxceleb1_attack_tar_snr_results.csv',
             'voxceleb1_attack_non_snr_results.csv',
             'voxceleb1_attack_tar_linf_results.csv',
             'voxceleb1_attack_non_linf_results.csv']

output_dir='exp/figs/resnet34_1/'
base_res_dir = 'exp/scores/resnet34_zir_e256_arc0.3_do0_adam_lr0.05_b512.v2'

def plot_figs1(res_dirs1, legends, title_base, fig_base, fmt=['b','r','g','m','c','y']): 
    df = []
    for i in range(len(res_dirs1)):
        file_path='%s/%s/%s' %(base_res_dir, res_dirs1[i], filenames[0])
        df_i = pd.read_csv(file_path, index_col=0)
        df.append(df_i)

    fig_file = output_dir + fig_base + '_tar_snr'
    Eval.plot_dcf_eer_vs_stat(df, 'snr', fig_file, clean_ref=0, 
                              xlabel='SNR(dB)', higher_better=True,
                              legends= legends, fmt=fmt,
                              title=title_base + ' attacks on target trials')
    df = []
    for i in range(len(res_dirs1)):
        file_path='%s/%s/%s' %(base_res_dir, res_dirs1[i], filenames[1])
        df_i = pd.read_csv(file_path,index_col=0)
        df.append(df_i)

    fig_file = output_dir + fig_base + '_non_snr'
    Eval.plot_dcf_eer_vs_stat(df, 'snr', fig_file, clean_ref=0, 
                              xlabel='SNR(dB)', higher_better=True,
                              legends=legends, fmt=fmt,
                              title=title_base + ' attacks on non-target trials')


    df = []
    for i in range(len(res_dirs1)):
        file_path='%s/%s/%s' %(base_res_dir, res_dirs1[i], filenames[2])
        df_i = pd.read_csv(file_path,index_col=0)
        df.append(df_i)

    fig_file = output_dir + fig_base + '_tar_linf'
    Eval.plot_dcf_eer_vs_stat(df, 'n_linf', fig_file, clean_ref=0, 
                              xlabel=r'$L_{\infty}$', log_x=True,
                              legends=legends, fmt=fmt,
                              title=title_base + ' attacks on target trials')
    df = []
    for i in range(len(res_dirs1)):
        file_path='%s/%s/%s' %(base_res_dir, res_dirs1[i], filenames[3])
        df_i = pd.read_csv(file_path, index_col=0)
        df.append(df_i)

    fig_file = output_dir + fig_base + '_non_linf'
    Eval.plot_dcf_eer_vs_stat(df, 'n_linf', fig_file, clean_ref=0, 
                              xlabel=r'$L_{\infty}$', log_x=True,
                              legends=legends, fmt=fmt,
                              title=title_base + ' attacks on non-target trials')


if __name__ == "__main__":


    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    res_dirs1 = ['cosine_fgsm_eall', 'cosine_randfgsm_eall', 'cosine_iterfgsm_eall']
    legends = ['FGSM', 'Rand-FGSM', 'Iter-FGSM']
    plot_figs1(res_dirs1, legends, 'FGSM', 'fgsm')

    res_dirs1 = ['cosine_cwl2_conf0', 'cosine_cwl2_conf1',  'cosine_cwlinf_conf0', 'cosine_cwlinf_conf1'] 
    legends = ['CW-L2 conf=0', 'CW-L2 conf=1', 'CW-Linf conf=0', 'CW-Linf conf=1']
    plot_figs1(res_dirs1, legends, 'Carlini-Wagner', 'cw')

    ###########################

    res_dirs0 = ['cosine_fgsm_eall', 'cosine_randfgsm_eall', 'cosine_iterfgsm_eall', 
                 'cosine_cwl2_conf0', 'cosine_cwl2_conf1',  'cosine_cwlinf_conf0', 'cosine_cwlinf_conf1'] 
    names = ['FGSM', 'Rand-FGSM', 'Iter-FGSM', 
              'CW-L2 conf=0', 'CW-L2 conf=1', 'CW-Linf conf=0', 'CW-Linf conf=1']
    fig_names = ['fgsm', 'randfgsm', 'iterfgsm', 'cwl2_conf0', 'cwl2_conf1', 'cwlinf_conf0', 'cwl2_conf1']
    legends = ['ResNet34 (white-box)', 'ThinResNet34', 'ResETDNN']
    fmt=['--b','r','g','m','c','y']
    for i in range(len(names)):
        res_dirs1 = [res_dirs0[i], 'transfer.lresnet34_zir_e256_arc0.3_do0_adam_lr0.05_b512.v2/' + res_dirs0[i],
                     'transfer.resetdnn_nl5ld512_e256_arcs30m0.3_do0.1_adam_lr0.05_b512_amp.v2/' + res_dirs0[i]]
        plot_figs1(res_dirs1, legends, names[i] + ' black-box', fig_names[i] + '_bbox', fmt=fmt)
    

