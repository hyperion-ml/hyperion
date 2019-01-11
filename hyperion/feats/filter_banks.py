"""
 Copyright 2018 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import logging

import numpy as np

from ..hyp_defs import float_cpu
from ..utils.misc import str2bool

class FilterBankFactory(object):

    @staticmethod
    def create(filter_bank_type, num_filters, fft_length, fs, low_freq, high_freq, norm_filters):

        if filter_bank_type == 'mel_kaldi':
            B = FilterBankFactory.make_mel_kaldi(num_filters, fft_length, fs, low_freq, high_freq)
        elif filter_bank_type == 'mel_etsi':
            B = FilterBankFactory.make_mel_etsi(num_filters, fft_length, fs, low_freq, high_freq)
        elif filter_bank_type == 'linear':
            B = FilterBankFactory.make_linear(num_filters, fft_length, fs, low_freq, high_freq)
        else:
            raise Exception('Invalid filter-bank type %s' % filter_bank_type)
        
        if norm_filters:
            B = B/np.sum(B, axis=0, keepdims=True)

        return B



    @staticmethod
    def lin2mel(x):
        return 1127.0 * np.log(1+x/700)


    @staticmethod
    def mel2lin(x):
        return 700 * (np.exp(x/1127.0) - 1)

        
    @staticmethod
    def make_mel_kaldi(num_filters, fft_length, fs, low_freq, high_freq):

        if high_freq == 0:
            high_freq = fs/2
            
        mel_low_freq = FilterBankFactory.lin2mel(low_freq)
        mel_high_freq = FilterBankFactory.lin2mel(high_freq)
        melfc = np.linspace(mel_low_freq, mel_high_freq, num_filters+2)
        mels = FilterBankFactory.lin2mel(np.linspace(0,fs,fft_length))

        B = np.zeros((int(fft_length/2+1), num_filters), dtype=float_cpu())
        for k in xrange(num_filters):
            left_mel = melfc[k]
            center_mel = melfc[k+1]
            right_mel = melfc[k+2]
            for j in xrange(int(fft_length/2)):
                mel_j = mels[j]
                if mel_j > left_mel and mel_j < right_mel:
                    if mel_j <= center_mel:
                        B[j,k] = (mel_j - left_mel)/(center_mel - left_mel)
                    else:
                        B[j,k] = (right_mel - mel_j)/(right_mel - center_mel)
                    
        return B


    @staticmethod
    def make_mel_etsi(num_filters, fft_length, fs, low_freq, high_freq):

        if high_freq == 0:
            high_freq = fs/2

        fs_2 = fs/2
        mel_low_freq = FilterBankFactory.lin2mel(low_freq)
        mel_high_freq = FilterBankFactory.lin2mel(high_freq)
        fc = FilterBankFactory.mel2lin(np.linspace(mel_low_freq, mel_high_freq, num_filters+2))
        cbin = np.round(fc/fs*fft_length).astype(int)

        B = np.zeros((int(fft_length/2+1), num_filters), dtype=float_cpu())
        for k in xrange(num_filters):
            for j in xrange(cbin[k], cbin[k+1]+1):
                B[j,k] = (j - cbin[k] + 1)/(cbin[k+1]-cbin[k]+1)
            for j in xrange(cbin[k+1]+1, cbin[k+2]+1):
                B[j,k] = (cbin[k+2] - j + 1)/(cbin[k+2]-cbin[k+1]+1)
                    
        return B


    @staticmethod
    def make_linear(num_filters, fft_length, fs, low_freq, high_freq):

        if high_freq == 0:
            high_freq = fs/2
        
        fs_2 = fs/2
        fc = np.linspace(low_freq, high_freq, num_filters+2)
        cbin = np.round(fc/fs*fft_length).astype(int)

        B = np.zeros((int(fft_length/2+1), num_filters), dtype=float_cpu())
        for k in xrange(num_filters):
            for j in xrange(cbin[k], cbin[k+1]+1):
                B[j,k] = (j - cbin[k] + 1)/(cbin[k+1]-cbin[k]+1)
            for j in xrange(cbin[k+1]+1, cbin[k+2]+1):
                B[j,k] = (cbin[k+2] - j + 1)/(cbin[k+2]-cbin[k+1]+1)
                    
        return B



    @staticmethod
    def add_argparse_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'


        parser.add_argument(
            p1+'fb-type', dest=(p2+'fb_type'), 
            default='mel_kaldi',
            choices=['mel_kaldi', 'mel_etsi', 'linear'],
            help='Filter-bank type: mel_kaldi, mel_etsi, linear')

        parser.add_argument(p1+'num-filters', dest=(p2+'num_filters'), type=int,
                            default=23,
                            help='Number of triangular mel-frequency bins')

        parser.add_argument(
            p1+'low-freq', dest=(p2+'low_freq'), type=float,
            default=20,
            help='Low cutoff frequency for mel bins')

        parser.add_argument(
            p1+'high-freq', dest=(p2+'high_freq'), type=float,
            default=0,
            help='High cutoff frequency for mel bins (if < 0, offset from Nyquist)')

        parser.add_argument(p1+'norm-filters', dest=(p2+'norm_filters'),
                            default=False, type=str2bool,
                            help='Normalize filters coeff to sum up to 1')

        
