"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#from __future__ import absolute_import

import re
from ...feats.filter_banks import FilterBankFactory as FBF
from .audio_feats import *

FFT = 'fft'
SPEC = 'spec'
LOG_SPEC = 'log_spec'
LOG_FB = 'logfb'
MFCC = 'mfcc'

FEAT_TYPES = [FFT, SPEC, LOG_SPEC, LOG_FB, MFCC]

class AudioFeatsFactory(object):

    @staticmethod
    def create(audio_feat, fs=16000, frame_length=25, frame_shift=10, 
               fft_length=512,
               remove_dc_offset=True, preemph_coeff=0.97, 
               window_type='povey', use_fft_mag=False, dither=1, 
               fb_type='mel_kaldi',
               low_freq=20, high_freq=0, num_filters=23, norm_filters=False,
               num_ceps=13, snip_edges=True, cepstral_lifter=22,
               energy_floor=0, raw_energy=True, use_energy=True):

        if audio_feat == FFT:
            return Wav2FFT(                
                fs, frame_length, frame_shift, fft_length,
                remove_dc_offset=remove_dc_offset, preemph_coeff=preemph_coeff, 
                window_type=window_type, dither=dither, 
                snip_edges=snip_edges, 
                energy_floor=energy_floor, raw_energy=raw_energy, 
                use_energy=use_energy)

        if audio_feat == SPEC:
            return Wav2Spec(                
                fs, frame_length, frame_shift, fft_length,
                remove_dc_offset=remove_dc_offset, preemph_coeff=preemph_coeff, 
                window_type=window_type, use_fft_mag=use_fft_mag, dither=dither, 
                snip_edges=snip_edges, 
                energy_floor=energy_floor, raw_energy=raw_energy, 
                use_energy=use_energy)

        if audio_feat == LOG_SPEC:
            return Wav2LogSpec(                
                fs, frame_length, frame_shift, fft_length,
                remove_dc_offset=remove_dc_offset, preemph_coeff=preemph_coeff, 
                window_type=window_type, use_fft_mag=use_fft_mag, dither=dither, 
                snip_edges=snip_edges, 
                energy_floor=energy_floor, raw_energy=raw_energy, 
                use_energy=use_energy)

        if audio_feat == LOG_FB:
            return Wav2LogFilterBank(                
                fs, frame_length, frame_shift, fft_length,
                remove_dc_offset=remove_dc_offset, preemph_coeff=preemph_coeff, 
                window_type=window_type, use_fft_mag=use_fft_mag, dither=dither, 
                fb_type=fb_type,
                low_freq=low_freq, high_freq=high_freq, 
                num_filters=num_filters, norm_filters=norm_filters,
                snip_edges=snip_edges, 
                energy_floor=energy_floor, raw_energy=raw_energy, 
                use_energy=use_energy)

        if audio_feat == MFCC:
            return Wav2MFCC(
                fs, frame_length, frame_shift, fft_length,
                remove_dc_offset=remove_dc_offset, preemph_coeff=preemph_coeff, 
                window_type=window_type, use_fft_mag=use_fft_mag, dither=dither, 
                fb_type=fb_type,
                low_freq=low_freq, high_freq=high_freq, 
                num_filters=num_filters, norm_filters=norm_filters,
                num_ceps=num_ceps, snip_edges=snip_edges, 
                cepstral_lifter=cepstral_lifter,
                energy_floor=energy_floor, raw_energy=raw_energy, 
                use_energy=use_energy)


    @staticmethod
    def filter_args(prefix=None, **kwargs):
        """Filters MFCC args from arguments dictionary.
           
           Args:
             prefix: Options prefix.
             kwargs: Arguments dictionary.
           
           Returns:
             Dictionary with MFCC options.
        """
        if prefix is None:
            p = ''
        else:
            p = re.sub('-','_',prefix) + '_'

        valid_args = ('fs', 'frame_length', 'frame_shift',
                      'fft_length', 'remove_dc_offset', 'preemph_coeff',
                      'window_type', 'blackman_coeff', 'use_fft_mag', 'dither',
                      'fb_type', 'low_freq', 'high_freq', 'num_filters', 'norm_filters',
                      'num_ceps', 'snip_edges',
                      'energy_floor', 'raw_energy', 'use_energy', 'cepstral_lifter',
                      'audio_feat')
        
        d = dict((k, kwargs[p+k])
                 for k in valid_args if p+k in kwargs)
        return d


    @staticmethod
    def add_argparse_args(parser, prefix=None):
        """Adds MFCC options to parser.
           
           Args:
             parser: Arguments parser
             prefix: Options prefix.
        """

        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = re.sub('-','_', prefix) + '_'

        parser.add_argument(p1+'sample-frequency', dest=(p2+'fs'), 
                            default=16000, type=int,
                            help='Waveform data sample frequency (must match the waveform file, if specified there)')
        
        parser.add_argument(p1+'frame-length', dest=(p2+'frame_length'), type=int,
                            default=25,
                            help='Frame length in milliseconds')
        parser.add_argument(p1+'frame-shift', dest=(p2+'frame_shift'), type=int,
                            default=10,
                            help='Frame shift in milliseconds')
        parser.add_argument(p1+'fft-length', dest=(p2+'fft_length'), type=int,
                            default=512,
                            help='Length of FFT')

        parser.add_argument(p1+'remove-dc-offset', dest=(p2+'remove_dc_offset'),
                            default=True, type=str2bool,
                            help='Subtract mean from waveform on each frame')

        parser.add_argument(
            p1+'preemphasis-coeff', dest=(p2+'preemph_coeff'), type=float,
            default=0.97,
            help='Coefficient for use in signal preemphasis')
        
        parser.add_argument(
            p1+'window-type', dest=(p2+'window_type'), 
            default='povey',
            choices=['hamming', 'hanning', 'povey', 'rectangular', 'blackman'],
            help=('Type of window ("hamming"|"hanning"|"povey"|"rectangular"|"blackmann")'))

        
        parser.add_argument(p1+'use-fft-mag', dest=(p2+'use_fft_mag'),
                            default=False, action='store_true',
                            help='If true, it uses |X(f)|, if false, it uses |X(f)|^2')

        parser.add_argument(
            p1+'dither', dest=(p2+'dither'), type=float,
            default=1,
            help='Dithering constant (0.0 means no dither)')
        
        FBF.add_argparse_args(parser, prefix)

        parser.add_argument(p1+'num-ceps', dest=(p2+'num_ceps'), type=int,
                            default=13,
                            help='Number of cepstra in MFCC computation (including C0)')
        
        parser.add_argument(p1+'snip-edges', dest=(p2+'snip_edges'),
                            default=True, type=str2bool,
                            help='If true, end effects will be handled by outputting only frames that completely fit in the file, and the number of frames depends on the frame-length.  If false, the number of frames depends only on the frame-shift, and we reflect the data at the ends.')

        parser.add_argument(
            p1+'energy-floor', dest=(p2+'energy_floor'), type=float,
            default=0,
            help='Floor on energy (absolute, not relative) in MFCC computation')
        
        parser.add_argument(p1+'raw-energy', dest=(p2+'raw_energy'),
                            default=True, type=str2bool,
                            help='If true, compute energy before preemphasis and windowing')
        parser.add_argument(p1+'use-energy', dest=(p2+'use_energy'),
                            default=True, type=str2bool,
                            help='Use energy (not C0) in MFCC computation')
        
        parser.add_argument(
            p1+'cepstral-lifter', dest=(p2+'cepstral_lifter'), type=float,
            default=22,
            help='Constant that controls scaling of MFCCs')
        
        parser.add_argument(
            p1+'audio-feat', dest=(p2+'audio_feat'), 
            default='cepstrum',
            choices=['fft', 'spec', 'log_spec', 'logfb', 'mfcc' ],
            help='It can return intermediate result: fft, spec, log_spec, logfb, mfcc')

        

    

