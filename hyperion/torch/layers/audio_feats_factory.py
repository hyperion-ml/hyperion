"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

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
    def create(audio_feat, sample_frequency=16000, frame_length=25, frame_shift=10, 
               fft_length=512,
               remove_dc_offset=True, preemphasis_coeff=0.97, 
               window_type='povey', use_fft_mag=False, dither=1, 
               fb_type='mel_kaldi',
               low_freq=20, high_freq=0, num_filters=23, norm_filters=False,
               num_ceps=13, snip_edges=True, cepstral_lifter=22,
               energy_floor=0, raw_energy=True, use_energy=True):

        if audio_feat == FFT:
            return Wav2FFT(                
                sample_frequency, frame_length, frame_shift, fft_length,
                remove_dc_offset=remove_dc_offset, preemph_coeff=preemphasis_coeff, 
                window_type=window_type, dither=dither, 
                snip_edges=snip_edges, 
                energy_floor=energy_floor, raw_energy=raw_energy, 
                use_energy=use_energy)

        if audio_feat == SPEC:
            return Wav2Spec(                
                sample_frequency, frame_length, frame_shift, fft_length,
                remove_dc_offset=remove_dc_offset, preemph_coeff=preemphasis_coeff, 
                window_type=window_type, use_fft_mag=use_fft_mag, dither=dither, 
                snip_edges=snip_edges, 
                energy_floor=energy_floor, raw_energy=raw_energy, 
                use_energy=use_energy)

        if audio_feat == LOG_SPEC:
            return Wav2LogSpec(                
                sample_frequency, frame_length, frame_shift, fft_length,
                remove_dc_offset=remove_dc_offset, preemph_coeff=preemphasis_coeff, 
                window_type=window_type, use_fft_mag=use_fft_mag, dither=dither, 
                snip_edges=snip_edges, 
                energy_floor=energy_floor, raw_energy=raw_energy, 
                use_energy=use_energy)

        if audio_feat == LOG_FB:
            return Wav2LogFilterBank(                
                sample_frequency, frame_length, frame_shift, fft_length,
                remove_dc_offset=remove_dc_offset, preemph_coeff=preemphasis_coeff, 
                window_type=window_type, use_fft_mag=use_fft_mag, dither=dither, 
                fb_type=fb_type,
                low_freq=low_freq, high_freq=high_freq, 
                num_filters=num_filters, norm_filters=norm_filters,
                snip_edges=snip_edges, 
                energy_floor=energy_floor, raw_energy=raw_energy, 
                use_energy=use_energy)

        if audio_feat == MFCC:
            return Wav2MFCC(
                sample_frequency, frame_length, frame_shift, fft_length,
                remove_dc_offset=remove_dc_offset, preemph_coeff=preemphasis_coeff, 
                window_type=window_type, use_fft_mag=use_fft_mag, dither=dither, 
                fb_type=fb_type,
                low_freq=low_freq, high_freq=high_freq, 
                num_filters=num_filters, norm_filters=norm_filters,
                num_ceps=num_ceps, snip_edges=snip_edges, 
                cepstral_lifter=cepstral_lifter,
                energy_floor=energy_floor, raw_energy=raw_energy, 
                use_energy=use_energy)


    @staticmethod
    def filter_args(**kwargs):
        """Filters MFCC args from arguments dictionary.
           
           Args:
             kwargs: Arguments dictionary.
           
           Returns:
             Dictionary with MFCC options.
        """
        valid_args = ('sample_frequency', 'frame_length', 'frame_shift',
                      'fft_length', 'remove_dc_offset', 'preemphasis_coeff',
                      'window_type', 'blackman_coeff', 'use_fft_mag', 'dither',
                      'fb_type', 'low_freq', 'high_freq', 'num_filters', 'norm_filters',
                      'num_ceps', 'snip_edges',
                      'energy_floor', 'raw_energy', 'use_energy', 'cepstral_lifter',
                      'audio_feat')
        
        d = dict((k, kwargs[k])
                 for k in valid_args if k in kwargs)
        return d


    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds MFCC options to parser.
           
           Args:
             parser: Arguments parser
             prefix: Options prefix.
        """

        if prefix is None:
            p1 = '--'
        else:
            p1 = '--' + prefix + '.'

        parser.add_argument(
            p1+'sample-frequency', 
            default=16000, type=int,
            help=('Waveform data sample frequency (must match the waveform file, '
                  'if specified there)'))
        
        parser.add_argument(
            p1+'frame-length', type=int,
            default=25,
            help='Frame length in milliseconds')
        parser.add_argument(
            p1+'frame-shift', type=int,
            default=10,
            help='Frame shift in milliseconds')
        parser.add_argument(
            p1+'fft-length', type=int,
            default=512,
            help='Length of FFT')

        parser.add_argument(
            p1+'remove-dc-offset', 
            default=True, type=str2bool,
            help='Subtract mean from waveform on each frame')

        parser.add_argument(
            p1+'preemphasis-coeff', type=float,
            default=0.97,
            help='Coefficient for use in signal preemphasis')
        
        parser.add_argument(
            p1+'window-type', 
            default='povey',
            choices=['hamming', 'hanning', 'povey', 'rectangular', 'blackman'],
            help=('Type of window ("hamming"|"hanning"|"povey"|'
                  '"rectangular"|"blackmann")'))

        
        parser.add_argument(
            p1+'use-fft-mag', 
            default=False, action='store_true',
            help='If true, it uses |X(f)|, if false, it uses |X(f)|^2')

        parser.add_argument(
            p1+'dither', type=float,
            default=1,
            help='Dithering constant (0.0 means no dither)')
        
        FBF.add_class_args(parser, prefix)

        parser.add_argument(
            p1+'num-ceps', type=int,
            default=13,
            help='Number of cepstra in MFCC computation (including C0)')
        
        parser.add_argument(
            p1+'snip-edges', 
            default=True, type=str2bool,
            help=('If true, end effects will be handled by outputting only '
                  'frames that completely fit in the file, and the number of '
                  'frames depends on the frame-length.  If false, the number '
                  'of frames depends only on the frame-shift, '
                  'and we reflect the data at the ends.'))

        parser.add_argument(
            p1+'energy-floor', type=float,
            default=0,
            help='Floor on energy (absolute, not relative) in MFCC computation')
        
        parser.add_argument(
            p1+'raw-energy', 
            default=True, type=str2bool,
            help='If true, compute energy before preemphasis and windowing')
        parser.add_argument(
            p1+'use-energy', 
            default=True, type=str2bool,
            help='Use energy (not C0) in MFCC computation')
        
        parser.add_argument(
            p1+'cepstral-lifter', type=float,
            default=22,
            help='Constant that controls scaling of MFCCs')
        
        parser.add_argument(
            p1+'audio-feat', 
            default='cepstrum',
            choices=['fft', 'spec', 'log_spec', 'logfb', 'mfcc' ],
            help=('It can return intermediate result: fft, spec, log_spec, '
                  'logfb, mfcc'))
        
    add_argparse_args = add_class_args
    

