"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

 Miscellaneous functions
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np


def generate_data(g):
    while 1:
        yield g.get_next_batch()

def str2bool(s):
    """Convert string to bool for argparse """

    values={'true': True, 't': True, 'yes': True, 'y': True,
            'false': False, 'f': False, 'no': False, 'n': False}
    if s.lower() not in values:
        raise ValueError('Need bool; got %r' % s)
    return values[s.lower()]

def apply_gain_logx(x, AdB):
    """Applies A dB gain to log(x) """
    return x+AdB/(20.*np.log10(np.exp(1)))

def apply_gain_logx2(x, AdB):
    """Applies A dB gain to log(x^2) """
    return x+AdB/(10.*np.log10(np.exp(1)))

def apply_gain_x(x, AdB):
    """Applies A dB gain to x """
    return x*10**(AdB/20)

def apply_gain_x2(x, AdB):
    """Applies A dB gain to x^2 """
    return x*10**(AdB/10)

def apply_gain(x, feat_type, AdB):
    f_dict={ 'fft' : apply_gain_x,
             'logfft' : apply_gain_logx,
             'fb' : apply_gain_x,
             'fb2' : apply_gain_x2,
             'logfb' : apply_gain_logx,
             'logfb2' : apply_gain_logx2}
    f=f_dict[feat_type]
    return f(x,AdB)
             
def energy_vad(P):
    thr=np.max(P)-35
    return P>thr


