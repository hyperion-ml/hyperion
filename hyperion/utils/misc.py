"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

 Miscellaneous functions
"""

import numpy as np


def generate_data(g):
    while 1:
        yield g.get_next_batch()


def str2bool(s):
    """Convert string to bool for argparse"""
    if isinstance(s, bool):
        return s

    values = {
        "true": True,
        "t": True,
        "yes": True,
        "y": True,
        "false": False,
        "f": False,
        "no": False,
        "n": False,
    }
    if s.lower() not in values:
        raise ValueError("Need bool; got %r" % s)
    return values[s.lower()]


def apply_gain_logx(x, AdB):
    """Applies A dB gain to log(x)"""
    return x + AdB / (20.0 * np.log10(np.exp(1)))


def apply_gain_logx2(x, AdB):
    """Applies A dB gain to log(x^2)"""
    return x + AdB / (10.0 * np.log10(np.exp(1)))


def apply_gain_x(x, AdB):
    """Applies A dB gain to x"""
    return x * 10 ** (AdB / 20)


def apply_gain_x2(x, AdB):
    """Applies A dB gain to x^2"""
    return x * 10 ** (AdB / 10)


def apply_gain(x, feat_type, AdB):
    f_dict = {
        "fft": apply_gain_x,
        "logfft": apply_gain_logx,
        "fb": apply_gain_x,
        "fb2": apply_gain_x2,
        "logfb": apply_gain_logx,
        "logfb2": apply_gain_logx2,
    }
    f = f_dict[feat_type]
    return f(x, AdB)


def energy_vad(P):
    thr = np.max(P) - 35
    return P > thr


def compute_snr(x, n, axis=-1):

    P_x = 10 * np.log10(np.mean(x ** 2, axis=axis))
    P_n = 10 * np.log10(np.mean(n ** 2, axis=axis))
    return P_x - P_n


def filter_args(valid_args, kwargs):
    """Filters arguments from a dictionary

    Args:
      valid_args: list/tuple of valid arguments
      kwargs: dictionary containing program config arguments
    Returns
      Dictionary with only valid_args keys if they exists
    """
    return dict((k, kwargs[k]) for k in valid_args if k in kwargs)
