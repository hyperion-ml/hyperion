"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

 Miscellaneous functions
"""
from inspect import signature
from pathlib import Path
from typing import TypeVar

import numpy as np

PathLike = TypeVar("PathLike", str, Path, type(None))


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
    P_x = 10 * np.log10(np.mean(x**2, axis=axis))
    P_n = 10 * np.log10(np.mean(n**2, axis=axis))
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


def filter_func_args(func, kwargs, skip=set()):
    """Filters arguments expected by a function

    Args:
      func: function object
      kwargs: dictionary containing arguments
      skip: set with keys of func arguments to remove from kwargs

    Returns
      Dictionary with arguments expected by the target function
    """
    sig = signature(func)
    valid_args = sig.parameters.keys()
    skip.add("self")
    for param in skip:
        if param in kwargs:
            del kwargs[param]

    my_kwargs = filter_args(valid_args, kwargs)
    if "kwargs" in kwargs:
        my_kwargs.update(kwargs["kwargs"])

    args = sig.bind_partial(**my_kwargs).arguments
    return args


from tqdm import tqdm


def tqdm_urlretrieve_hook(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> from urllib.request import urlretrieve
    >>> with tqdm(...) as t:
    ...     reporthook = tqdm_urlretrieve_hook(t)
    ...     urlretrieve(..., reporthook=reporthook)
    Source: https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] or -1,
            remains unchanged.
        """
        if tsize not in (None, -1):
            t.total = tsize
            displayed = t.update((b - last_b[0]) * bsize)
            last_b[0] = b
            return displayed

    return update_to


def urlretrieve_progress(url, filename=None, data=None, desc=None):
    """
    Works exactly like urllib.request.urlretrieve, but attaches a tqdm hook to display
    a progress bar of the download.
    Use "desc" argument to display a user-readable string that informs what is being downloaded.
    Taken from lhotse: https://github.com/lhotse-speech/lhotse/blob/master/lhotse/utils.py
    """
    from urllib.request import urlretrieve

    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
        reporthook = tqdm_urlretrieve_hook(t)
        return urlretrieve(url=url, filename=filename, reporthook=reporthook, data=data)
