"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

import numpy as np

from ..hyp_defs import float_cpu


def stft(x, frame_length, frame_shift, fft_length, window=None):

    if window is None:
        window = 1

    num_frames = int(np.floor((len(x) - frame_length + frame_shift) / frame_shift))
    X = np.zeros((num_frames, fft_length), dtype="complex64")
    j = 0
    for i in range(num_frames):
        X[i, :] = np.fft.fft(x[j : j + frame_length] * window, n=fft_length)
        j += frame_shift

    return X


def istft(X, frame_length, frame_shift, window=None):

    if window is None:
        window = np.ones((frame_length,), dtype=float_cpu())

    num_samples = (X.shape[0] - 1) * frame_shift + frame_length
    x_overlap = np.zeros((num_samples,), dtype="complex64")
    w_overlap = np.zeros((num_samples,), dtype=float_cpu())

    xx = np.fft.ifft(X, axis=-1)[:, :frame_length]
    j = 0
    for i in range(X.shape[0]):
        x_overlap[j : j + frame_length] += xx[i]
        w_overlap[j : j + frame_length] += window
        j += frame_shift

    w_overlap[w_overlap == 0] = 1
    iw = 1 / w_overlap
    # iw[w_overlap==0] = 0
    x = x_overlap * iw
    return x


def strft(x, frame_length, frame_shift, fft_length, window=None):

    if window is None:
        window = 1

    num_frames = int(np.floor((len(x) - frame_length + frame_shift) / frame_shift))
    X = np.zeros((num_frames, int(fft_length / 2 + 1)), dtype="complex64")
    j = 0
    for i in range(num_frames):
        X[i, :] = np.fft.rfft(x[j : j + frame_length] * window, n=fft_length)
        j += frame_shift

    return X


def istrft(X, frame_length, frame_shift, window=None):

    if window is None:
        window = np.ones((frame_length,), dtype=float_cpu())

    num_samples = (X.shape[0] - 1) * frame_shift + frame_length
    x_overlap = np.zeros((num_samples,), dtype=float_cpu())
    w_overlap = np.zeros((num_samples,), dtype=float_cpu())

    xx = np.fft.irfft(X, axis=-1)[:, :frame_length]
    j = 0
    for i in range(X.shape[0]):
        x_overlap[j : j + frame_length] += xx[i]
        w_overlap[j : j + frame_length] += window
        j += frame_shift

    w_overlap[w_overlap == 0] = 1
    iw = 1 / w_overlap
    # iw[w_overlap==0] = 0
    x = x_overlap * iw
    return x


def st_logE(x, frame_length, frame_shift):
    """Computes log-energy before preemphasis filter

    Args:
      x: wave signal

    Returns:
      Log-energy
    """

    num_frames = int(np.floor((len(x) - frame_length + frame_shift) / frame_shift))

    x2 = x ** 2
    e = np.zeros((num_frames,), dtype=float_cpu())
    j = 0
    for i in range(num_frames):
        e[i] = np.sum(x2[j : j + frame_length])
        j += frame_shift

    return np.log(e + 1e-15)
