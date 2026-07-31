"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional

import numpy as np

from ...hyp_defs import float_cpu


def stft(
    x: np.ndarray,
    frame_length: int,
    frame_shift: int,
    fft_length: int,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Short-time Fourier Transform for real or complex signals.

    Args:
      x: input signal (num_samples,).
      frame_length: frame length.
      frame_shift: frame shift.
      fft_length: length of the FFT.
      window: window function as numpy array (frame_length,)

    Returns:
      Fourier transform (num_frames, fft_length)
    """
    if x.ndim != 1:
        raise ValueError(f"x must be a 1D signal, got shape={x.shape}")
    if frame_length <= 0:
        raise ValueError(f"frame_length must be > 0, got {frame_length!r}")
    if frame_shift <= 0:
        raise ValueError(f"frame_shift must be > 0, got {frame_shift!r}")
    if fft_length <= 0:
        raise ValueError(f"fft_length must be > 0, got {fft_length!r}")

    if window is None:
        window = np.ones((frame_length,), dtype=float_cpu())
    elif not isinstance(window, np.ndarray):
        raise TypeError(f"window must be a numpy array, got {type(window)!r}")
    elif window.shape[0] != frame_length:
        raise ValueError(
            f"window length ({window.shape[0]}) must match frame_length ({frame_length})"
        )

    num_frames = int(np.floor((len(x) - frame_length + frame_shift) / frame_shift))
    num_frames = max(0, num_frames)
    X = np.zeros((num_frames, fft_length), dtype="complex64")
    j = 0
    for i in range(num_frames):
        X[i, :] = np.fft.fft(x[j : j + frame_length] * window, n=fft_length)
        j += frame_shift

    return X


def istft(
    X: np.ndarray,
    frame_length: int,
    frame_shift: int,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Short-time Fourier Transform for real or complex signals.

    Args:
      X: input FFT (num_frames, fft_length).
      frame_length: frame length.
      frame_shift: frame shift.
      window: window function as numpy array (frame_length,)

    Returns:
      Reconstructed signal ((num_frames - 1) * frame_shift + frame_length,).
    """
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D matrix, got shape={X.shape}")
    if frame_length <= 0:
        raise ValueError(f"frame_length must be > 0, got {frame_length!r}")
    if frame_shift <= 0:
        raise ValueError(f"frame_shift must be > 0, got {frame_shift!r}")
    if window is None:
        window = np.ones((frame_length,), dtype=float_cpu())
    elif not isinstance(window, np.ndarray):
        raise TypeError(f"window must be a numpy array, got {type(window)!r}")
    elif window.shape[0] != frame_length:
        raise ValueError(
            f"window length ({window.shape[0]}) must match frame_length ({frame_length})"
        )

    if X.shape[0] == 0:
        return np.zeros((0,), dtype="complex64")

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


def strft(
    x: np.ndarray,
    frame_length: int,
    frame_shift: int,
    fft_length: int,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Short-time Fourier Transform for real signals.

    Args:
      x: input signal (num_samples,).
      frame_length: frame length.
      frame_shift: frame shift.
      fft_length: length of the FFT.
      window: window function as numpy array (frame_length,)

    Returns:
      Fourier transform (num_frames, fft_length/2+1)
    """
    if x.ndim != 1:
        raise ValueError(f"x must be a 1D signal, got shape={x.shape}")
    if frame_length <= 0:
        raise ValueError(f"frame_length must be > 0, got {frame_length!r}")
    if frame_shift <= 0:
        raise ValueError(f"frame_shift must be > 0, got {frame_shift!r}")
    if fft_length <= 0:
        raise ValueError(f"fft_length must be > 0, got {fft_length!r}")

    if window is None:
        window = np.ones((frame_length,), dtype=float_cpu())
    elif not isinstance(window, np.ndarray):
        raise TypeError(f"window must be a numpy array, got {type(window)!r}")
    elif window.shape[0] != frame_length:
        raise ValueError(
            f"window length ({window.shape[0]}) must match frame_length ({frame_length})"
        )

    num_frames = int(np.floor((len(x) - frame_length + frame_shift) / frame_shift))
    num_frames = max(0, num_frames)
    X = np.zeros((num_frames, int(fft_length / 2 + 1)), dtype="complex64")
    j = 0
    for i in range(num_frames):
        X[i, :] = np.fft.rfft(x[j : j + frame_length] * window, n=fft_length)
        j += frame_shift

    return X


def istrft(
    X: np.ndarray,
    frame_length: int,
    frame_shift: int,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Short-time Fourier Transform for real signals.

    Args:
      X: input FFT (num_frames, fft_length/2+1).
      frame_length: frame length.
      frame_shift: frame shift.
      window: window function as numpy array (frame_length,)

    Returns:
      Reconstructed signal ((num_frames - 1) * frame_shift + frame_length,).
    """
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D matrix, got shape={X.shape}")
    if frame_length <= 0:
        raise ValueError(f"frame_length must be > 0, got {frame_length!r}")
    if frame_shift <= 0:
        raise ValueError(f"frame_shift must be > 0, got {frame_shift!r}")
    if window is None:
        window = np.ones((frame_length,), dtype=float_cpu())
    elif not isinstance(window, np.ndarray):
        raise TypeError(f"window must be a numpy array, got {type(window)!r}")
    elif window.shape[0] != frame_length:
        raise ValueError(
            f"window length ({window.shape[0]}) must match frame_length ({frame_length})"
        )

    if X.shape[0] == 0:
        return np.zeros((0,), dtype=float_cpu())

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


def st_logE(x: np.ndarray, frame_length: int, frame_shift: int) -> np.ndarray:
    """Computes log-energy before preemphasis filter

    Args:
      x: wave signal

    Returns:
      Log-energy
    """
    if x.ndim != 1:
        raise ValueError(f"x must be a 1D signal, got shape={x.shape}")
    if frame_length <= 0:
        raise ValueError(f"frame_length must be > 0, got {frame_length!r}")
    if frame_shift <= 0:
        raise ValueError(f"frame_shift must be > 0, got {frame_shift!r}")

    num_frames = int(np.floor((len(x) - frame_length + frame_shift) / frame_shift))
    num_frames = max(0, num_frames)

    x2 = x**2
    e = np.zeros((num_frames,), dtype=float_cpu())
    j = 0
    for i in range(num_frames):
        e[i] = np.sum(x2[j : j + frame_length])
        j += frame_shift

    return np.log(e + 1e-15)
