"""
Copyright 2026 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Union

import numpy as np

from ..feats.stft import strft
from .snr import project_target_scale_invariant


def compute_lsd(
    pred: np.ndarray,
    target: np.ndarray,
    axis: int = -1,
    frame_length: int = 512,
    frame_shift: int = 256,
    fft_length: Union[int, None] = None,
    eps: float = 1e-10,
    scale_invariant: bool = False,
) -> Union[float, np.ndarray]:
    """Computes Log-Spectral Distance (LSD) between predicted and target signals.

    For each frame, LSD is computed as:
    .. math::
       \\sqrt{\\frac{1}{K}\\sum_k
       (20\\log_{10}|\\hat{X}_k| - 20\\log_{10}|X_k|)^2}

    and then averaged across frames.

    Args:
      pred: Predicted/processed signal array.
      target: Reference (clean) signal array.
      axis: Time/sample axis.
      frame_length: STFT frame length.
      frame_shift: STFT frame shift.
      fft_length: FFT size. If None, uses ``frame_length``.
      eps: Floor value for spectral magnitude before ``log10``.
      scale_invariant: If True, applies SI-SNR style target projection
        (zero-mean and optimal scalar projection of target onto pred) before
        computing LSD.

    Returns:
      LSD (dB) as float (1D inputs) or an array with shape ``pred.shape``
      excluding ``axis``.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)

    if pred.shape != target.shape:
        raise ValueError(
            "pred and target must have the same shape, got "
            f"{pred.shape} and {target.shape}"
        )

    if pred.ndim == 0:
        raise ValueError("pred and target must have at least 1 dimension")

    if frame_length <= 0:
        raise ValueError(f"frame_length must be > 0, got {frame_length}")
    if frame_shift <= 0:
        raise ValueError(f"frame_shift must be > 0, got {frame_shift}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    if fft_length is None:
        fft_length = frame_length
    if fft_length <= 0:
        raise ValueError(f"fft_length must be > 0, got {fft_length}")

    axis = np.core.numeric.normalize_axis_index(axis, pred.ndim)

    pred_2d = np.moveaxis(pred, axis, -1).reshape(-1, pred.shape[axis])
    target_2d = np.moveaxis(target, axis, -1).reshape(-1, target.shape[axis])
    if scale_invariant:
        target_2d = project_target_scale_invariant(
            pred=pred_2d, target=target_2d, axis=-1, eps=1e-10
        )

    window = np.hanning(frame_length).astype(np.float64, copy=False)
    scores = np.empty((pred_2d.shape[0],), dtype=np.float64)

    for i in range(pred_2d.shape[0]):
        pred_i = np.ascontiguousarray(pred_2d[i], dtype=np.float64)
        target_i = np.ascontiguousarray(target_2d[i], dtype=np.float64)

        # Ensure at least one analysis frame.
        if pred_i.shape[0] < frame_length:
            pad = frame_length - pred_i.shape[0]
            pred_i = np.pad(pred_i, (0, pad), mode="constant")
            target_i = np.pad(target_i, (0, pad), mode="constant")

        pred_stft = strft(
            pred_i,
            frame_length=frame_length,
            frame_shift=frame_shift,
            fft_length=fft_length,
            window=window,
        )
        target_stft = strft(
            target_i,
            frame_length=frame_length,
            frame_shift=frame_shift,
            fft_length=fft_length,
            window=window,
        )

        pred_log = 20.0 * np.log10(np.maximum(np.abs(pred_stft), eps))
        target_log = 20.0 * np.log10(np.maximum(np.abs(target_stft), eps))
        lsd_frames = np.sqrt(np.mean((pred_log - target_log) ** 2, axis=-1))
        scores[i] = float(np.mean(lsd_frames))

    if pred.ndim == 1:
        return float(scores[0])

    out_shape = tuple(s for i, s in enumerate(pred.shape) if i != axis)
    return scores.reshape(out_shape)
