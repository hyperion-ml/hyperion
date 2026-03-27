"""
Copyright 2026 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Union

import numpy as np

from .snr import project_target_scale_invariant

def compute_pesq(
    pred: np.ndarray,
    target: np.ndarray,
    fs: int,
    mode: Union[str, None] = None,
    axis: int = -1,
    num_workers: int = 1,
    scale_invariant: bool = False,
) -> Union[float, np.ndarray]:
    """Computes PESQ score between enhanced and reference signals.

    Args:
      pred: Predicted/processed signal array.
      target: Reference (clean) signal array.
      fs: Sampling frequency in Hz.
      mode: PESQ mode. Values are ``"wb"`` (wideband) or ``"nb"``.
        If None, it is derived from ``fs`` as:
        - ``fs=8000`` -> ``"nb"``
        - ``fs=16000`` -> ``"wb"``
      axis: Time/sample axis. PESQ is computed independently for each item in
        the remaining dimensions.
      num_workers: Number of worker processes used by ``pesq_batch`` when
        processing batched inputs.
      scale_invariant: If True, applies SI-SNR style target projection
        (zero-mean and optimal scalar projection of target onto pred) before
        computing PESQ.

    Returns:
      PESQ score as float (1D inputs) or an array with shape ``pred.shape``
      excluding ``axis``.
    """
    try:
        from pesq import pesq as pesq_fn
        from pesq import pesq_batch as pesq_batch_fn
    except Exception as e:
        raise ImportError(
            "compute_pesq requires the 'pesq' package. Install it with "
            "'pip install pesq'."
        ) from e

    pred = np.asarray(pred)
    target = np.asarray(target)

    if pred.shape != target.shape:
        raise ValueError(
            "pred and target must have the same shape, got "
            f"{pred.shape} and {target.shape}"
        )

    if pred.ndim == 0:
        raise ValueError("pred and target must have at least 1 dimension")

    axis = np.core.numeric.normalize_axis_index(axis, pred.ndim)
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")

    if fs not in (8000, 16000):
        raise ValueError(f"Unsupported fs={fs}. PESQ supports fs in {{8000, 16000}}")

    if mode is None:
        mode = "nb" if fs == 8000 else "wb"
    else:
        mode = mode.lower()
        if mode not in ("wb", "nb"):
            raise ValueError(f"Unsupported mode='{mode}', expected 'wb' or 'nb'")

    if fs == 8000 and mode != "nb":
        raise ValueError("fs=8000 only supports mode='nb'")

    # Move the sample axis to the end and evaluate PESQ independently per item.
    pred_2d = np.moveaxis(pred, axis, -1).reshape(-1, pred.shape[axis])
    target_2d = np.moveaxis(target, axis, -1).reshape(-1, target.shape[axis])
    if scale_invariant:
        target_2d = project_target_scale_invariant(
            pred=pred_2d, target=target_2d, axis=-1, eps=1e-10
        )

    if pred_2d.shape[0] == 1:
        scores = np.empty((1,), dtype=np.float64)
        scores[0] = pesq_fn(
            fs,
            np.ascontiguousarray(target_2d[0], dtype=np.float64),
            np.ascontiguousarray(pred_2d[0], dtype=np.float64),
            mode=mode,
        )
    else:
        try:
            scores = np.asarray(
                pesq_batch_fn(
                    fs,
                    np.ascontiguousarray(target_2d, dtype=np.float64),
                    np.ascontiguousarray(pred_2d, dtype=np.float64),
                    mode=mode,
                    n_processor=num_workers,
                ),
                dtype=np.float64,
            )
        except Exception:
            scores = np.empty((pred_2d.shape[0],), dtype=np.float64)
            for i in range(pred_2d.shape[0]):
                scores[i] = pesq_fn(
                    fs,
                    np.ascontiguousarray(target_2d[i], dtype=np.float64),
                    np.ascontiguousarray(pred_2d[i], dtype=np.float64),
                    mode=mode,
                )

    if pred.ndim == 1:
        return float(scores[0])

    out_shape = tuple(s for i, s in enumerate(pred.shape) if i != axis)
    return scores.reshape(out_shape)
