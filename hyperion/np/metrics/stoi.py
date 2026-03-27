"""
Copyright 2026 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Union

import numpy as np


def compute_stoi(
    pred: np.ndarray,
    target: np.ndarray,
    fs: int,
    axis: int = -1,
    extended: bool = False,
) -> Union[float, np.ndarray]:
    """Computes STOI between enhanced and reference signals.

    Args:
      pred: Predicted/processed signal array.
      target: Reference (clean) signal array.
      fs: Sampling frequency in Hz.
      axis: Time/sample axis. STOI is computed independently for each item in
        the remaining dimensions.
      extended: If True, computes ESTOI instead of STOI.

    Returns:
      STOI/ESTOI score as float (1D inputs) or an array with shape
      ``pred.shape`` excluding ``axis``.
    """
    try:
        from pystoi import stoi as stoi_fn
    except Exception as e:
        raise ImportError(
            "compute_stoi requires the 'pystoi' package. Install it with "
            "'pip install pystoi'."
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

    # Move sample axis to last dim and compute one STOI score per row.
    pred_2d = np.moveaxis(pred, axis, -1).reshape(-1, pred.shape[axis])
    target_2d = np.moveaxis(target, axis, -1).reshape(-1, target.shape[axis])

    scores = np.empty((pred_2d.shape[0],), dtype=np.float64)
    for i in range(pred_2d.shape[0]):
        scores[i] = stoi_fn(
            np.ascontiguousarray(target_2d[i], dtype=np.float64),
            np.ascontiguousarray(pred_2d[i], dtype=np.float64),
            fs_sig=fs,
            extended=extended,
        )

    if pred.ndim == 1:
        return float(scores[0])

    out_shape = tuple(s for i, s in enumerate(pred.shape) if i != axis)
    return scores.reshape(out_shape)
