"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Union

import numpy as np


def project_target_scale_invariant(
    pred: np.ndarray, target: np.ndarray, axis: int = -1, eps: float = 1e-10
) -> np.ndarray:
    """Projects target onto pred using SI-SNR style zero-mean projection.

    Args:
      pred: Predicted/estimated signal array.
      target: Reference signal array.
      axis: Axis along which to compute the projection.
      eps: Small constant to avoid division by zero.

    Returns:
      Scale-invariant projection of ``target`` onto ``pred``.
    """
    pred_zm = pred - np.mean(pred, axis=axis, keepdims=True)
    target_zm = target - np.mean(target, axis=axis, keepdims=True)

    proj = np.sum(pred_zm * target_zm, axis=axis, keepdims=True)
    target_ref_energy = np.sum(target_zm**2, axis=axis, keepdims=True) + eps
    return proj * target_zm / target_ref_energy


def compute_snr(
    pred: np.ndarray, target: np.ndarray, axis: int = -1
) -> Union[float, np.ndarray]:
    """Computes Signal-to-Noise Ratio (SNR) in decibels.

    Args:
      pred: Predicted/processed signal array.
      target: Reference (clean) signal array.
      axis: Axis along which to compute the mean power.

    Returns:
      SNR in dB.
    """
    noise = pred - target
    P_target = 10 * np.log10(np.mean(target**2, axis=axis))
    P_noise = 10 * np.log10(np.mean(noise**2, axis=axis))
    return P_target - P_noise


def compute_si_snr(
    pred: np.ndarray, target: np.ndarray, axis: int = -1, eps: float = 1e-10
) -> Union[float, np.ndarray]:
    """Computes scale-invariant Signal-to-Noise Ratio (SI-SNR) in decibels.

    Per sample along ``axis``, define
    .. math::
       \tilde{\mathbf{p}} = \mathbf{pred} - \operatorname{mean}(\mathbf{pred}),
       \qquad
       \tilde{\mathbf{t}} = \mathbf{target} - \operatorname{mean}(\mathbf{target})

    Then compute the target projection and residual:
    .. math::
       \mathbf{t}_{\mathrm{proj}} =
       \frac{\langle \tilde{\mathbf{p}}, \tilde{\mathbf{t}} \rangle}
            {\lVert \tilde{\mathbf{t}} \rVert_2^2}
       \tilde{\mathbf{t}},
       \qquad
       \mathbf{e}_{\mathrm{noise}} =
       \tilde{\mathbf{p}} - \mathbf{t}_{\mathrm{proj}}

    and finally
    .. math::
       \mathrm{SI\mbox{-}SNR} =
       10 \log_{10}
       \left(
       \frac{\lVert \mathbf{t}_{\mathrm{proj}} \rVert_2^2}
            {\lVert \mathbf{e}_{\mathrm{noise}} \rVert_2^2}
       \right)

    Args:
      pred: Predicted/estimated signal array.
      target: Reference signal array.
      axis: Axis along which to compute the projection and power.
      eps: Small constant to avoid division by zero.

    Returns:
      SI-SNR in dB.
    """
    pred_zm = pred - np.mean(pred, axis=axis, keepdims=True)
    target_proj = project_target_scale_invariant(
        pred=pred, target=target, axis=axis, eps=eps
    )
    e_noise = pred_zm - target_proj

    target_energy = np.sum(target_proj**2, axis=axis)
    noise_energy = np.sum(e_noise**2, axis=axis) + eps
    return 10 * np.log10(target_energy / noise_energy)
