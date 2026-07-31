"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Literal, Optional, Tuple, Union, overload

import torch
import torch.amp as amp


def l2_norm(x: torch.Tensor, dim: int = 1, axis: Optional[int] = None) -> torch.Tensor:
    """Apply L2 normalization along a given dimension.

    Args:
      x: Input tensor.
      dim: Dimension used for normalization.
      axis: Deprecated alias for ``dim``.

    Returns:
      L2-normalized tensor with the same shape as ``x``.
    """
    if axis is not None:
        dim = axis

    with amp.autocast(enabled=False, device_type=x.device.type):
        norm = torch.norm(x.float(), 2, dim, True) + 1e-10
        y = torch.div(x, norm)
    return y


def compute_snr(
    x: torch.Tensor, n: torch.Tensor, dim: int = 1, axis: Optional[int] = None
) -> torch.Tensor:
    """Compute signal-to-noise ratio (SNR) in dB.

    Args:
      x: Tensor with signal values.
      n: Tensor with noise values.
      dim: Dimension along which power is averaged.
      axis: Deprecated alias for ``dim``.

    Returns:
      Tensor with SNR values in dB.
    """
    if axis is not None:
        dim = axis

    P_x = 10 * torch.log10(torch.mean(x**2, dim=dim))
    P_n = 10 * torch.log10(torch.mean(n**2, dim=dim))
    return P_x - P_n


def compute_stats_adv_attack(x: torch.Tensor, x_adv: torch.Tensor) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compute per-sample attack statistics from clean and adversarial inputs.

    Args:
      x: Clean signal tensor of shape ``(B, ...)``.
      x_adv: Adversarial signal tensor with same shape as ``x``.

    Returns:
      Tuple containing:
      ``(snr, p_x, p_n, x_l2, x_max, n_l0, n_l2, n_max)``.
      Inputs are flattened to shape ``(B, T)`` when ``x.dim() > 2``.
    """

    if x.dim() > 2:
        x = torch.flatten(x, start_dim=1)
        x_adv = torch.flatten(x_adv, start_dim=1)

    noise = x_adv - x
    P_x = 10 * torch.log10(torch.mean(x**2, dim=-1))
    P_n = 10 * torch.log10(torch.mean(noise**2, dim=-1))
    snr = P_x - P_n
    # x_l1 = torch.sum(torch.abs(x), dim=-1)
    x_l2 = torch.norm(x, dim=-1)
    x_linf = torch.max(torch.abs(x), dim=-1)[0]
    abs_n = torch.abs(noise)
    n_l0 = torch.sum(abs_n > 0, dim=-1).float()
    # n_l1 = torch.sum(abs_n, dim=-1)
    n_l2 = torch.norm(noise, dim=-1)
    n_linf = torch.max(abs_n, dim=-1)[0]
    return snr, P_x, P_n, x_l2, x_linf, n_l0, n_l2, n_linf


@overload
def get_selfsim_tarnon(
    y: torch.Tensor, return_mask: Literal[False] = False
) -> torch.Tensor: ...


@overload
def get_selfsim_tarnon(
    y: torch.Tensor, return_mask: Literal[True]
) -> Tuple[torch.Tensor, torch.Tensor]: ...


def get_selfsim_tarnon(
    y: torch.Tensor, return_mask: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute target/non-target self-similarity from integer class labels.

    Args:
      y: Integer label tensor of shape ``(B,)``.
      return_mask: If ``True``, also return the strict upper-triangular mask.

    Returns:
      If ``return_mask`` is ``False``, a float matrix of shape ``(B, B)`` with
      ones for same-class pairs and zeros otherwise.
      If ``return_mask`` is ``True``, returns ``(selfsim, mask)``, where ``mask``
      is a boolean upper-triangular matrix (zero diagonal).
    """
    y_bin = y.unsqueeze(-1) - y.unsqueeze(0) + 1
    y_bin[y_bin != 1] = 0
    y_bin = y_bin.float()
    if not return_mask:
        return y_bin

    mask = torch.triu(torch.ones_like(y_bin, dtype=torch.bool), diagonal=1)
    return y_bin, mask


def slice_segments(
    x: torch.Tensor,
    start_idx: torch.Tensor,
    segment_length: int,
    dim: int = 1,
    permissive: int = 0,
) -> torch.Tensor:
    """Slice fixed-length segments per batch item along ``dim``.

    Args:
        x: Input tensor of shape ``(B, ..., T, ...)``.
        start_idx: Start indices of shape ``(B,)`` (one per batch element).
        segment_length: Length of each extracted segment.
        dim: Dimension along which to slice.
        permissive: Allowed right overflow. If a segment exceeds the tensor by at
            most this value, the window is shifted left to fit; otherwise raises.

    Returns:
        Tensor with same shape as ``x`` except length ``segment_length`` along
        ``dim``. If ``segment_length > x.size(dim)``, returns ``x`` unchanged.
    """
    if dim < 0:
        dim = x.dim() + dim

    t = x.size(dim)
    if segment_length > t:
        return x

    transposed = False
    if dim != x.dim() - 1:
        x = x.transpose(dim, -1)
        transposed = True
    # print("slice", x.shape, start_idx, segment_length, flush=True)
    y = torch.zeros_like(x[..., :segment_length])
    for i in range(x.size(0)):
        start_i = start_idx[i]
        end_i = start_i + segment_length
        overflow = end_i - x.shape[-1]
        if overflow > 0:
            if overflow > permissive:
                raise ValueError(
                    f"Segment end index {end_i} exceeds tensor length {x.shape[-1]} for element {i}."
                )
            else:
                start_i -= overflow
                end_i -= overflow
        y[i] = x[i, ..., start_i:end_i]

    if transposed:
        y = y.transpose(dim, -1)
    return y


def rand_slice_segments(
    x: torch.Tensor,
    x_lengths: Optional[torch.Tensor],
    segment_length: int,
    dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly slice fixed-length segments along a target dimension.

    Args:
        x: Input tensor of shape ``(B, ..., T, ...)``.
        x_lengths: Optional valid lengths along ``dim`` with shape ``(B,)``.
            If ``None``, full length ``T`` is used for every batch item.
        segment_length: Segment length to extract.
        dim: Dimension along which to slice.

    Returns:
        Tuple ``(segments, start_idx)`` where:
        - ``segments`` has the same shape as ``x`` except length
          ``segment_length`` along ``dim``.
        - ``start_idx`` is a ``(B,)`` tensor of sampled start indices.
    """
    b = x.size(0)
    t = x.size(dim)
    if segment_length > t:
        return x, torch.zeros(b, dtype=torch.long, device=x.device)

    if x_lengths is None:
        x_lengths = torch.full((b,), t, dtype=torch.long, device=x.device)

    max_start = x_lengths - segment_length
    start_idx = (
        (max_start * torch.rand(size=(b,), device=x.device))
        .to(dtype=torch.long)
        .clamp(min=0)
    )
    y = slice_segments(x, start_idx, segment_length, dim=dim)
    return y, start_idx


def rand_slice_feat_segments(
    feats: torch.Tensor,
    feat_lengths: Optional[torch.Tensor],
    segment_duration: float,
    sample_freq: int,
    frame_shift: int,
    dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly slice fixed-duration segments from feature sequences.

    Args:
        feats: Feature tensor of shape ``(B, ..., T, ...)``.
        feat_lengths: Optional valid frame lengths of shape ``(B,)``.
        segment_duration: Segment duration in seconds.
        sample_freq: Sampling rate in Hz.
        frame_shift: Frame shift in samples.
        dim: Dimension along which to slice.

    Returns:
        Tuple ``(segments, start_idx)`` from :func:`rand_slice_segments`.
    """
    segment_size = int(segment_duration * sample_freq / frame_shift)
    return rand_slice_segments(feats, feat_lengths, segment_size, dim=dim)


def rand_slice_audio_segments(
    audios: torch.Tensor,
    audio_lengths: Optional[torch.Tensor],
    segment_duration: float,
    sample_freq: int,
    dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly slice fixed-duration segments from audio waveforms.

    Args:
        audios: Audio tensor of shape ``(B, ..., T, ...)``.
        audio_lengths: Optional valid lengths in samples with shape ``(B,)``.
        segment_duration: Segment duration in seconds.
        sample_freq: Sampling rate in Hz.
        dim: Dimension along which to slice.

    Returns:
        Tuple ``(segments, start_idx)`` from :func:`rand_slice_segments`.
    """
    segment_size = int(segment_duration * sample_freq)
    return rand_slice_segments(audios, audio_lengths, segment_size, dim=dim)
