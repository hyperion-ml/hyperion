"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Tuple

import torch
import torch.cuda.amp as amp
import torch.nn as nn


def l2_norm(x, dim=1, axis=None):
    """Applies length normalization to vectors.

    Args:
      x: input tensor.
      dim: dimension along which normalize the vectors.
      axis: same as dim (deprecated).

      Returns:
        Normalized tensor.
    """
    if axis is not None:
        dim = axis

    with amp.autocast(enabled=False):
        norm = torch.norm(x.float(), 2, dim, True) + 1e-10
        y = torch.div(x, norm)
    return y


def compute_snr(x, n, dim=1, axis=None):
    """Computes SNR (dB)

    Args:
      x: tensor with clean signal.
      n: tensor with noisy signal
      dim: dimension along which normalize power.
      axis: same as dim (deprecated).

    Returns:
      Tensor with SNR(dB)
    """
    if axis is not None:
        dim = axis

    P_x = 10 * torch.log10(torch.mean(x**2, dim=dim))
    P_n = 10 * torch.log10(torch.mean(n**2, dim=dim))
    return P_x - P_n


def compute_stats_adv_attack(x, x_adv):
    """Compute statistics of adversarial attack sample.

    Args:
      x: benign signal tensor.
      x_adv: adversarial signal tensor.

    Returns:
      SNR (dB).
      Power of x.
      Power of n.
      L2 norm of x.
      Linf norm of x.
      L0 norm of n.
      L2 norm of n.
      Linf norm of n.
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
    x_linf = torch.max(x, dim=-1)[0]
    abs_n = torch.abs(noise)
    n_l0 = torch.sum(abs_n > 0, dim=-1).float()
    # n_l1 = torch.sum(abs_n, dim=-1)
    n_l2 = torch.norm(noise, dim=-1)
    n_linf = torch.max(noise, dim=-1)[0]
    return snr, P_x, P_n, x_l2, x_linf, n_l0, n_l2, n_linf


def get_selfsim_tarnon(y, return_mask=False):
    """Computes ground truth selfsimilarity matrix given
       integer class labels.

    Args:
      y: integer tensor with class labels of shape (batch,).
      return_mask: If True, it returns upper triangular mask with zero diagonal.

    Returns:
      Self-similarity binary matrix wiht shape=(batch, batch).
      Upper triangular mask.
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
    """
    Slices segments from the input tensor along a specified dimension using start indices.

    Args:
        x (torch.Tensor): Input tensor of shape (B, ..., T, ...).
        start_idx (torch.Tensor): Tensor of shape (B,) with start indices per batch.
        segment_length (int): Length of the segment to slice.
        dim (int): Dimension along which to slice (default=1).
        permissive (int):

    Returns:
        torch.Tensor: Tensor with sliced segments, same shape as x but sliced along `dim`.
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
    x_lengths: torch.Tensor,
    segment_length: int,
    dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly slices segments from the input tensor along a specified dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (B, ..., T, ...).
        x_lengths (torch.Tensor): Tensor of shape (B,) containing the valid lengths along `dim`.
        segment_length (int): Length of the segment to slice.
        dim (int): Dimension along which to slice (default: 1).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Sliced tensor of shape similar to `x` but with length `segment_size` along `dim`.
            - Tensor of shape (B,) with the starting indices of the slices.
    """
    b = x.size(0)
    t = x.size(dim)
    if segment_length > t:
        return x, torch.zeros(b, dtype=torch.long, device=x.device)

    if x_lengths is None:
        x_lengths = t

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
    feat_lengths: torch.Tensor,
    segment_duration: float,
    sample_freq: int,
    frame_shift: int,
    dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly slices fixed-duration segments from feature sequences.

    Args:
        feats (torch.Tensor): Feature tensor of shape (B, ..., T, ...).
        feat_lengths (torch.Tensor): Tensor of shape (B,) with valid lengths in frames.
        segment_duration (float): Desired segment length in seconds.
        sample_freq (int): Sampling rate in Hz.
        frame_shift (int): Frame shift in samples (e.g., 160 for 10 ms @ 16 kHz).
        dim (int): Dimension along which to slice (default: 1).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Sliced feature tensor with segment length along `dim`.
            - Tensor of shape (B,) with start indices for each slice.
    """
    segment_size = int(segment_duration * sample_freq / frame_shift)
    return rand_slice_segments(feats, feat_lengths, segment_size, dim=dim)


def rand_slice_audio_segments(
    audios: torch.Tensor,
    audio_lengths: torch.Tensor,
    segment_duration: float,
    sample_freq: int,
    dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly slices fixed-duration segments from raw audio waveforms.

    Args:
        audios (torch.Tensor): Audio tensor of shape (B, ..., T, ...).
        audio_lengths (torch.Tensor): Tensor of shape (B,) with audio lengths in samples.
        segment_duration (float): Desired segment length in seconds.
        sample_freq (int): Sampling rate in Hz.
        dim (int): Dimension along which to slice (default: -1).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Sliced audio tensor with segment length along `dim`.
            - Tensor of shape (B,) with start indices for each slice.
    """
    segment_size = int(segment_duration * sample_freq)
    return rand_slice_segments(audios, audio_lengths, segment_size, dim=dim)
