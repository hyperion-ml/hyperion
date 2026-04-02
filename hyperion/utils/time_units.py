"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np


def _to_frame_array(frames: Union[np.ndarray, Sequence[int]]) -> np.ndarray:
    """Convert frame indices to a 1D int64 numpy array."""
    arr = np.asarray(frames, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"frames must be 1D, got shape {arr.shape}")
    return arr


def _frame_params(
    fs: Union[int, float], frame_length: Union[int, float], frame_shift: Union[int, float]
) -> Tuple[int, int]:
    """Convert frame length/shift in ms to samples and validate them."""
    if fs <= 0:
        raise ValueError(f"fs must be > 0, got {fs}")
    frame_length_samples = int(frame_length * fs // 1000)
    frame_shift_samples = int(frame_shift * fs // 1000)
    if frame_length_samples <= 0:
        raise ValueError(
            f"frame_length in samples must be > 0, got {frame_length_samples}"
        )
    if frame_shift_samples <= 0:
        raise ValueError(
            f"frame_shift in samples must be > 0, got {frame_shift_samples}"
        )
    return frame_length_samples, frame_shift_samples


def _merge_frames(s_start: np.ndarray, s_end: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Merge overlapping/contiguous sample intervals.

    Args:
        s_start: Interval starts (inclusive), sorted ascending.
        s_end: Interval ends (exclusive), sorted by corresponding starts.

    Returns:
        Two arrays with merged starts and ends.
    """
    if s_start.shape != s_end.shape:
        raise ValueError(
            f"s_start and s_end must have same shape, got {s_start.shape} and {s_end.shape}"
        )
    if len(s_start) == 0:
        return s_start.copy(), s_end.copy()

    merge_idx = s_start[1:] <= s_end[:-1]
    num_frames = len(s_start) - np.sum(merge_idx)
    new_s_start = np.zeros((num_frames,), dtype=s_start.dtype)
    new_s_end = np.zeros((num_frames,), dtype=s_start.dtype)
    cur_frame = 0
    cur_end = s_end[0]
    new_s_start[0] = s_start[0]
    for i in range(1, len(s_start)):
        if merge_idx[i - 1]:
            cur_end = max(cur_end, s_end[i])
        else:
            new_s_end[cur_frame] = cur_end
            cur_frame += 1
            new_s_start[cur_frame] = s_start[i]
            cur_end = s_end[i]
    new_s_end[cur_frame] = cur_end
    return new_s_start, new_s_end


def frames_to_start_samples(
    frames: Union[np.ndarray, Sequence[int]],
    fs: Union[int, float],
    frame_length: Union[int, float],
    frame_shift: Union[int, float],
    snip_edges: bool,
    center: bool,
) -> np.ndarray:
    """Convert frame indices to start sample indices."""
    frames = _to_frame_array(frames)
    frame_length, frame_shift = _frame_params(fs, frame_length, frame_shift)
    if center:
        left_padding = int(frame_length // 2)
    else:
        if snip_edges:
            left_padding = 0
        else:
            left_padding = int((frame_length - frame_shift) // 2)

    s_start = frame_shift * frames - left_padding
    s_start = np.clip(s_start, a_min=0, a_max=None)
    return s_start


def frames_to_bound_samples(
    frames: Union[np.ndarray, Sequence[int]],
    fs: Union[int, float],
    frame_length: Union[int, float],
    frame_shift: Union[int, float],
    snip_edges: bool,
    center: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert frame indices to [start, end) sample boundaries."""
    frames = _to_frame_array(frames)
    frame_length, frame_shift = _frame_params(fs, frame_length, frame_shift)
    if center:
        left_padding = int(frame_length // 2)
    else:
        if snip_edges:
            left_padding = 0
        else:
            left_padding = int((frame_length - frame_shift) // 2)

    s_start = frame_shift * frames - left_padding
    s_end = s_start + frame_length
    s_start = np.clip(s_start, a_min=0, a_max=None)
    return s_start, s_end


def frames_to_center_samples(
    frames: Union[np.ndarray, Sequence[int]],
    fs: Union[int, float],
    frame_length: Union[int, float],
    frame_shift: Union[int, float],
    snip_edges: bool,
    center: bool,
) -> np.ndarray:
    """Convert frame indices to center sample indices."""
    frames = _to_frame_array(frames)
    frame_length, frame_shift = _frame_params(fs, frame_length, frame_shift)
    if center:
        center_0 = 0
    else:
        if snip_edges:
            center_0 = int(frame_length // 2)
        else:
            center_0 = int(frame_shift // 2)

    s_center = frame_shift * frames + center_0
    return s_center


def frames_to_samples(
    frames: Union[np.ndarray, Sequence[int]],
    fs: Union[int, float],
    frame_length: Union[int, float],
    frame_shift: Union[int, float],
    snip_edges: bool,
    center: bool,
) -> np.ndarray:
    """Expand frame indices into an explicit sample-index vector."""
    s_start, s_end = frames_to_bound_samples(
        frames, fs, frame_length, frame_shift, snip_edges, center
    )
    if len(s_start) == 0:
        return np.zeros((0,), dtype=np.int64)

    s_start, s_end = _merge_frames(s_start, s_end)
    deltas = s_end - s_start
    num_samples = int(np.sum(deltas))
    samples = np.zeros((num_samples,), dtype=s_start.dtype)
    cur_pos = 0
    for i in range(len(s_start)):
        cur_end = cur_pos + deltas[i]
        samples[cur_pos:cur_end] = np.arange(s_start[i], s_end[i])
        cur_pos = cur_end

    return samples


def frames_to_sample_mask(
    frames: Union[np.ndarray, Sequence[int]],
    max_samples: Optional[int],
    fs: Union[int, float],
    frame_length: Union[int, float],
    frame_shift: Union[int, float],
    snip_edges: bool,
    center: bool,
) -> np.ndarray:
    """Convert frame indices to a boolean sample mask."""
    s_start, s_end = frames_to_bound_samples(
        frames, fs, frame_length, frame_shift, snip_edges, center
    )
    if len(s_end) == 0:
        max_samples = 0 if max_samples is None else max_samples
        if max_samples < 0:
            raise ValueError(f"max_samples must be >= 0, got {max_samples}")
        return np.zeros((max_samples,), dtype=bool)

    if max_samples is None:
        max_samples = int(s_end[-1])
    if max_samples < 0:
        raise ValueError(f"max_samples must be >= 0, got {max_samples}")
    mask = np.zeros((max_samples,), dtype=bool)
    for i in range(len(s_start)):
        mask[s_start[i] : s_end[i]] = True

    return mask


def frames_to_start_timestamps(
    frames: Union[np.ndarray, Sequence[int]],
    fs: Union[int, float],
    frame_length: Union[int, float],
    frame_shift: Union[int, float],
    snip_edges: bool,
    center: bool,
) -> np.ndarray:
    """Convert frame indices to start timestamps in seconds."""
    s_start = frames_to_start_samples(
        frames, fs, frame_length, frame_shift, snip_edges, center
    )
    return s_start / fs


def frames_to_bound_timestamps(
    frames: Union[np.ndarray, Sequence[int]],
    fs: Union[int, float],
    frame_length: Union[int, float],
    frame_shift: Union[int, float],
    snip_edges: bool,
    center: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert frame indices to boundary timestamps in seconds."""
    s_start, s_end = frames_to_bound_samples(
        frames, fs, frame_length, frame_shift, snip_edges, center
    )
    t_start = s_start / fs
    t_end = s_end / fs
    return t_start, t_end


def frames_to_center_timestamps(
    frames: Union[np.ndarray, Sequence[int]],
    fs: Union[int, float],
    frame_length: Union[int, float],
    frame_shift: Union[int, float],
    snip_edges: bool,
    center: bool,
) -> np.ndarray:
    """Convert frame indices to center timestamps in seconds."""
    s_center = frames_to_center_samples(
        frames, fs, frame_length, frame_shift, snip_edges, center
    )
    return s_center / fs
