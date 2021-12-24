"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import math
import numpy as np


def _merge_frames(s_start, s_end):
    merge_idx = s_start[1:] <= s_end[:-1]
    num_frames = len(s_start) - np.sum(merge_idx)
    new_s_start = np.zeros((num_frames,), dtype=s_start.dtype)
    new_s_end = np.zeros((num_frames,), dtype=s_start.dtype)
    cur_frame = 0
    cur_end = s_end[0]
    new_s_start[0] = s_start[0]
    for i in range(1, len(s_start)):
        if merge_idx[i - 1]:
            cur_end = s_end[i]
        else:
            new_s_end[cur_frame] = cur_end
            cur_frame += 1
            new_s_start[cur_frame] = s_start[i]
            cur_end = s_end[i]
    new_s_end[cur_frame] = cur_end
    return new_s_start, new_s_end


def frames_to_start_samples(frames, fs, frame_length, frame_shift, snip_edges, center):
    frame_length = int(frame_length * fs // 1000)
    frame_shift = int(frame_shift * fs // 1000)
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


def frames_to_bound_samples(frames, fs, frame_length, frame_shift, snip_edges, center):
    frame_length = int(frame_length * fs // 1000)
    frame_shift = int(frame_shift * fs // 1000)
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


def frames_to_center_samples(frames, fs, frame_length, frame_shift, snip_edges, center):
    frame_length = int(frame_length * fs // 1000)
    frame_shift = int(frame_shift * fs // 1000)
    if center:
        center_0 = 0
    else:
        if snip_edges:
            center_0 = int(frame_length // 2)
        else:
            center_0 = int(frame_shift // 2)

    s_center = frame_shift * frames + center_0
    return s_center


def frames_to_samples(frames, fs, frame_length, frame_shift, snip_edges, center):
    s_start, s_end = frames_to_bound_samples(
        frames, fs, frame_length, frame_shift, snip_edges, center
    )
    s_start, s_end = _merge_frames(s_start, s_end)
    deltas = s_end - s_start
    num_samples = np.sum(deltas)
    samples = np.zeros((num_samples,), dtype=s_start.dtype)
    cur_pos = 0
    for i in range(len(s_start)):
        cur_end = cur_pos + deltas[i]
        samples[cur_pos:cur_end] = np.arange(s_start[i], s_end[i])
        cur_pos = cur_end

    return samples


def frames_to_sample_mask(
    frames, max_samples, fs, frame_length, frame_shift, snip_edges, center
):
    s_start, s_end = frames_to_bound_samples(
        frames, fs, frame_length, frame_shift, snip_edges, center
    )
    if max_samples is None:
        max_samples = s_end[-1] - 1
    mask = np.zeros((max_samples,), dtype=np.bool)
    for i in range(len(s_start)):
        mask[s_start[i] : s_end[i]] = True

    return mask


def frames_to_start_timestamps(
    frames, fs, frame_length, frame_shift, snip_edges, center
):
    s_start = frames_to_start_samples(
        frames, fs, frame_length, frame_shift, snip_edges, center
    )
    return s_start / fs


def frames_to_bound_timestamps(
    frames, fs, frame_length, frame_shift, snip_edges, center
):
    s_start, s_end = frames_to_bound_samples(
        frames, fs, frame_length, frame_shift, snip_edges, center
    )
    t_start = s_start / fs
    t_end = s_end / fs
    return t_start, t_end


def frames_to_center_timestamps(
    frames, fs, frame_length, frame_shift, snip_edges, center
):
    s_center = frames_to_center_samples(
        frames, fs, frame_length, frame_shift, snip_edges, center
    )
    return s_center / fs
