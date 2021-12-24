"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import numpy as np

from ..hyp_defs import float_cpu


def _assert_sorted(t):
    delta = np.diff(t[:, 0])
    assert np.all(delta >= 0), "time-stamps must be sorted"


def _assert_pos_dur(t):
    delta = t[:, 1] - t[:, 0]
    assert np.all(delta >= 0), "segments must have positve duration"


def merge_vad_timestamps(in_timestamps, tol=0.001):
    """Merges vad timestamps that are contiguous

    Args:
      in_timestamps: original time-stamps in start-time, end-time format
      tol: tolerance, segments separted less than tol will be merged
    Returns:
      Merged timestamps
    """
    # if empty return the same
    if in_timestamps.shape[0] == 0:
        return in_timestamps

    # assert segments are shorted by start time, and positive dur
    _assert_sorted(in_timestamps)
    _assert_pos_dur(in_timestamps)

    # assert segments are shorted by start time
    delta = np.diff(in_timestamps[:, 0])
    assert np.all(delta >= 0), "time-stamps must be sorted"

    out_timestamps = np.zeros_like(in_timestamps)
    t_start = in_timestamps[0, 0]
    t_end = in_timestamps[0, 1]
    j = 0
    for i in range(1, in_timestamps.shape[0]):
        t_start_i = in_timestamps[i, 0]
        t_end_i = in_timestamps[i, 1]
        if t_end >= t_start_i - tol:
            # we merge with previous
            if t_end_i > t_end:
                # this should be  true always except odd cases
                t_end = t_end_i
        else:
            # new segment found
            # we write current segment to out_timestamps
            # and start new segment
            out_timestamps[j, 0] = t_start
            out_timestamps[j, 1] = t_end
            t_start = t_start_i
            t_end = t_end_i
            j += 1

    # write final segment
    out_timestamps[j, 0] = t_start
    out_timestamps[j, 1] = t_end
    out_timestamps = out_timestamps[: j + 1]
    return out_timestamps


def bin_vad_to_timestamps(
    vad, frame_length, frame_shift, snip_edges=False, merge_tol=0.001
):
    """Converts binary VAD to a list of start end time stamps

    Args:
       vad: Binary VAD
       frame_length: frame-length used to compute the VAD
       frame_shift: frame-shift used to compute the VAD
       snip_edges: if True, computing VAD used snip-edges option
       merge_tol: tolerance to merge contiguous segments
    Returns:
       VAD time stamps refered to the begining of the file
    """
    if snip_edges:
        start = 0
    else:
        start = -(frame_length - frame_shift) / 2

    start_timestamps = np.asarray(
        [start + frame_shift * i for i in range(len(vad)) if vad[i]]
    )[:, None]
    end_timestamps = start_timestamps + frame_length
    start_timestamps[start_timestamps < 0] = 0
    timestamps = np.concatenate((start_timestamps, end_timestamps), axis=1)
    return merge_vad_timestamps(timestamps, tol=merge_tol)


def vad_timestamps_to_bin(
    in_timestamps,
    frame_length,
    frame_shift,
    snip_edges=False,
    signal_length=None,
    max_frames=None,
):
    """Converts VAD time-stamps to a binary vector

    Args:
       in_timestamps: vad timestamps
       frame_length: frame-length used to compute the VAD
       frame_shift: frame-shift used to compute the VAD
       snip_edges: if True, computing VAD used snip-edges option
       signal_length: total duration of the signal, if None it takes it from the last timestamp
       max_frames: expected number of frames, if None it computes automatically
    Returns:
       Binary VAD np.array
    """
    _assert_pos_dur(in_timestamps)

    if signal_length is None:
        signal_length = in_timestamps[-1, 1]
    else:
        assert signal_length >= in_timestamps[-1, 1]

    frame_center = frame_length / 2
    if snip_edges:
        num_frames = int(
            np.floor((signal_length - frame_length + frame_shift) / frame_shift)
        )
        pad = 0
    else:
        num_frames = int(np.round(signal_length / frame_shift))
        pad = -(frame_length - frame_shift) / 2

    if max_frames is not None and num_frames < max_frames:
        num_frames = max_frames

    vad = np.zeros((num_frames,), dtype=np.bool)
    frame_start = np.ceil(
        (in_timestamps[:, 0] - (pad + frame_center)) / frame_shift
    ).astype(dtype=np.int)
    frame_end = (
        np.floor((in_timestamps[:, 1] - (pad + frame_center)) / frame_shift).astype(
            dtype=np.int
        )
        + 1
    )
    frame_start[frame_start < 0] = 0
    frame_end[frame_end > num_frames] = num_frames
    for i, j in zip(frame_start, frame_end):
        if j > i:
            vad[i:j] = True

    if max_frames is not None and num_frames > max_frames:
        vad = vad[:max_frames]

    return vad


def timestamps_wrt_vad_to_absolute_timestamps(in_timestamps, vad_timestamps):
    """Converts time stamps relative to a signal with silence removed
       to absoulute time stamps in the original signal

       VAD is provided in start-end timestamps format also.

    Args:
       in_timestamps: time stamps relative to a signal with silence removed
       vad_timestamps: vad timestamps used to remove silence from signal

    Returns:
       Absolute VAD time-stamps
    """

    bin_in = vad_timestamps_to_bin(in_timestamps, frame_length=0.001, frame_shift=0.001)
    bin_vad = vad_timestamps_to_bin(
        vad_timestamps, frame_length=0.001, frame_shift=0.001
    )

    bin_out = np.zeros_like(bin_vad)
    j = 0
    max_j = len(bin_in)
    for i in range(len(bin_out)):
        if bin_vad[i]:
            bin_out[i] = bin_in[j]
            j += 1
            if j == max_j:
                break

    out_timestamps = bin_vad_to_timestamps(
        bin_out, frame_length=0.001, frame_shift=0.001, merge_tol=0.001
    )
    return out_timestamps


def timestamps_wrt_bin_vad_to_absolute_timestamps(
    in_timestamps, vad, frame_length, frame_shift, snip_edges=False
):
    """Converts time stamps relative to a signal with silence removed
       to absoulute time stamps in the original signal

       VAD is provided in binary format
    Args:
       in_timestamps: time stamps relative to a signal with silence removed
       vad: Binary VAD
       frame_length: frame-length used to compute the VAD
       frame_shift: frame-shift used to compute the VAD
       snip_edges: if True, computing VAD used snip-edges option

    Returns:
       Absolute VAD time-stamps
    """
    vad_timestamps = bin_vad_to_timestamps(vad, frame_length, frame_shift, snip_edges)
    return timestamps_wrt_vad_to_absolute_timestamps(in_timestamps, vad_timestamps)


def intersect_segment_timestamps_with_vad(in_timestamps, vad_timestamps):
    """Intersects a list of segment timestamps with a VAD time-stamps
        It returns only the segments that contain speech modifying
        the start and end times to remove silence from the segments.

    Args:
       in_timestamps: time stamps of a list of segments refered to time 0.
       vad_timestamps: vad timestamps

    Returns:
       Boolean array indicating which input segments contain speech
       Array of output segments with silence removed
       Array of indices, one index for each output segment indicating to which
        input speech segment correspond to. The index correspond to input segments
        after removing input segments that only contain silence.
    """
    # if empty return the same
    if in_timestamps.shape[0] == 0:
        return in_timestamps

    # assert segments are shorted by start time, and positive dur
    _assert_sorted(in_timestamps)
    _assert_pos_dur(in_timestamps)
    _assert_sorted(vad_timestamps)
    _assert_pos_dur(vad_timestamps)

    vad_start = vad_timestamps[:, 0]
    vad_end = vad_timestamps[:, 1]
    num_vad_segs = len(vad_start)
    speech_idx = np.zeros((in_timestamps.shape[0],), dtype=np.bool)
    out_timestamps = []
    out_timestamps2speech_segs = []
    count_speech = 0
    j = 0
    for i, stamps in enumerate(in_timestamps):
        t_start, t_end = stamps
        is_speech = False
        while j < num_vad_segs and vad_end[j] <= t_start:
            j += 1

        if j == num_vad_segs:
            break

        k = j
        while t_start < t_end:
            if k == num_vad_segs or vad_start[k] >= t_end or vad_end[k] <= t_start:
                break
            # print('...', vad_start[k], vad_end[k], t_start, t_end)
            is_speech = True
            if vad_start[k] <= t_start:
                if vad_end[k] < t_end:
                    new_seg = [t_start, vad_end[k]]
                    t_start = vad_end[k]
                else:
                    new_seg = [t_start, t_end]
                    t_start = t_end

            else:
                if vad_end[k] < t_end:
                    new_seg = [vad_start[k], vad_end[k]]
                    t_start = vad_end[k]
                else:
                    new_seg = [vad_start[k], t_end]
                    t_start = t_end

            out_timestamps.append(new_seg)
            # print('......', out_timestamps)
            out_timestamps2speech_segs.append(count_speech)
            k += 1

        speech_idx[i] = is_speech
        if is_speech:
            count_speech += 1

    out_timestamps = np.asarray(out_timestamps)
    out_timestamps2speech_segs = np.asarray(out_timestamps2speech_segs, dtype=np.int)

    return speech_idx, out_timestamps, out_timestamps2speech_segs
