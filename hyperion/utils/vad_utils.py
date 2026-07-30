"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional, Tuple

import numpy as np

from ..hyp_defs import float_cpu


def _assert_sorted(t_start: np.ndarray) -> None:
    """Assert that input timestamps are sorted in non-decreasing order."""
    t_start = np.asarray(t_start)
    if t_start.ndim != 1:
        raise ValueError(f"timestamps must be 1-D, got shape={t_start.shape}")
    if t_start.size <= 1:
        return
    delta = np.diff(t_start)
    if np.any(delta < 0):
        raise ValueError(f"time-stamps must be sorted {t_start=} {delta=}")


def _assert_pos_dur(t_start: np.ndarray, t_end: np.ndarray) -> None:
    """Assert that all segments have non-negative duration."""
    t_start = np.asarray(t_start)
    t_end = np.asarray(t_end)
    if t_start.ndim != 1 or t_end.ndim != 1:
        raise ValueError(f"timestamps must be 1-D, got {t_start.shape=} {t_end.shape=}")
    if t_start.shape[0] != t_end.shape[0]:
        raise ValueError(
            f"t_start and t_end must have same length, got {len(t_start)} and {len(t_end)}"
        )
    delta = t_end - t_start
    if np.any(delta < 0):
        raise ValueError(
            f"segments must have non-negative duration {t_start=} {t_end=} {delta=}"
        )


def merge_vad_timestamps(
    t_start: np.ndarray, t_end: np.ndarray, tol: float = 0.001
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge contiguous or overlapping VAD segments.

    Args:
      t_start: Segment start timestamps in seconds.
      t_end: Segment end timestamps in seconds.
      tol: Merge tolerance in seconds. Segments separated by less than ``tol``
        are merged.

    Returns:
      Tuple ``(t_start_out, t_end_out)`` with merged timestamps.
    """
    if tol < 0:
        raise ValueError(f"tol must be >= 0, got {tol}")

    # assert segments are shorted by start time, and positive dur
    _assert_sorted(t_start)
    _assert_pos_dur(t_start, t_end)
    # if empty return the same
    if len(t_start) == 0:
        return t_start, t_end
    t_start_out = np.zeros_like(t_start)
    t_end_out = np.zeros_like(t_end)
    t_start_cur = t_start[0]
    t_end_cur = t_end[0]
    j = 0
    for i in range(1, len(t_start)):
        t_start_i = t_start[i]
        t_end_i = t_end[i]
        if t_end_cur >= t_start_i - tol:
            # we merge with previous
            if t_end_i > t_end_cur:
                # this should be  true always except odd cases
                t_end_cur = t_end_i
        else:
            # new segment found
            # we write current segment to out_timestamps
            # and start new segment
            t_start_out[j] = t_start_cur
            t_end_out[j] = t_end_cur
            t_start_cur = t_start_i
            t_end_cur = t_end_i
            j += 1

    # write final segment
    t_start_out[j] = t_start_cur
    t_end_out[j] = t_end_cur
    t_start_out = t_start_out[: j + 1]
    t_end_out = t_end_out[: j + 1]
    return t_start_out, t_end_out


def bin_vad_to_timestamps(
    vad: np.ndarray,
    frame_length: float,
    frame_shift: float,
    snip_edges: bool = False,
    merge_tol: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a binary frame-level VAD vector to time segments.

    Args:
       vad: Boolean/0-1 VAD vector at frame resolution.
       frame_length: Frame length in milliseconds.
       frame_shift: Frame shift in milliseconds.
       snip_edges: If ``True``, timestamps are computed with Kaldi-style
         ``snip_edges=True``.
       merge_tol: Merge tolerance in seconds for adjacent output segments.

    Returns:
       Tuple ``(t_start, t_end)`` in seconds, relative to the file start.
    """
    if not isinstance(vad, np.ndarray):
        raise TypeError("vad must be np.ndarray")
    if vad.ndim != 1:
        raise ValueError(f"vad must be 1-D, got shape={vad.shape}")
    if frame_length <= 0 or frame_shift <= 0:
        raise ValueError(
            f"frame_length and frame_shift must be > 0, got {frame_length=} {frame_shift=}"
        )
    if merge_tol < 0:
        raise ValueError(f"merge_tol must be >= 0, got {merge_tol}")

    frame_length = frame_length / 1000
    frame_shift = frame_shift / 1000
    if snip_edges:
        start = 0
    else:
        start = -(frame_length - frame_shift) / 2

    start_timestamps = np.asarray(
        [start + frame_shift * i for i in range(len(vad)) if vad[i]]
    )
    end_timestamps = start_timestamps + frame_length
    start_timestamps[start_timestamps < 0] = 0
    return merge_vad_timestamps(start_timestamps, end_timestamps, tol=merge_tol)


def vad_timestamps_to_bin(
    t_start: np.ndarray,
    t_end: np.ndarray,
    frame_length: float,
    frame_shift: float,
    snip_edges: bool = False,
    duration: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """Convert VAD timestamp segments to a frame-level binary vector.

    Args:
       t_start: Segment start timestamps in seconds.
       t_end: Segment end timestamps in seconds.
       frame_length: Frame length in milliseconds.
       frame_shift: Frame shift in milliseconds.
       snip_edges: If ``True``, use Kaldi-style ``snip_edges=True`` framing.
       duration: Signal duration in seconds. If ``None``, it is inferred from
         the last ``t_end`` value.
       max_frames: Optional fixed output length. Output is padded or clipped to
         this number of frames.

    Returns:
       Boolean VAD vector indexed by frame.
    """
    if frame_length <= 0 or frame_shift <= 0:
        raise ValueError(
            f"frame_length and frame_shift must be > 0, got {frame_length=} {frame_shift=}"
        )
    if max_frames is not None and max_frames < 0:
        raise ValueError(f"max_frames must be >= 0, got {max_frames}")

    _assert_pos_dur(t_start, t_end)
    last_t_end = float(t_end[-1]) if len(t_end) > 0 else 0.0

    if duration is None:
        duration = last_t_end
    else:
        if duration < last_t_end:
            raise ValueError(
                f"duration must be >= last t_end ({last_t_end}), got {duration}"
            )

    frame_length = frame_length / 1000
    frame_shift = frame_shift / 1000

    frame_center = frame_length / 2
    if snip_edges:
        num_frames = int(
            np.floor((duration - frame_length + frame_shift) / frame_shift)
        )
        pad = 0
    else:
        num_frames = int(np.round(duration / frame_shift))
        pad = -(frame_length - frame_shift) / 2

    num_frames = max(0, num_frames)
    if max_frames is not None and num_frames < max_frames:
        num_frames = max_frames

    vad = np.zeros((num_frames,), dtype=bool)
    frame_start = np.ceil((t_start - (pad + frame_center)) / frame_shift).astype(
        dtype=int
    )
    frame_end = (
        np.floor((t_end - (pad + frame_center)) / frame_shift).astype(dtype=int) + 1
    )
    frame_start[frame_start < 0] = 0
    frame_end[frame_end > num_frames] = num_frames
    for i, j in zip(frame_start, frame_end):
        if j > i:
            vad[i:j] = True

    if max_frames is not None and num_frames > max_frames:
        vad = vad[:max_frames]

    return vad


def vad_timestamps_to_bin_samples(
    t_start: np.ndarray,
    t_end: np.ndarray,
    sample_frequency: float,
    duration: Optional[float] = None,
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """Convert VAD timestamp segments to a sample-level binary vector.

    Args:
       t_start: Segment start timestamps in seconds.
       t_end: Segment end timestamps in seconds.
       sample_frequency: Sampling rate in Hz.
       duration: Signal duration in seconds. If ``None``, it is inferred from
         the last ``t_end`` value.
       max_samples: Optional fixed output length. Output is padded or clipped to
         this number of samples.

    Returns:
       Boolean VAD vector indexed by sample.
    """
    if sample_frequency <= 0:
        raise ValueError(f"sample_frequency must be > 0, got {sample_frequency}")
    if max_samples is not None and max_samples < 0:
        raise ValueError(f"max_samples must be >= 0, got {max_samples}")

    _assert_pos_dur(t_start, t_end)
    last_t_end = float(t_end[-1]) if len(t_end) > 0 else 0.0

    if duration is None:
        duration = last_t_end
    else:
        if duration < last_t_end:
            raise ValueError(
                f"duration must be >= last t_end ({last_t_end}), got {duration}"
            )

    num_samples = int(duration * sample_frequency)
    num_samples = max(0, num_samples)
    if max_samples is not None:
        num_samples = max(num_samples, max_samples)

    sample_start = (t_start * sample_frequency).astype(int)
    sample_end = (t_end * sample_frequency + 1).astype(int)
    sample_start = np.clip(sample_start, 0, num_samples)
    sample_end = np.clip(sample_end, 0, num_samples)
    vad = np.zeros((num_samples,), dtype=bool)
    for i, j in zip(sample_start, sample_end):
        if j > i:
            vad[i:j] = True

    if max_samples is not None and max_samples < num_samples:
        vad = vad[:max_samples]

    return vad


def timestamps_wrt_vad_to_absolute_timestamps(
    t_start: np.ndarray,
    t_end: np.ndarray,
    vad_t_start: np.ndarray,
    vad_t_end: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert timestamps from VAD-compressed time to absolute time.

    This implementation works directly on timestamp intervals and avoids
    constructing dense binary vectors, which is substantially more efficient
    for long recordings.

    Args:
       t_start: Segment start timestamps in compressed (silence-removed) time.
       t_end: Segment end timestamps in compressed (silence-removed) time.
       vad_t_start: VAD segment starts in absolute time.
       vad_t_end: VAD segment ends in absolute time.

    Returns:
       Tuple ``(t_start_abs, t_end_abs)`` in absolute time.
    """
    _assert_pos_dur(t_start, t_end)
    _assert_sorted(vad_t_start)
    _assert_pos_dur(vad_t_start, vad_t_end)

    dtype = float_cpu()
    if len(t_start) == 0 or len(vad_t_start) == 0:
        empty = np.empty((0,), dtype=dtype)
        return empty, empty

    # Build cumulative speech timeline produced by concatenating VAD segments.
    vad_t_start = np.asarray(vad_t_start, dtype=dtype)
    vad_t_end = np.asarray(vad_t_end, dtype=dtype)
    # Merge overlaps/contiguity to avoid double-counting compressed speech time.
    vad_t_start, vad_t_end = merge_vad_timestamps(vad_t_start, vad_t_end, tol=0.0)
    vad_dur = vad_t_end - vad_t_start
    cum_speech = np.concatenate(
        (np.asarray([0], dtype=dtype), np.cumsum(vad_dur, dtype=dtype))
    )
    total_speech = cum_speech[-1]
    if total_speech <= 0:
        empty = np.empty((0,), dtype=dtype)
        return empty, empty

    # Match legacy behavior: operate on the union of input compressed segments.
    order = np.argsort(t_start, kind="mergesort")
    t_start = np.asarray(t_start, dtype=dtype)[order]
    t_end = np.asarray(t_end, dtype=dtype)[order]
    t_start, t_end = merge_vad_timestamps(t_start, t_end, tol=0.0)

    # Clamp to valid compressed timeline [0, total_speech].
    t_start = np.clip(t_start, 0, total_speech)
    t_end = np.clip(t_end, 0, total_speech)
    keep = t_end > t_start
    if not np.any(keep):
        empty = np.empty((0,), dtype=dtype)
        return empty, empty
    t_start = t_start[keep]
    t_end = t_end[keep]

    out_start = []
    out_end = []
    num_vad_segs = len(vad_dur)

    for s, e in zip(t_start, t_end):
        cur = s
        while cur < e:
            # Segment index on compressed timeline.
            k = int(np.searchsorted(cum_speech, cur, side="right") - 1)
            if k >= num_vad_segs:
                break

            chunk_end_in_compressed = min(e, cum_speech[k + 1])
            if chunk_end_in_compressed <= cur:
                # Numerical boundary guard.
                cur = cum_speech[k + 1]
                continue

            abs_start = vad_t_start[k] + (cur - cum_speech[k])
            abs_end = abs_start + (chunk_end_in_compressed - cur)
            out_start.append(abs_start)
            out_end.append(abs_end)
            cur = chunk_end_in_compressed

    if len(out_start) == 0:
        empty = np.empty((0,), dtype=dtype)
        return empty, empty

    out_start = np.asarray(out_start, dtype=dtype)
    out_end = np.asarray(out_end, dtype=dtype)
    out_start, out_end = merge_vad_timestamps(out_start, out_end, tol=0.0)
    return out_start, out_end


def timestamps_wrt_bin_vad_to_absolute_timestamps(
    t_start: np.ndarray,
    t_end: np.ndarray,
    vad: np.ndarray,
    frame_length: float,
    frame_shift: float,
    snip_edges: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert timestamps from VAD-compressed time to absolute time.

       VAD is provided in binary frame format.

    Args:
       t_start: Segment start timestamps in compressed (silence-removed) time.
       t_end: Segment end timestamps in compressed (silence-removed) time.
       vad: Binary VAD at frame resolution.
       frame_length: Frame length in milliseconds.
       frame_shift: Frame shift in milliseconds.
       snip_edges: If ``True``, VAD framing used ``snip_edges=True``.

    Returns:
       Tuple ``(t_start_abs, t_end_abs)`` in absolute time.
    """
    vad_t_start, vad_t_end = bin_vad_to_timestamps(
        vad, frame_length, frame_shift, snip_edges
    )
    return timestamps_wrt_vad_to_absolute_timestamps(
        t_start, t_end, vad_t_start, vad_t_end
    )


def intersect_segment_timestamps_with_vad(
    t_start: np.ndarray,
    t_end: np.ndarray,
    t_vad_start: np.ndarray,
    t_vad_end: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect input segments with VAD segments.

    The function keeps only voiced portions of each input segment.

    Args:
       t_start: Input segment starts in seconds.
       t_end: Input segment ends in seconds.
       t_vad_start: VAD segment starts in seconds.
       t_vad_end: VAD segment ends in seconds.

    Returns:
       ``speech_idx``: Boolean vector indicating which input segments contain
       speech.
       ``out_timestamps``: ``(N, 2)`` array with voiced output segments.
       ``out_timestamps2speech_segs``: For each output segment, index of the
       corresponding speech-containing input segment after dropping silent ones.
    """
    if not isinstance(t_start, np.ndarray) or not isinstance(t_end, np.ndarray):
        raise TypeError("t_start and t_end must be np.ndarray")
    if not isinstance(t_vad_start, np.ndarray) or not isinstance(t_vad_end, np.ndarray):
        raise TypeError("t_vad_start and t_vad_end must be np.ndarray")

    # assert segments are shorted by start time, and positive dur
    _assert_sorted(t_start)
    _assert_pos_dur(t_start, t_end)
    _assert_sorted(t_vad_start)
    _assert_pos_dur(t_vad_start, t_vad_end)

    # if empty return the same
    if t_start.shape[0] == 0:
        return (
            np.empty((0,), dtype=bool),
            np.empty((0, 2), dtype=float_cpu()),
            np.empty((0,), dtype=int),
        )

    num_vad_segs = len(t_vad_start)
    speech_idx = np.zeros((t_start.shape[0],), dtype=bool)
    out_timestamps = []
    out_timestamps2speech_segs = []
    count_speech = 0
    j = 0
    for (
        i,
        (t_start_i, t_end_i),
    ) in enumerate(zip(t_start, t_end)):
        is_speech = False
        while j < num_vad_segs and t_vad_end[j] <= t_start_i:
            j += 1

        if j == num_vad_segs:
            break

        k = j
        while t_start_i < t_end_i:
            if (
                k == num_vad_segs
                or t_vad_start[k] >= t_end_i
                or t_vad_end[k] <= t_start_i
            ):
                break
            # print('...', t_vad_start[k], t_vad_end[k], t_start_i, t_end_i)
            is_speech = True
            if t_vad_start[k] <= t_start_i:
                if t_vad_end[k] < t_end_i:
                    new_seg = [t_start_i, t_vad_end[k]]
                    t_start_i = t_vad_end[k]
                else:
                    new_seg = [t_start_i, t_end_i]
                    t_start_i = t_end_i

            else:
                if t_vad_end[k] < t_end_i:
                    new_seg = [t_vad_start[k], t_vad_end[k]]
                    t_start_i = t_vad_end[k]
                else:
                    new_seg = [t_vad_start[k], t_end_i]
                    t_start_i = t_end_i

            out_timestamps.append(new_seg)
            # print('......', out_timestamps)
            out_timestamps2speech_segs.append(count_speech)
            k += 1

        speech_idx[i] = is_speech
        if is_speech:
            count_speech += 1

    if len(out_timestamps) > 0:
        out_timestamps = np.asarray(out_timestamps, dtype=float_cpu())
    else:
        out_timestamps = np.empty((0, 2), dtype=float_cpu())
    out_timestamps2speech_segs = np.asarray(out_timestamps2speech_segs, dtype=int)

    return speech_idx, out_timestamps, out_timestamps2speech_segs


# def _assert_sorted(t):
#     delta = np.diff(t[:, 0])
#     assert np.all(delta >= 0), "time-stamps must be sorted"


# def _assert_pos_dur(t):
#     delta = t[:, 1] - t[:, 0]
#     assert np.all(delta >= 0), "segments must have positve duration"


# def merge_vad_timestamps(in_timestamps, tol=0.001):
#     """Merges vad timestamps that are contiguous

#     Args:
#       in_timestamps: original time-stamps in start-time, end-time format
#       tol: tolerance, segments separted less than tol will be merged
#     Returns:
#       Merged timestamps
#     """
#     # if empty return the same
#     if in_timestamps.shape[0] == 0:
#         return in_timestamps

#     # assert segments are shorted by start time, and positive dur
#     _assert_sorted(in_timestamps)
#     _assert_pos_dur(in_timestamps)

#     # assert segments are shorted by start time
#     delta = np.diff(in_timestamps[:, 0])
#     assert np.all(delta >= 0), "time-stamps must be sorted"

#     out_timestamps = np.zeros_like(in_timestamps)
#     t_start = in_timestamps[0, 0]
#     t_end = in_timestamps[0, 1]
#     j = 0
#     for i in range(1, in_timestamps.shape[0]):
#         t_start_i = in_timestamps[i, 0]
#         t_end_i = in_timestamps[i, 1]
#         if t_end >= t_start_i - tol:
#             # we merge with previous
#             if t_end_i > t_end:
#                 # this should be  true always except odd cases
#                 t_end = t_end_i
#         else:
#             # new segment found
#             # we write current segment to out_timestamps
#             # and start new segment
#             out_timestamps[j, 0] = t_start
#             out_timestamps[j, 1] = t_end
#             t_start = t_start_i
#             t_end = t_end_i
#             j += 1

#     # write final segment
#     out_timestamps[j, 0] = t_start
#     out_timestamps[j, 1] = t_end
#     out_timestamps = out_timestamps[: j + 1]
#     return out_timestamps


# def bin_vad_to_timestamps(
#     vad, frame_length, frame_shift, snip_edges=False, merge_tol=0.001
# ):
#     """Converts binary VAD to a list of start end time stamps

#     Args:
#        vad: Binary VAD
#        frame_length: frame-length used to compute the VAD
#        frame_shift: frame-shift used to compute the VAD
#        snip_edges: if True, computing VAD used snip-edges option
#        merge_tol: tolerance to merge contiguous segments
#     Returns:
#        VAD time stamps refered to the begining of the file
#     """
#     if snip_edges:
#         start = 0
#     else:
#         start = -(frame_length - frame_shift) / 2

#     start_timestamps = np.asarray(
#         [start + frame_shift * i for i in range(len(vad)) if vad[i]]
#     )[:, None]
#     end_timestamps = start_timestamps + frame_length
#     start_timestamps[start_timestamps < 0] = 0
#     timestamps = np.concatenate((start_timestamps, end_timestamps), axis=1)
#     return merge_vad_timestamps(timestamps, tol=merge_tol)


# def vad_timestamps_to_bin(
#     in_timestamps,
#     frame_length,
#     frame_shift,
#     snip_edges=False,
#     signal_length=None,
#     max_frames=None,
# ):
#     """Converts VAD time-stamps to a binary vector

#     Args:
#        in_timestamps: vad timestamps
#        frame_length: frame-length used to compute the VAD
#        frame_shift: frame-shift used to compute the VAD
#        snip_edges: if True, computing VAD used snip-edges option
#        signal_length: total duration of the signal, if None it takes it from the last timestamp
#        max_frames: expected number of frames, if None it computes automatically
#     Returns:
#        Binary VAD np.array
#     """
#     _assert_pos_dur(in_timestamps)

#     if signal_length is None:
#         signal_length = in_timestamps[-1, 1]
#     else:
#         assert signal_length >= in_timestamps[-1, 1]

#     frame_center = frame_length / 2
#     if snip_edges:
#         num_frames = int(
#             np.floor((signal_length - frame_length + frame_shift) / frame_shift)
#         )
#         pad = 0
#     else:
#         num_frames = int(np.round(signal_length / frame_shift))
#         pad = -(frame_length - frame_shift) / 2

#     if max_frames is not None and num_frames < max_frames:
#         num_frames = max_frames

#     vad = np.zeros((num_frames,), dtype=bool)
#     frame_start = np.ceil(
#         (in_timestamps[:, 0] - (pad + frame_center)) / frame_shift
#     ).astype(dtype=np.int)
#     frame_end = (
#         np.floor((in_timestamps[:, 1] - (pad + frame_center)) / frame_shift).astype(
#             dtype=np.int
#         )
#         + 1
#     )
#     frame_start[frame_start < 0] = 0
#     frame_end[frame_end > num_frames] = num_frames
#     for i, j in zip(frame_start, frame_end):
#         if j > i:
#             vad[i:j] = True

#     if max_frames is not None and num_frames > max_frames:
#         vad = vad[:max_frames]

#     return vad


# def timestamps_wrt_vad_to_absolute_timestamps(in_timestamps, vad_timestamps):
#     """Converts time stamps relative to a signal with silence removed
#        to absoulute time stamps in the original signal

#        VAD is provided in start-end timestamps format also.

#     Args:
#        in_timestamps: time stamps relative to a signal with silence removed
#        vad_timestamps: vad timestamps used to remove silence from signal

#     Returns:
#        Absolute VAD time-stamps
#     """

#     bin_in = vad_timestamps_to_bin(in_timestamps, frame_length=0.001, frame_shift=0.001)
#     bin_vad = vad_timestamps_to_bin(
#         vad_timestamps, frame_length=0.001, frame_shift=0.001
#     )

#     bin_out = np.zeros_like(bin_vad)
#     j = 0
#     max_j = len(bin_in)
#     for i in range(len(bin_out)):
#         if bin_vad[i]:
#             bin_out[i] = bin_in[j]
#             j += 1
#             if j == max_j:
#                 break

#     out_timestamps = bin_vad_to_timestamps(
#         bin_out, frame_length=0.001, frame_shift=0.001, merge_tol=0.001
#     )
#     return out_timestamps


# def timestamps_wrt_bin_vad_to_absolute_timestamps(
#     in_timestamps, vad, frame_length, frame_shift, snip_edges=False
# ):
#     """Converts time stamps relative to a signal with silence removed
#        to absoulute time stamps in the original signal

#        VAD is provided in binary format
#     Args:
#        in_timestamps: time stamps relative to a signal with silence removed
#        vad: Binary VAD
#        frame_length: frame-length used to compute the VAD
#        frame_shift: frame-shift used to compute the VAD
#        snip_edges: if True, computing VAD used snip-edges option

#     Returns:
#        Absolute VAD time-stamps
#     """
#     vad_timestamps = bin_vad_to_timestamps(vad, frame_length, frame_shift, snip_edges)
#     return timestamps_wrt_vad_to_absolute_timestamps(in_timestamps, vad_timestamps)


# def intersect_segment_timestamps_with_vad(in_timestamps, vad_timestamps):
#     """Intersects a list of segment timestamps with a VAD time-stamps
#         It returns only the segments that contain speech modifying
#         the start and end times to remove silence from the segments.

#     Args:
#        in_timestamps: time stamps of a list of segments refered to time 0.
#        vad_timestamps: vad timestamps

#     Returns:
#        Boolean array indicating which input segments contain speech
#        Array of output segments with silence removed
#        Array of indices, one index for each output segment indicating to which
#         input speech segment correspond to. The index correspond to input segments
#         after removing input segments that only contain silence.
#     """
#     # if empty return the same
#     if in_timestamps.shape[0] == 0:
#         return in_timestamps

#     # assert segments are shorted by start time, and positive dur
#     _assert_sorted(in_timestamps)
#     _assert_pos_dur(in_timestamps)
#     _assert_sorted(vad_timestamps)
#     _assert_pos_dur(vad_timestamps)

#     vad_start = vad_timestamps[:, 0]
#     vad_end = vad_timestamps[:, 1]
#     num_vad_segs = len(vad_start)
#     speech_idx = np.zeros((in_timestamps.shape[0],), dtype=bool)
#     out_timestamps = []
#     out_timestamps2speech_segs = []
#     count_speech = 0
#     j = 0
#     for i, stamps in enumerate(in_timestamps):
#         t_start, t_end = stamps
#         is_speech = False
#         while j < num_vad_segs and vad_end[j] <= t_start:
#             j += 1

#         if j == num_vad_segs:
#             break

#         k = j
#         while t_start < t_end:
#             if k == num_vad_segs or vad_start[k] >= t_end or vad_end[k] <= t_start:
#                 break
#             # print('...', vad_start[k], vad_end[k], t_start, t_end)
#             is_speech = True
#             if vad_start[k] <= t_start:
#                 if vad_end[k] < t_end:
#                     new_seg = [t_start, vad_end[k]]
#                     t_start = vad_end[k]
#                 else:
#                     new_seg = [t_start, t_end]
#                     t_start = t_end

#             else:
#                 if vad_end[k] < t_end:
#                     new_seg = [vad_start[k], vad_end[k]]
#                     t_start = vad_end[k]
#                 else:
#                     new_seg = [vad_start[k], t_end]
#                     t_start = t_end

#             out_timestamps.append(new_seg)
#             # print('......', out_timestamps)
#             out_timestamps2speech_segs.append(count_speech)
#             k += 1

#         speech_idx[i] = is_speech
#         if is_speech:
#             count_speech += 1

#     out_timestamps = np.asarray(out_timestamps)
#     out_timestamps2speech_segs = np.asarray(out_timestamps2speech_segs, dtype=np.int)

#     return speech_idx, out_timestamps, out_timestamps2speech_segs
