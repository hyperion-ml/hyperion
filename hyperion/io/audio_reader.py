"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import io
import logging
import math
import os
import subprocess
from types import TracebackType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from numpy.typing import NDArray

from ..hyp_defs import float_cpu
from ..np.preprocessing.resampler import Any2AnyFreqResampler
from ..utils import HyperDataset, PathLike, RecordingSet, SegmentSet

valid_ext = [
    ".wav",
    ".flac",
    ".ogg",
    ".au",
    ".avr",
    ".caf",
    ".htk",
    ".iff",
    ".mat",
    ".mpc",
    ".oga",
    ".pvf",
    ".rf64",
    ".sd2",
    ".sds",
    ".sf",
    ".voc",
    ".w64",
    ".wve",
    ".xi",
    ".mp3",
]


class AudioReader:
    """Read audio waveforms from wav, flac, pipe commands, and related formats.

    This class receives either a :class:`HyperDataset` or standalone
    :class:`RecordingSet` (with an optional :class:`SegmentSet`). When providing
    only a recordings table, the reader can use the accompanying segment table
    to extract specific time spans from each recording.

    Args:
        dataset (Union[HyperDataset, PathLike, None]): Dataset instance or path to
            a dataset file containing recordings and, optionally, segments.
        recordings (Union[RecordingSet, PathLike, None]): Recording table or path
            to a recordings file when ``dataset`` is ``None``.
        segments (Union[SegmentSet, PathLike, None]): Segment table or path to a
            segments file. Must be ``None`` when ``dataset`` is provided.
        wav_scale (float): Multiplicative factor applied to every waveform.
        target_sample_freq (Optional[int]): Target sampling frequency used for
            optional resampling.
        channels_first (bool): If ``True`` returns waveforms using the
            ``(channels, num_samples)`` ordering; otherwise uses
            ``(num_samples, channels)``.
        always_2d (bool): If ``True`` keeps a trailing channel dimension even for
            mono audio.
        return_all_channels (bool): If ``True`` returns every channel from
            multi-channel recordings instead of selecting a single channel.

    Attributes:
        recordings (RecordingSet): Table describing the recordings to load.
        segments (Optional[SegmentSet]): Table with segment definitions when the
            reader operates in segment mode.
        with_segments (bool): Whether the reader was configured with segment
            metadata.
        wav_scale (float): Multiplicative factor applied to each waveform.
        target_sample_freq (Optional[int]): Requested resampling target. ``None``
            disables resampling.
        channels_first (bool): Shape convention for returned waveforms.
        always_2d (bool): Whether mono audio retains a channel axis.
        return_all_channels (bool): Whether to return every channel for
            multi-channel audio.
        resampler (Optional[Any2AnyFreqResampler]): Resampler used when a target
            sampling rate is requested.
    """

    def __init__(
        self,
        dataset: Union[HyperDataset, PathLike, None] = None,
        recordings: Union[RecordingSet, PathLike, None] = None,
        segments: Union[SegmentSet, PathLike, None] = None,
        wav_scale: float = 1.0,
        target_sample_freq: Optional[int] = None,
        channels_first: bool = True,
        always_2d: bool = False,
        return_all_channels: bool = False,
    ):
        assert (dataset is None) != (
            recordings is None
        ), "dataset xor recordings must be given"
        assert (segments is None) or (
            dataset is None
        ), "if dataset is given, segments must be None"

        if dataset is not None:
            if not isinstance(dataset, HyperDataset):
                dataset = HyperDataset.load(dataset)

            recordings = dataset.recordings(keep_loaded=False)
            segments = dataset.segments(keep_loaded=False)
        else:
            if not isinstance(recordings, RecordingSet):
                recordings = RecordingSet.load(recordings)

            if segments is not None:
                if not isinstance(segments, SegmentSet):
                    segments = SegmentSet.load(segments)

        self.recordings: RecordingSet = recordings
        self.segments: Optional[SegmentSet] = segments
        self.with_segments: bool = False if segments is None else True
        self.wav_scale: float = wav_scale
        self.target_sample_freq: Optional[int] = target_sample_freq
        self.channels_first: bool = channels_first
        self.always_2d: bool = always_2d
        self.return_all_channels: bool = return_all_channels
        self.resampler: Optional[Any2AnyFreqResampler] = None
        if self.target_sample_freq is not None or "target_sample_freq" in recordings:
            self.resampler = Any2AnyFreqResampler()

    @property
    def keys(self) -> NDArray[Any]:
        if self.with_segments:
            return self.segments["id"].values
        return self.recordings["id"].values

    def __enter__(self) -> "AudioReader":
        """Enter the context manager.

        Returns:
            AudioReader: The current reader instance for chained usage.

        Example:
            >>> with AudioReader(dataset="file.h5") as reader:
            ...     keys, data, fs = reader.read()
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Exit the context manager.

        Args:
            exc_type: Exception type raised inside the context (if any).
            exc_value: Exception value raised inside the context (if any).
            traceback: Traceback information for the exception (if any).

        The reader does not allocate external resources, so no explicit cleanup
        is required.
        """
        pass

    @staticmethod
    def channel_name_to_idx(channel: Union[int, str], num_channels: int) -> int:
        """Convert a human-readable channel descriptor to a zero-based index.

        Args:
            channel (Union[int, str]): Channel index (1-based) or mnemonic such
                as ``"left"``/``"right"``/``"center"``.
            num_channels (int): Number of channels available in the recording.

        Returns:
            int: Zero-based channel index.

        Raises:
            Exception: If ``channel`` is unknown or not available in the
                recording.
        """
        if isinstance(channel, int):
            return channel - 1
        if isinstance(channel, str):
            if channel.lower() in ["left", "l", "1", "a", "A"]:
                return 0
            if channel.lower() in ["right", "r", "2", "b", "B"]:
                if num_channels < 2:
                    raise Exception(
                        f"Audio has only {num_channels} channels, requested channel {channel}"
                    )
                return 1
            if channel.lower() in ["center", "c"]:
                if num_channels < 3:
                    raise Exception(
                        f"Audio has only {num_channels} channels, requested channel {channel}"
                    )
                return 2
        raise Exception(f"Unknown channel {channel}")

    @staticmethod
    def read_wavspecifier(
        wavspecifier: PathLike,
        scale: float = 1.0,
        time_offset: float = 0.0,
        time_dur: float = 0.0,
        channels_first: bool = True,
        always_2d: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """Read audio from a file path or shell pipe specification.

        Supports pipes as well as every format handled by libsndfile (wav,
        flac, ogg, etc.) together with the extensions listed in :data:`valid_ext`.

        Args:
            wavspecifier (PathLike): Pipe command or audio file path (wav, flac,
                ogg, etc.).
            scale (float): Multiplicative factor applied to the waveform.
            time_offset (float): Start time in seconds relative to the beginning
                of the utterance.
            time_dur (float): Duration in seconds to read. ``0`` reads the audio
                until the end of the utterance.
            channels_first (bool): If ``True`` returns waveforms as
                ``(channels, num_samples)``.
            always_2d (bool): Forces single-channel audio to retain a channel
                dimension.

        Returns:
            Tuple[np.ndarray, int]: Waveform and sampling rate.

        Raises:
            Exception: If the specifier points to an unsupported format.
        """
        wavspecifier = wavspecifier.strip()
        if wavspecifier[-1] == "|":
            wavspecifier = wavspecifier[:-1]
            return AudioReader.read_pipe(
                wavspecifier, scale, time_offset, time_dur, channels_first, always_2d
            )

        ext = os.path.splitext(wavspecifier)[1]
        if ext in valid_ext:
            return AudioReader.read_file(
                wavspecifier, scale, time_offset, time_dur, channels_first, always_2d
            )

        raise Exception("Unknown format for %s" % (wavspecifier))

    @staticmethod
    def read_pipe(
        wavspecifier: PathLike,
        scale: float = 1.0,
        time_offset: float = 0,
        time_dur: float = 0,
        channels_first: bool = True,
        always_2d: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """Read audio produced by a shell pipe command.

        Args:
            wavspecifier (PathLike): Shell command whose stdout returns the
                encoded waveform.
            scale (float): Multiplicative factor applied to the waveform.
            time_offset (float): Start time in seconds relative to the pipe
                output.
            time_dur (float): Duration in seconds to read. ``0`` reads until the
                end of the available samples.
            channels_first (bool): If ``True`` returns waveforms as
                ``(channels, num_samples)``.
            always_2d (bool): Forces single-channel audio to retain a channel
                dimension.

        Returns:
            Tuple[np.ndarray, int]: Waveform and sampling rate.

        Raises:
            Exception: If the pipe command returns a non-zero exit status.
        """
        if wavspecifier[-1] == "|":
            wavspecifier = wavspecifier[:-1]

        proc = subprocess.Popen(wavspecifier, shell=True, stdout=subprocess.PIPE)
        pipe = proc.communicate()[0]
        if proc.returncode != 0:
            raise Exception(
                "Wave read pipe command %s returned code %d"
                % (wavspecifier, proc.returncode)
            )
        x, fs = sf.read(io.BytesIO(pipe), dtype=float_cpu())
        x *= scale
        if time_offset == 0 and time_dur == 0:
            return x, fs

        start_sample = int(math.floor(time_offset * fs))
        num_samples = int(math.floor(time_dur * fs))
        if num_samples == 0:
            return x[start_sample:], fs

        end_sample = start_sample + num_samples
        assert end_sample <= len(x)
        x = x[start_sample:end_sample]
        if always_2d and len(x.shape) == 1:
            x = x[:, np.newaxis]

        if channels_first and len(x.shape) > 1:
            x = x.T

        return x, fs

    @staticmethod
    def read_file_sf(
        wavspecifier: PathLike,
        scale: float = 1.0,
        time_offset: float = 0,
        time_dur: float = 0,
        channels_first: bool = True,
        always_2d: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """Read audio from disk using :mod:`soundfile`.

        Args:
            wavspecifier (PathLike): Audio file path readable by libsndfile.
            scale (float): Multiplicative factor applied to the waveform.
            time_offset (float): Start time in seconds relative to the beginning
                of the recording.
            time_dur (float): Duration in seconds to read. ``0`` reads until the
                end of the recording.
            channels_first (bool): If ``True`` returns waveforms as
                ``(channels, num_samples)``.
            always_2d (bool): Forces single-channel audio to retain a channel
                dimension.

        Returns:
            Tuple[np.ndarray, int]: Waveform and sampling rate.
        """
        if time_offset == 0 and time_dur == 0:
            x, fs = sf.read(wavspecifier, dtype=float_cpu())
            x *= scale
        else:
            with sf.SoundFile(wavspecifier, "r") as f:
                fs = f.samplerate
                start_sample = int(math.floor(time_offset * fs))
                num_samples = int(math.floor(time_dur * fs))
                f.seek(start_sample)
                if num_samples > 0:
                    x = scale * f.read(num_samples, dtype=float_cpu())
                else:
                    x = scale * f.read(dtype=float_cpu())

        if always_2d and len(x.shape) == 1:
            x = x[:, np.newaxis]
        if channels_first and len(x.shape) > 1:
            x = x.T

        return x, fs

    @staticmethod
    def read_file_torchaudio(
        wavspecifier: PathLike,
        scale: float = 1.0,
        time_offset: float = 0,
        time_dur: float = 0,
        channels_first: bool = True,
        always_2d: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """Read audio from disk using :mod:`torchaudio`.

        Args:
            wavspecifier (PathLike): Audio file path readable by torchaudio.
            scale (float): Multiplicative factor applied to the waveform.
            time_offset (float): Start time in seconds relative to the beginning
                of the recording.
            time_dur (float): Duration in seconds to read. ``0`` reads until the
                end of the recording.
            channels_first (bool): If ``True`` returns waveforms as
                ``(channels, num_samples)``.
            always_2d (bool): Forces single-channel audio to retain a channel
                dimension.

        Returns:
            Tuple[np.ndarray, int]: Waveform and sampling rate.
        """
        if time_offset == 0 and time_dur == 0:
            x, fs = torchaudio.load(
                wavspecifier, normalize=True, channels_first=channels_first
            )
        else:
            with torchaudio.backend.sox_io_backend.stream(wavspecifier) as f:
                fs = f.info.samplerate
                start_sample = int(math.floor(time_offset * fs))
                num_samples = int(math.floor(time_dur * fs))
                if num_samples > 0:
                    x = f.read(frame_offset=start_sample, num_frames=num_samples)
                else:
                    x = f.read(frame_offset=start_sample, num_frames=-1)

                if not channels_first:
                    x = x.T

        if not always_2d:
            ch_dim = 0 if channels_first else 1
            if x.shape[ch_dim] == 1:
                x = x.squeeze(ch_dim)

        x = scale * x.numpy().astype(float_cpu())

        return x, fs

    @staticmethod
    def read_file(
        wavspecifier: PathLike,
        scale: float = 1.0,
        time_offset: float = 0,
        time_dur: float = 0,
        channels_first: bool = True,
        always_2d: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """Read audio from disk, retrying with progressively broader fallbacks.

        Args:
            wavspecifier (PathLike): Audio file path.
            scale (float): Multiplicative factor applied to the waveform.
            time_offset (float): Start time in seconds relative to the beginning
                of the recording.
            time_dur (float): Duration in seconds to read. ``0`` reads until the
                end of the recording.
            channels_first (bool): If ``True`` returns waveforms as
                ``(channels, num_samples)``.
            always_2d (bool): Forces single-channel audio to retain a channel
                dimension.

        Returns:
            Tuple[np.ndarray, int]: Waveform and sampling rate.

        Raises:
            RuntimeError: If the audio cannot be read by any backend.

        Notes:
            Attempts to read with :mod:`soundfile` first, including a relaxed
            slicing strategy to recover from libsndfile ``fseek`` issues, and
            finally falls back to :mod:`torchaudio` when required.
        """
        try:
            return AudioReader.read_file_sf(
                wavspecifier, scale, time_offset, time_dur, channels_first, always_2d
            )
        except:
            # some files produce error in the fseek after reading the data,
            # this seems an issue from pysoundfile or soundfile lib itself
            # we try to read from
            # time-offset to the end of the file, and remove the extra frames later,
            # this solves the problem in most cases
            logging.info(
                (
                    "error-1 reading %s offset=%f duration=%f"
                    "retrying reading until end-of-file ..."
                ),
                wavspecifier,
                time_offset,
                time_dur,
            )
            try:
                x, fs = AudioReader.read_file_sf(
                    wavspecifier,
                    scale,
                    time_offset,
                    channels_first=channels_first,
                    always_2d=always_2d,
                )
                if time_dur > 0:
                    num_samples = int(math.floor(time_dur * fs))
                    if channels_first:
                        x = x[..., :num_samples]
                    else:
                        x = x[:num_samples]
                return x, fs
            except:
                logging.info(
                    (
                        "error-2 reading %s offset=%f duration=%f"
                        "retrying reading full file ..."
                    ),
                    wavspecifier,
                    time_offset,
                    time_dur,
                )

                try:
                    x, fs = AudioReader.read_file_sf(
                        wavspecifier,
                        scale,
                        channels_first=channels_first,
                        always_2d=always_2d,
                    )
                    if time_dur > 0:
                        start_sample = int(math.floor(time_offset * fs))
                        num_samples = int(math.floor(time_dur * fs))
                        if channels_first:
                            x = x[..., start_sample : start_sample + num_samples]
                        else:
                            x = x[start_sample : start_sample + num_samples]
                    return x, fs

                except:
                    try:
                        logging.info(
                            (
                                "error-3 reading %s offset=%f duration=%f"
                                "retrying with torchaudio ..."
                            ),
                            wavspecifier,
                            time_offset,
                            time_dur,
                        )
                        x, fs = AudioReader.read_file_torchaudio(
                            wavspecifier,
                            scale,
                            time_offset,
                            time_dur,
                            channels_first,
                            always_2d,
                        )
                        return x, fs
                        # x, fs = torchaudio.load(wavspecifier, channels_first=channels_first)
                        # x = x.numpy().astype(float_cpu()).squeeze(0)
                        # if time_dur > 0:
                        #     start_sample = int(math.floor(time_offset * fs))
                        #     num_samples = int(math.floor(time_dur * fs))
                        #     x = x[:, start_sample : start_sample + num_samples]

                        # return x, fs
                    except RuntimeError as err:
                        logging.info(
                            "fatal error reading %s offset=%f duration=%f",
                            wavspecifier,
                            time_offset,
                            time_dur,
                        )

                    raise err

    def _read_recording(
        self,
        recording: pd.Series,
        time_offset: float = 0,
        time_dur: float = 0,
        channels_first: bool = True,
        always_2d: bool = False,
        return_all_channels: bool = False,
    ) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, Optional[int]]]:
        """Load a recording defined in the recordings table.

        Args:
            recording (pd.Series): Row from the recordings table (requires a
                ``storage_path`` field and optional ``channel``).
            time_offset (float): Start time in seconds relative to the beginning
                of the recording.
            time_dur (float): Duration in seconds to read.
            channels_first (bool): If ``True`` returns waveforms as
                ``(channels, num_samples)``.
            always_2d (bool): Forces single-channel audio to retain a channel
                dimension.
            return_all_channels (bool): If ``True`` returns every channel
                available.

        Returns:
            Tuple[np.ndarray, int]: Waveform and sampling rate when a single
            channel is returned.

            Tuple[np.ndarray, int, Optional[int]]: Waveform, sampling rate, and
            optional selected channel index when ``return_all_channels`` is
            ``True``.
        """
        if return_all_channels:
            always_2d = True

        storage_path = str(recording["storage_path"])
        x_i, fs_i = self.read_wavspecifier(
            storage_path,
            self.wav_scale,
            time_offset,
            time_dur,
            channels_first=channels_first,
            always_2d=always_2d,
        )
        channel_dim = None
        num_channels = 1
        if len(x_i.shape) > 1:
            channel_dim = 0 if channels_first else 1
            num_channels = x_i.shape[channel_dim]

        channel = None
        if "channel" in recording:
            channel = recording["channel"]
            if pd.notna(channel):
                channel = self.channel_name_to_idx(channel, num_channels)

        if channel is None and num_channels == 1:
            channel = 0

        if not return_all_channels and num_channels > 1 and channel is not None:
            x_i = x_i[channel] if channels_first else x_i[:, channel]
            if always_2d:
                x_i = np.expand_dims(x_i, channel_dim)

        if self.resampler is not None:
            target_sample_freq = (
                self.target_sample_freq
                if self.target_sample_freq is not None
                else recording["target_sample_freq"]
            )
            if target_sample_freq is not None and not math.isnan(target_sample_freq):
                if not channels_first and len(x_i.shape) > 1:
                    x_i = x_i.T

                x_i, fs_i = self.resampler(x_i, fs_i, target_sample_freq)
                if not channels_first and len(x_i.shape) > 1:
                    x_i = x_i.T

                # import re
                # f = re.sub(".*/", "", storage_path)
                # f = re.sub(" .*", "", f)
                # sf.write(f"audio_8k/{f}.flac", x_i, fs_i)
                # x_i, fs_i = self.resampler(x_i, fs_i, target_sample_freq)
                # sf.write(f"audio_16k/{f}.flac", x_i, fs_i)
        if return_all_channels:
            return x_i, fs_i, channel

        return x_i, fs_i

    def _read_segment(
        self,
        segment: pd.Series,
        time_offset: float = 0,
        time_dur: float = 0,
        channels_first: bool = True,
        always_2d: bool = False,
        return_all_channels: bool = False,
    ) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, Optional[int]]]:
        """Load a segment defined in the segments table.

        Args:
            segment (pd.Series): Row from the segments table (expects
                ``recording``, ``start``, and ``duration`` fields).
            time_offset (float): Additional start offset in seconds relative to
                the segment start.
            time_dur (float): Duration in seconds to read. ``0`` reads the full
                segment after applying ``time_offset``.
            channels_first (bool): If ``True`` returns waveforms as
                ``(channels, num_samples)``.
            always_2d (bool): Forces single-channel audio to retain a channel
                dimension.
            return_all_channels (bool): If ``True`` returns every channel
                available.

        Returns:
            Tuple[np.ndarray, int]: Waveform and sampling rate when a single
            channel is returned.

            Tuple[np.ndarray, int, Optional[int]]: Waveform, sampling rate, and
            optional selected channel index when ``return_all_channels`` is
            ``True``.
        """
        recording_id = segment["recording"] if "recording" in segment else segment["id"]
        t_start = segment["start"] if "start" in segment else 0.0
        t_start = t_start + time_offset
        if time_dur > 0:
            t_dur = min(time_dur, segment["duration"] - time_offset)
        else:
            t_dur = segment["duration"] - time_offset
        recording = self.recordings.loc[recording_id]
        if "channel" in segment:
            channel = segment["channel"]
            if pd.notna(channel):
                recording["channel"] = channel

        return self._read_recording(
            recording, t_start, t_dur, channels_first, always_2d, return_all_channels
        )

    def read(self, *args: Any, **kwargs: Any) -> Any:
        pass


class SequentialAudioReader(AudioReader):
    """Iterate through recordings or segments sequentially.

    Args:
        dataset (Union[HyperDataset, PathLike, None]): Dataset instance or path to
            a dataset file.
        recordings (Union[RecordingSet, PathLike, None]): Recording table or
            path to a recordings file when ``dataset`` is ``None``.
        segments (Union[SegmentSet, PathLike, None]): Segment table or path to a
            segments file. Must be ``None`` when ``dataset`` is provided.
        wav_scale (float): Multiplicative factor applied to every waveform.
        part_idx (int): Index of the partition to process when splitting the
            dataset.
        num_parts (int): Number of partitions used to split the dataset.
        target_sample_freq (Optional[int]): Target sampling frequency used for
            optional resampling.
        channels_first (bool): If ``True`` returns waveforms as
            ``(channels, num_samples)``.
        always_2d (bool): Forces single-channel audio to retain a channel
            dimension.
        return_all_channels (bool): If ``True`` returns every channel available.

    Attributes:
        dataset (Optional[Union[HyperDataset, PathLike]]): Dataset reference used
            to initialize the reader, when applicable.
        recordings (RecordingSet): Table describing the recordings to load.
        segments (Optional[SegmentSet]): Table with segment definitions when the
            reader operates in segment mode.
        wav_scale (float): Multiplicative factor applied to each waveform.
        part_idx (int): Partition index being processed by this reader.
        num_parts (int): Total number of partitions across which the dataset is
            split.
        target_sample_freq (Optional[int]): Target sampling frequency used for
            optional resampling.
        channels_first (bool): Shape convention for returned waveforms.
        always_2d (bool): Whether mono audio retains a channel axis.
        return_all_channels (bool): Whether to return every channel for
            multi-channel audio.
        cur_item (int): Index of the next item to read.
    """

    def __init__(
        self,
        dataset: Union[HyperDataset, PathLike, None] = None,
        recordings: Union[RecordingSet, PathLike, None] = None,
        segments: Union[SegmentSet, PathLike, None] = None,
        wav_scale: float = 1.0,
        part_idx: int = 1,
        num_parts: int = 1,
        target_sample_freq: Optional[int] = None,
        channels_first: bool = True,
        always_2d: bool = False,
        return_all_channels: bool = False,
    ):
        super().__init__(
            dataset,
            recordings,
            segments,
            wav_scale=wav_scale,
            target_sample_freq=target_sample_freq,
            channels_first=channels_first,
            always_2d=always_2d,
            return_all_channels=return_all_channels,
        )
        self.cur_item: int = 0
        self.part_idx: int = part_idx
        self.num_parts: int = num_parts
        if self.num_parts > 1:
            if self.with_segments:
                self.segments = self.segments.split(self.part_idx, self.num_parts)
            else:
                self.recordings = self.recordings.split(self.part_idx, self.num_parts)

    def __iter__(self) -> "SequentialAudioReader":
        """Return the iterator so the reader can be used in loops.

        Example:
            >>> for key, wav, fs in SequentialAudioReader(recordings=rs):
            ...     process(key, wav, fs)
        """
        return self

    def __next__(self) -> Union[
        Tuple[str, np.ndarray, int],
        Tuple[str, np.ndarray, int, Optional[int]],
    ]:
        """Return the next sequential item.

        Returns:
            Tuple[str, np.ndarray, int]: Key, waveform, and sampling rate when a
            single channel is returned.

            Tuple[str, np.ndarray, int, Optional[int]]: Key, waveform, sampling
            rate, and channel index when ``return_all_channels`` is ``True``.

        Raises:
            StopIteration: When the reader is exhausted.
        """
        data = self.read(1)
        key = data[0]
        if len(key) == 0:
            raise StopIteration

        data = tuple(v[0] for v in data)
        return data

    def next(self) -> Union[
        Tuple[str, np.ndarray, int],
        Tuple[str, np.ndarray, int, Optional[int]],
    ]:
        """Python 2 compatibility alias for :meth:`__next__`."""
        return self.__next__()

    def reset(self) -> None:
        """Reset the internal pointer to the beginning of the dataset."""
        self.cur_item = 0

    def eof(self) -> bool:
        """Check whether all recordings or segments have been consumed.

        Returns:
            bool: ``True`` when the reader has produced every item.
        """
        if self.with_segments:
            return self.cur_item == len(self.segments)
        return self.cur_item == len(self.recordings)

    def read(
        self,
        num_records: int = 0,
        time_offset: Union[float, Sequence[float], np.ndarray] = 0,
        time_durs: Union[float, Sequence[float], np.ndarray] = 0,
    ) -> Union[
        Tuple[List[str], List[np.ndarray], List[int]],
        Tuple[List[str], List[np.ndarray], List[int], List[Optional[int]]],
    ]:
        """Read the next group of recordings or segments.

        Args:
            num_records (int): Number of items to read (``0`` reads the
                remainder of the dataset).
            time_offset (float): Scalar or per-item offsets in seconds to apply
                before reading each item.
            time_durs (float): Scalar or per-item durations in seconds. ``0``
                reads until the end of each item.

        Returns:
            Tuple[List[str], List[np.ndarray], List[int]]: Keys, waveforms, and
            sampling rates when returning a single channel per item.

            Tuple[List[str], List[np.ndarray], List[int], List[Optional[int]]]:
            Keys, waveforms, sampling rates, and channel indices when
            ``return_all_channels`` is ``True``.
        """
        channels_first = self.channels_first
        always_2d = self.always_2d
        return_all_channels = self.return_all_channels
        if num_records == 0:
            if self.with_segments:
                num_records = len(self.segments) - self.cur_item
            else:
                num_records = len(self.recordings) - self.cur_item

        offset_is_list = isinstance(time_offset, (list, np.ndarray))
        dur_is_list = isinstance(time_durs, (list, np.ndarray))

        keys = []
        x = []
        fs = []
        channel = []
        for i in range(num_records):
            if self.eof():
                break

            offset_i = time_offset[i] if offset_is_list else time_offset
            dur_i = time_durs[i] if dur_is_list else time_durs

            if self.with_segments:
                segment = self.segments.iloc[self.cur_item]
                key = segment["id"]
                data_i = self._read_segment(
                    segment,
                    offset_i,
                    dur_i,
                    channels_first=channels_first,
                    always_2d=always_2d,
                    return_all_channels=return_all_channels,
                )
            else:
                recording = self.recordings.iloc[self.cur_item]
                key = recording["id"]
                data_i = self._read_recording(
                    recording,
                    offset_i,
                    dur_i,
                    channels_first=channels_first,
                    always_2d=always_2d,
                    return_all_channels=return_all_channels,
                )

            if return_all_channels:
                x_i, fs_i, channel_i = data_i
                channel.append(channel_i)
            else:
                x_i, fs_i = data_i

            keys.append(key)
            x.append(x_i)
            fs.append(fs_i)
            self.cur_item += 1

        if return_all_channels:
            return keys, x, fs, channel

        return keys, x, fs

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Select keyword arguments relevant to the sequential reader.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dict[str, Any]: Subset containing only recognized reader arguments.
        """
        valid_args = ("part_idx", "num_parts", "wav_scale", "target_sample_freq")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Register command-line arguments for :class:`SequentialAudioReader`.

        Args:
            parser (ArgumentParser): Parser where arguments are to be added.
            prefix (Optional[str]): Optional prefix to nest arguments under a
                group.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--wav-scale",
            default=1.0,
            # default=2 ** 15 - 1,
            type=float,
            help=("multiplicative factor for waveform"),
        )
        try:
            parser.add_argument(
                "--part-idx",
                type=int,
                default=1,
                help=("splits the list of files into num-parts and processes part-idx"),
            )
            parser.add_argument(
                "--num-parts",
                type=int,
                default=1,
                help=("splits the list of files into num-parts and processes part-idx"),
            )
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    add_argparse_args = add_class_args


class RandomAccessAudioReader(AudioReader):
    """Provide random access to recordings or segments on demand.

    Args:
        dataset (Union[HyperDataset, PathLike, None]): Dataset instance or path to
            a dataset file.
        recordings (Union[RecordingSet, PathLike, None]): Recording table or
            path to a recordings file when ``dataset`` is ``None``.
        segments (Union[SegmentSet, PathLike, None]): Segment table or path to a
            segments file. Must be ``None`` when ``dataset`` is provided.
        wav_scale (float): Multiplicative factor applied to every waveform.
        target_sample_freq (Optional[int]): Target sampling frequency used for
            optional resampling.
        channels_first (bool): If ``True`` returns waveforms as
            ``(channels, num_samples)``.
        always_2d (bool): Forces single-channel audio to retain a channel
            dimension.
        return_all_channels (bool): If ``True`` returns every channel available.

    Attributes:
        dataset (Optional[Union[HyperDataset, PathLike]]): Dataset reference used
            to initialize the reader, when applicable.
        recordings (RecordingSet): Table describing the recordings to load.
        segments (Optional[SegmentSet]): Table with segment definitions when the
            reader operates in segment mode.
        wav_scale (float): Multiplicative factor applied to each waveform.
        target_sample_freq (Optional[int]): Target sampling frequency used for
            optional resampling.
        channels_first (bool): Shape convention for returned waveforms.
        always_2d (bool): Whether mono audio retains a channel axis.
        return_all_channels (bool): Whether queries return every channel rather
            than a single mixdown.
    """

    def __init__(
        self,
        dataset: Union[HyperDataset, PathLike, None] = None,
        recordings: Union[RecordingSet, PathLike, None] = None,
        segments: Union[SegmentSet, PathLike, None] = None,
        wav_scale: float = 1.0,
        target_sample_freq: Optional[int] = None,
        channels_first: bool = True,
        always_2d: bool = False,
        return_all_channels: bool = False,
    ):
        super().__init__(
            dataset,
            recordings,
            segments,
            wav_scale=wav_scale,
            target_sample_freq=target_sample_freq,
            channels_first=channels_first,
            always_2d=always_2d,
            return_all_channels=return_all_channels,
        )

    def read(
        self,
        keys: Union[str, Sequence[str]],
        time_offset: Union[float, Sequence[float], np.ndarray] = 0,
        time_durs: Union[float, Sequence[float], np.ndarray] = 0,
        channels_first: bool = True,
        always_2d: bool = False,
        return_all_channels: bool = False,
    ) -> Union[
        Tuple[List[np.ndarray], List[int]],
        Tuple[List[np.ndarray], List[int], List[Optional[int]]],
    ]:
        """Fetch the waveforms for the requested keys.

        Args:
            keys (Union[str, Sequence[str]]): Recording or segment identifiers.
            time_offset (float): Scalar or per-item offsets in seconds.
            time_durs (float): Scalar or per-item durations in seconds. ``0``
                reads until the end of each item.
            channels_first (bool): If ``True`` returns waveforms as
                ``(channels, num_samples)``.
            always_2d (bool): Forces single-channel audio to retain a channel
                dimension.
            return_all_channels (bool): If ``True`` returns every channel
                available.

        Returns:
            Tuple[List[np.ndarray], List[int]]: Waveforms and sampling rates
            when returning a single channel per item.

            Tuple[List[np.ndarray], List[int], List[Optional[int]]]: Waveforms,
            sampling rates, and channel indices when ``return_all_channels`` is
            ``True``.

        Raises:
            Exception: If a requested key is not found.
        """
        channels_first = self.channels_first
        always_2d = self.always_2d
        return_all_channels = self.return_all_channels
        if isinstance(keys, str):
            keys = [keys]

        offset_is_list = isinstance(time_offset, (list, np.ndarray))
        dur_is_list = isinstance(time_durs, (list, np.ndarray))

        x = []
        fs = []
        channel = []
        for i, key in enumerate(keys):

            offset_i = time_offset[i] if offset_is_list else time_offset
            dur_i = time_durs[i] if dur_is_list else time_durs

            if self.with_segments:
                if not (key in self.segments.index):
                    raise Exception("Key %s not found" % key)

                segment = self.segments.loc[key]
                data_i = self._read_segment(
                    segment,
                    offset_i,
                    dur_i,
                    channels_first=channels_first,
                    always_2d=always_2d,
                    return_all_channels=return_all_channels,
                )
            else:
                if not (key in self.recordings.index):
                    raise Exception("Key %s not found" % key)

                recording = self.recordings.loc[key]
                data_i = self._read_recording(
                    recording,
                    offset_i,
                    dur_i,
                    channels_first=channels_first,
                    always_2d=always_2d,
                    return_all_channels=return_all_channels,
                )

            if return_all_channels:
                x_i, fs_i, channel_i = data_i
                channel.append(channel_i)
            else:
                x_i, fs_i = data_i

            x.append(x_i)
            fs.append(fs_i)

        if return_all_channels:
            return x, fs, channel

        return x, fs

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Select keyword arguments relevant to the random-access reader.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dict[str, Any]: Subset containing only recognized reader arguments.
        """
        valid_args = ("wav_scale",)
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Register command-line arguments for :class:`RandomAccessAudioReader`.

        Args:
            parser (ArgumentParser): Parser where arguments are to be added.
            prefix (Optional[str]): Optional prefix to nest arguments under a
                group.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--wav-scale",
            default=1.0,
            type=float,
            help=("multiplicative factor for waveform"),
        )
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    add_argparse_args = add_class_args
