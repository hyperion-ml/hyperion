"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import io
import logging
import math
import os
import subprocess
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ..hyp_defs import float_cpu
from ..np.preprocessing.resampler import Any2AnyFreqResampler
from ..utils import HypDataset, PathLike, RecordingSet, SegmentSet

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
    """Class to read audio files from wav, flac or pipe

    This class recives HypDataset or RecordingSet,
    When reciving RecordingSet, it can also recive a SegmentSet

    Attributes:
        dataset:    HypDataset or file path to HypDataset
        recordings: RecordingSet or file path to RecordingSet
        segments:   SegmentSet or file path to SegmentSet
        wav_scale:     multiplies signal by scale factor
        target_sample_freq: All audios are resample this sample freq.
        channels_first:  If True, the returned waveforms have shape (num_channels, num_samples)
                            If False, the returned waveforms have shape (num_samples, num_channels)
        always_2d: If True, the returned waveforms have shape (num_samples, num_channels)
                        even when num_channels=1
        return_all_channels: If True, returns all channels in multi-channel audio files
    """

    def __init__(
        self,
        dataset: Union[HypDataset, PathLike, None] = None,
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
            if not isinstance(dataset, HypDataset):
                dataset = HypDataset.load(dataset)

            recordings = dataset.recordings(keep_loaded=False)
            segments = dataset.segments(keep_loaded=False)
        else:
            if not isinstance(recordings, RecordingSet):
                recordings = RecordingSet.load(recordings)

            if segments is not None:
                if not isinstance(segments, SegmentSet):
                    segments = SegmentSet.load(segments)

        self.recordings = recordings
        self.segments = segments
        self.with_segments = False if segments is None else True
        self.wav_scale = wav_scale
        self.target_sample_freq = target_sample_freq
        self.channels_first = channels_first
        self.always_2d = always_2d
        self.return_all_channels = return_all_channels
        self.resampler = None
        if self.target_sample_freq is not None or "target_sample_freq" in recordings:
            self.resampler = Any2AnyFreqResampler()

    @property
    def keys(self):
        if self.with_segments:
            return self.segments["id"].values
        return self.recordings["id"].values

    def __enter__(self):
        """Function required when entering contructions of type

        with AudioReader('file.h5') as f:
           keys, data = f.read()
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Function required when exiting from contructions of type

        with AudioReader('file.h5') as f:
           keys, data = f.read()
        """
        pass

    @staticmethod
    def channel_name_to_idx(channel, num_channels):
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
    ):
        """Reads an audiospecifier (audio_file/pipe)
           It reads from pipe or from all the files that can be read
           by `libsndfile <http://www.mega-nerd.com/libsndfile/#Features>`

        Args:
          wavspecifier: A pipe, wav, flac, ogg file etc.
          scale:        Multiplies signal by scale factor
          time_offset: float indicating the start time to read in the utterance.
          time_durs: floats indicating the number of seconds to read from the utterance,
                     if 0 it reads untils the end

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
    ):
        """Reads wave file from a pipe
        Args:
          wavspecifier: Shell command with pipe output
          scale:        Multiplies signal by scale factor
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
    ):
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
    ):
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
                    x = f.read(num_frames=num_samples)
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
    ):
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
    ):
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
                x_i = x_i.expand_dims(channel_dim)

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
        self, segment: pd.Series, time_offset: float = 0, time_dur: float = 0
    ):
        """Reads a wave segment

        Args:
          segment: pandas DataFrame (segment_id , file_id, tbeg, tend)
        Returns:
          Wave, sampling frequency
        """
        recording_id = segment["recording"] if "recording" in segment else segment["id"]
        t_start = segment["start"] if "start" in segment else 0.0
        t_start = t_start + time_offset
        t_dur = segment["duration"]
        recording = self.recordings.loc[recording_id]
        if "channel" in segment:
            channel = segment["channel"]
            if pd.notna(channel):
                recording["channel"] = channel

        return self._read_recording(recording, t_start, t_dur)

    def read(self):
        pass


class SequentialAudioReader(AudioReader):
    """
    Class to read audio files sequentially from wav, flac or pipe
    This class recives HypDataset or RecordingSet,
    When reciving RecordingSet, it can also recive a SegmentSet
    Attributes:
        dataset:    HypDataset or file path to HypDataset
        recordings: RecordingSet or file path to RecordingSet
        segments:   SegmentSet or file path to SegmentSet
        wav_scale:     multiplies signal by scale factor
        part_idx:      splits the list of files into num_parts and processes part_idx
        num_parts:     splits the list of files into num_parts and processes part_idx
        target_sample_freq: All audios are resample this sample freq.
        channels_first:  If True, the returned waveforms have shape (num_channels, num_samples)
                            If False, the returned waveforms have shape (num_samples, num_channels)
        always_2d: If True, the returned waveforms have shape (num_samples, num_channels)
                        even when num_channels=1
        return_all_channels: If True, returns all channels in multi-channel audio files
    """

    def __init__(
        self,
        dataset: Union[HypDataset, PathLike, None] = None,
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
        self.cur_item = 0
        self.part_idx = part_idx
        self.num_parts = num_parts
        if self.num_parts > 1:
            if self.with_segments:
                self.segments = self.segments.split(self.part_idx, self.num_parts)
            else:
                self.recordings = self.recordings.split(self.part_idx, self.num_parts)

    def __iter__(self):
        """Needed to build an iterator, e.g.:
        r = SequentialAudioReader(...)
        for key, s, fs in r:
           print(key)
           process(s)
        """
        return self

    def __next__(self):
        """Needed to build an iterator, e.g.:
        r = SequentialAudioReader(...)
        for key , s, fs in r:
           process(s)
        """
        data = self.read(1)
        key = data[0]
        if len(key) == 0:
            raise StopIteration

        data = tuple(v[0] for v in data)
        return data
        # if self.return_all_channels:
        #     key, x, fs, channel = self.read(1)
        #     if len(key) == 0:
        #         raise StopIteration
        #     return key[0], x[0], fs[0], channel[0]
        # else:
        #     key, x, fs = self.read(1)
        #     if len(key) == 0:
        #         raise StopIteration
        #     return key[0], x[0], fs[0]

    def next(self):
        """__next__ for Python 2"""
        return self.__next__()

    def reset(self):
        """Returns the file pointer to the begining of the dataset,
        then we can start reading the features again.
        """
        self.cur_item = 0

    def eof(self):
        """End of file.

        Returns:
          True, when we have read all the recordings in the dataset.
        """
        if self.with_segments:
            return self.cur_item == len(self.segments)
        return self.cur_item == len(self.recordings)

    def read(
        self,
        num_records: int = 0,
        time_offset: float = 0,
        time_durs: float = 0,
    ):
        """Reads next num_records audio files

        Args:
          num_records: Number of audio files to read.
          time_offset: List of floats indicating the start time to read in the utterance.
          time_durs: List of floats indicating the number of seconds to read from each utterance

        Returns:
          key: List of recording names.
          data: List of waveforms
          fs: list of sample freqs
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
    def filter_args(**kwargs):
        valid_args = ("part_idx", "num_parts", "wav_scale", "target_sample_freq")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix: Optional[str] = None):
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
    """
    Class to read audio files randomly from wav, flac or pipe
    This class recives HypDataset or RecordingSet,
    When reciving RecordingSet, it can also recive a SegmentSet
    Attributes:
        dataset:    HypDataset or file path to HypDataset
        recordings: RecordingSet or file path to RecordingSet
        segments:   SegmentSet or file path to SegmentSet
        wav_scale:     multiplies signal by scale factor
        target_sample_freq: All audios are resample this sample freq.
        channels_first:  If True, the returned waveforms have shape (num_channels, num_samples)
                            If False, the returned waveforms have shape (num_samples, num_channels)
        always_2d: If True, the returned waveforms have shape (num_samples, num_channels)
                        even when num_channels=1
        return_all_channels: If True, returns all channels in multi-channel audio files
    """

    def __init__(
        self,
        dataset: Union[HypDataset, PathLike, None] = None,
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
        keys: Union[str, List, np.array],
        time_offset: float = 0,
        time_durs: float = 0,
        channels_first: bool = True,
        always_2d: bool = False,
        return_all_channels: bool = False,
    ):
        """Reads the waveforms  for the recordings in keys.

        Args:
          keys: List of recording/segment_ids names.
          time_offset: float or float list with time-offsets
          time_durs: float or float list with durations

        Returns:
          data: List of waveforms
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

                # file_path = self.recordings.loc[key, "storage_path"]
                # data_i = self.read_wavspecifier(
                #     file_path,
                #     self.wav_scale,
                #     offset_i,
                #     dur_i,
                #     channels_first=channels_first,
                #     always_2d=always_2d,
                #     return_all_channels=return_all_channels,
                # )
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
    def filter_args(**kwargs):
        valid_args = ("wav_scale",)
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix: Optional[str] = None):
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
