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
]


class AudioReader(object):
    """Class to read audio files from wav, flac or pipe

    This class recives HypDataset or RecordingSet,
    When reciving RecordingSet, it can also recive a SegmentSet

    Attributes:
         dataset:    HypDataset or file path to HypDataset
         recordings: RecordingSet or file path to RecordingSet
         segments:   SegmentSet or file path to SegmentSet
         wav_scale:     multiplies signal by scale factor
         target_sample_freq: All audios are resample this sample freq.
    """

    def __init__(
        self,
        dataset: Union[HypDataset, PathLike, None] = None,
        recordings: Union[RecordingSet, PathLike, None] = None,
        segments: Union[SegmentSet, PathLike, None] = None,
        wav_scale: float = 1.0,
        target_sample_freq: Optional[int] = None,
    ):
        assert (dataset is None) != (
            recordings is None
        ), "dataset xor recordings must be given"
        assert (segments is None) or (
            dataset is None
        ), "if dataset is given, segments must be None"
        self.with_segments = False
        if dataset is not None:
            if not isinstance(dataset, HypDataset):
                dataset = HypDataset.load(dataset)

            recordings = dataset.recordings(keep_loaded=False)
            segments = dataset.segments(keep_loaded=False)
            if segments.has_time_marks:
                self.with_segments = True
            else:
                segments = None
        elif not isinstance(recordings, RecordingSet):
            recordings = RecordingSet.load(recordings)
            if segments is not None:
                self.with_segments = True
                if not isinstance(segments, SegmentSet):
                    segments = SegmentSet.load(segments)

        self.recordings = recordings
        self.segments = segments
        self.wav_scale = wav_scale
        self.target_sample_freq = target_sample_freq
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
    def read_wavspecifier(
        wavspecifier: PathLike,
        scale: float = 1.0,
        time_offset: float = 0.0,
        time_dur: float = 0.0,
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
            return AudioReader.read_pipe(wavspecifier, scale, time_offset, time_dur)

        ext = os.path.splitext(wavspecifier)[1]
        if ext in valid_ext:
            return AudioReader.read_file(wavspecifier, scale, time_offset, time_dur)

        raise Exception("Unknown format for %s" % (wavspecifier))

    @staticmethod
    def read_pipe(
        wavspecifier: PathLike,
        scale: float = 1.0,
        time_offset: float = 0,
        time_dur: float = 0,
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
        return x[start_sample:end_sample], fs

    @staticmethod
    def read_file_sf(
        wavspecifier: PathLike,
        scale: float = 1.0,
        time_offset: float = 0,
        time_dur: float = 0,
    ):
        if time_offset == 0 and time_dur == 0:
            x, fs = sf.read(wavspecifier, dtype=float_cpu())
            x *= scale
            return x, fs

        with sf.SoundFile(wavspecifier, "r") as f:
            fs = f.samplerate
            start_sample = int(math.floor(time_offset * fs))
            num_samples = int(math.floor(time_dur * fs))
            f.seek(start_sample)
            if num_samples > 0:
                x = scale * f.read(num_samples, dtype=float_cpu())
            else:
                x = scale * f.read(dtype=float_cpu())

            return x, fs

    @staticmethod
    def read_file(
        wavspecifier: PathLike,
        scale: float = 1.0,
        time_offset: float = 0,
        time_dur: float = 0,
    ):
        try:
            return AudioReader.read_file_sf(wavspecifier, scale, time_offset, time_dur)
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
                x, fs = AudioReader.read_file_sf(wavspecifier, scale, time_offset)
                num_samples = int(math.floor(time_dur * fs))
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
                    x, fs = AudioReader.read_file_sf(wavspecifier, scale)
                    start_sample = int(math.floor(time_offset * fs))
                    num_samples = int(math.floor(time_dur * fs))
                    x = x[start_sample : start_sample + num_samples]
                    return x, fs
                except RuntimeError as err:
                    logging.info(
                        "fatal error reading %s offset=%f duration=%f",
                        wavspecifier,
                        time_offset,
                        time_dur,
                    )
                    print(
                        "fatal error reading %s offset=%f duration=%f"
                        % (wavspecifier, time_offset, time_dur),
                        flush=True,
                    )
                    raise err

    def _read_recording(
        self, recording: pd.Series, time_offset: float = 0, time_dur: float = 0
    ):
        storage_path = recording["storage_path"]
        x_i, fs_i = self.read_wavspecifier(
            storage_path,
            self.wav_scale,
            time_offset,
            time_dur,
        )

        if self.resampler is not None:
            target_sample_freq = (
                self.target_sample_freq
                if self.target_sample_freq is not None
                else recording["target_sample_freq"]
            )
            if target_sample_freq is not None:
                x_i, fs_i = self.resampler(x_i, fs_i, target_sample_freq)
                # import re
                # f = re.sub(".*/", "", storage_path)
                # f = re.sub(" .*", "", f)
                # sf.write(f"audio_8k/{f}.flac", x_i, fs_i)
                # x_i, fs_i = self.resampler(x_i, fs_i, target_sample_freq)
                # sf.write(f"audio_16k/{f}.flac", x_i, fs_i)

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
        recording_id = segment["recording"]
        t_start = segment["start"] + time_offset
        t_dur = segment["duration"]
        recording = self.recordings.loc[recording_id]
        return self._read_recording(recording, t_start, t_dur)
        # storage_path = self.recordings.loc[recording_id, "storage_path"]
        # x_i, fs_i = self.read_wavspecifier(storage_path, self.wav_scale, t_start, t_dur)

        return x_i, fs_i

    def read(self):
        pass


class SequentialAudioReader(AudioReader):
    def __init__(
        self,
        dataset: Union[HypDataset, PathLike, None] = None,
        recordings: Union[RecordingSet, PathLike, None] = None,
        segments: Union[SegmentSet, PathLike, None] = None,
        wav_scale: float = 1.0,
        part_idx: int = 1,
        num_parts: int = 1,
        target_sample_freq: Optional[int] = None,
    ):
        super().__init__(
            dataset,
            recordings,
            segments,
            wav_scale=wav_scale,
            target_sample_freq=target_sample_freq,
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
        key, x, fs = self.read(1)
        if len(key) == 0:
            raise StopIteration
        return key[0], x[0], fs[0]

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

    def read(self, num_records: int = 0, time_offset: float = 0, time_durs: float = 0):
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
        if num_records == 0:
            if self.with_segments:
                num_records = len(self.segments) - self.cur_item
            else:
                num_records = len(self.recordings) - self.cur_item

        offset_is_list = isinstance(time_offset, (list, np.ndarray))
        dur_is_list = isinstance(time_durs, (list, np.ndarray))

        keys = []
        data = []
        fs = []
        for i in range(num_records):
            if self.eof():
                break

            offset_i = time_offset[i] if offset_is_list else time_offset
            dur_i = time_durs[i] if dur_is_list else time_durs

            if self.with_segments:
                segment = self.segments.iloc[self.cur_item]
                key = segment["id"]
                x_i, fs_i = self._read_segment(segment, offset_i, dur_i)
            else:
                recording = self.recordings.iloc[self.cur_item]
                key = recording["id"]
                x_i, fs_i = self._read_recording(recording, offset_i, dur_i)
                # file_path = segment["storage_path"]
                # x_i, fs_i = self.read_wavspecifier(
                #     file_path, self.wav_scale, offset_i, dur_i
                # )

            keys.append(key)
            data.append(x_i)
            fs.append(fs_i)
            self.cur_item += 1

        return keys, data, fs

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
                help=(
                    "splits the list of files into num-parts and " "processes part-idx"
                ),
            )
            parser.add_argument(
                "--num-parts",
                type=int,
                default=1,
                help=(
                    "splits the list of files into num-parts and " "processes part-idx"
                ),
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
    def __init__(
        self,
        dataset: Union[HypDataset, PathLike, None] = None,
        recordings: Union[RecordingSet, PathLike, None] = None,
        segments: Union[SegmentSet, PathLike, None] = None,
        wav_scale: float = 1.0,
        target_sample_freq: Optional[int] = None,
    ):
        super().__init__(
            dataset,
            recordings,
            segments,
            wav_scale=wav_scale,
            target_sample_freq=target_sample_freq,
        )

    def read(
        self,
        keys: Union[str, List, np.array],
        time_offset: float = 0,
        time_durs: float = 0,
    ):
        """Reads the waveforms  for the recordings in keys.

        Args:
          keys: List of recording/segment_ids names.
          time_offset: float or float list with time-offsets
          time_durs: float or float list with durations

        Returns:
          data: List of waveforms
        """
        if isinstance(keys, str):
            keys = [keys]

        offset_is_list = isinstance(time_offset, (list, np.ndarray))
        dur_is_list = isinstance(time_durs, (list, np.ndarray))

        data = []
        fs = []
        for i, key in enumerate(keys):

            offset_i = time_offset[i] if offset_is_list else time_offset
            dur_i = time_durs[i] if dur_is_list else time_durs

            if self.with_segments:
                if not (key in self.segments.index):
                    raise Exception("Key %s not found" % key)

                segment = self.segments.loc[key]
                x_i, fs_i = self._read_segment(segment, offset_i, dur_i)
            else:
                if not (key in self.recordings.index):
                    raise Exception("Key %s not found" % key)

                file_path = self.recordings.loc[key, "storage_path"]
                x_i, fs_i = self.read_wavspecifier(
                    file_path, self.wav_scale, offset_i, dur_i
                )

            data.append(x_i)
            fs.append(fs_i)

        return data, fs

    # def read(self, keys, time_offset=0, time_durs=0):
    #     """Reads the waveforms  for the recordings in keys.

    #     Args:
    #       keys: List of recording/segment_ids names.

    #     Returns:
    #       data: List of waveforms
    #       fs: List of sampling freq.
    #     """
    #     try:
    #         x, fs = self._read(keys, time_offset=time_offset, time_durs=time_durs)
    #     except:
    #         if isinstance(keys, str):
    #             keys = [keys]

    #         if not isinstance(time_offset, (list, np.ndarray)):
    #             time_offset = [time_offset] * len(keys)
    #         if not isinstance(time_durs, (list, np.ndarray)):
    #             time_durs = [time_durs] * len(keys)

    #         try:
    #             logging.info(
    #                 (
    #                     "error-1 reading at keys={} offset={} "
    #                     "retrying reading until end-of-file ..."
    #                 ).format(keys, time_offset)
    #             )
    #             x, fs = self._read(keys, time_offset=time_offset)
    #             for i in range(len(x)):
    #                 end_sample = int(time_durs[i] * fs[i])
    #                 x[i] = x[i][:end_sample]
    #         except:
    #             # try to read the full file
    #             logging.info(
    #                 (
    #                     "error-2 reading at key={}, " "retrying reading full file ..."
    #                 ).format(keys)
    #             )
    #             x, fs = self._read(keys)
    #             for i in range(len(x)):
    #                 start_sample = int(time_offset[i] * fs[i])
    #                 end_sample = start_sample + int(time_durs[i] * fs[i])
    #                 x[i] = x[i][start_sample:end_sample]

    #     return x, fs

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
            # default=2 ** 15 - 1,
            type=float,
            help=("multiplicative factor for waveform"),
        )
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    add_argparse_args = add_class_args
