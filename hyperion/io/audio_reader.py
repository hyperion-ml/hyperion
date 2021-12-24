"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os
import logging
import io
import math
import subprocess
import soundfile as sf

import numpy as np

from ..hyp_defs import float_cpu
from ..utils import SCPList, SegmentList

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
    "w64",
    ".wve",
    ".xi",
]


class AudioReader(object):
    """Class to read audio files from wav, flac or pipe

    Attributes:
         file_path:     scp file with formant file_key wavspecifier (audio_file/pipe) or SCPList object.
         segments_path: segments file with format: segment_id file_id tbeg tend
         wav_scale:     multiplies signal by scale factor
    """

    def __init__(self, file_path, segments_path=None, wav_scale=2 ** 15 - 1):
        self.file_path = file_path
        if isinstance(file_path, SCPList):
            self.scp = file_path
        else:
            self.scp = SCPList.load(file_path, sep=" ", is_wav=True)

        self.segments_path = segments_path
        if segments_path is None:
            self.segments = None
            self.with_segments = False
        else:
            self.with_segments = True
            if isinstance(file_path, SegmentList):
                self.segments = segments_path
            else:
                self.segments = SegmentList.load(
                    segments_path, sep=" ", index_by_file=False
                )

        self.wav_scale = wav_scale

    @property
    def keys(self):
        if self.with_segments:
            return np.asarray(self.segments["segment_id"])
        return self.scp.key

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
    def read_wavspecifier(wavspecifier, scale=2 ** 15, time_offset=0, time_dur=0):
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
            x, fs = AudioReader.read_pipe(wavspecifier, scale)
            if time_offset == 0 and time_dur == 0:
                return x, fs

            start_sample = int(math.floor(time_offset * fs))
            num_samples = int(math.floor(time_dur * fs))
            if num_samples == 0:
                return x[start_sample:], fs

            end_sample = start_sample + num_samples
            assert end_sample <= len(x)
            return x[start_sample:end_sample], fs

        ext = os.path.splitext(wavspecifier)[1]
        if ext in valid_ext:
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

        raise Exception("Unknown format for %s" % (wavspecifier))

    @staticmethod
    def read_pipe(wavspecifier, scale=2 ** 15):
        """Reads wave file from a pipe
        Args:
          wavspecifier: Shell command with pipe output
          scale:        Multiplies signal by scale factor
        """
        # proc = subprocess.Popen(wavspecifier, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        proc = subprocess.Popen(wavspecifier, shell=True, stdout=subprocess.PIPE)
        pipe = proc.communicate()[0]
        if proc.returncode != 0:
            raise Exception(
                "Wave read pipe command %s returned code %d"
                % (wavspecifier, proc.returncode)
            )
        x, fs = sf.read(io.BytesIO(pipe), dtype=float_cpu())
        x *= scale
        return x, fs

    def _read_segment(self, segment, time_offset=0, time_dur=0):
        """Reads a wave segment

        Args:
          segment: pandas DataFrame (segment_id , file_id, tbeg, tend)
        Returns:
          Wave, sampling frequency
        """
        file_id = segment["file_id"]
        t_beg = segment["tbeg"] + time_offset
        t_end = segment["tend"]
        if time_dur > 0:
            t_end_new = t_beg + time_dur
            assert t_end_new <= t_end
            t_end = t_end_new

        file_path, _, _ = self.scp[file_id]
        x_i, fs_i = self.read_wavspecifier(file_path, self.wav_scale)
        num_samples_i = len(x_i)
        s_beg = int(t_beg * fs_i)
        if s_beg >= num_samples_i:
            raise Exception(
                "segment %s tbeg=%.2f (num_sample=%d) longer that wav file %s (num_samples=%d)"
                % (key, tbeg, sbeg, file_id, num_samples_i)
            )

        s_end = int(t_end * fs_i)
        if s_end > num_samples_i or t_end < 0:
            s_end = num_samples_i

        x_i = x_i[s_beg:s_end]
        return x_i, fs_i

    def read(self):
        pass


class SequentialAudioReader(AudioReader):
    def __init__(
        self,
        file_path,
        segments_path=None,
        wav_scale=2 ** 15 - 1,
        part_idx=1,
        num_parts=1,
    ):
        super().__init__(file_path, segments_path, wav_scale=wav_scale)
        self.cur_item = 0
        self.part_idx = part_idx
        self.num_parts = num_parts
        if self.num_parts > 1:
            if self.with_segments:
                self.segments = self.segments.split(self.part_idx, self.num_parts)
            else:
                self.scp = self.scp.split(
                    self.part_idx, self.num_parts, group_by_key=False
                )

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
        return self.cur_item == len(self.scp)

    def read(self, num_records=0, time_offset=0, time_durs=0):
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
                num_records = len(self.scp) - self.cur_item

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
                segment = self.segments[self.cur_item]
                key = segment["segment_id"]
                x_i, fs_i = self._read_segment(segment, offset_i, dur_i)
            else:
                key, file_path, _, _ = self.scp[self.cur_item]
                x_i, fs_i = self.read_wavspecifier(
                    file_path, self.wav_scale, offset_i, dur_i
                )

            keys.append(key)
            data.append(x_i)
            fs.append(fs_i)
            self.cur_item += 1

        return keys, data, fs

    @staticmethod
    def filter_args(**kwargs):
        valid_args = ("part_idx", "num_parts", "wav_scale")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
        else:
            p1 = "--" + prefix + "."

        parser.add_argument(
            p1 + "wav-scale",
            default=2 ** 15 - 1,
            type=float,
            help=("multiplicative factor for waveform"),
        )
        try:
            parser.add_argument(
                p1 + "part-idx",
                type=int,
                default=1,
                help=(
                    "splits the list of files into num-parts and " "processes part-idx"
                ),
            )
            parser.add_argument(
                p1 + "num-parts",
                type=int,
                default=1,
                help=(
                    "splits the list of files into num-parts and " "processes part-idx"
                ),
            )
        except:
            pass

    add_argparse_args = add_class_args


class RandomAccessAudioReader(AudioReader):
    def __init__(self, file_path, segments_path=None, wav_scale=2 ** 15 - 1):
        super().__init__(file_path, segments_path, wav_scale)

    def _read(self, keys, time_offset=0, time_durs=0):
        """Reads the waveforms  for the recordings in keys.

        Args:
          keys: List of recording/segment_ids names.

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
                if not (key in self.segments):
                    raise Exception("Key %s not found" % key)

                segment = self.segments[key]
                x_i, fs_i = self._read_segment(segment, offset_i, dur_i)
            else:
                if not (key in self.scp):
                    raise Exception("Key %s not found" % key)

                file_path, _, _ = self.scp[key]
                x_i, fs_i = self.read_wavspecifier(
                    file_path, self.wav_scale, offset_i, dur_i
                )

            data.append(x_i)
            fs.append(fs_i)

        return data, fs

    def read(self, keys, time_offset=0, time_durs=0):
        """Reads the waveforms  for the recordings in keys.

        Args:
          keys: List of recording/segment_ids names.

        Returns:
          data: List of waveforms
          fs: List of sampling freq.
        """
        try:
            x, fs = self._read(keys, time_offset=time_offset, time_durs=time_durs)
        except:
            if isinstance(keys, str):
                keys = [keys]

            if not isinstance(time_offset, (list, np.ndarray)):
                time_offset = [time_offset] * len(keys)
            if not isinstance(time_durs, (list, np.ndarray)):
                time_durs = [time_durs] * len(keys)

            try:
                # some files produce error in the fseek after reading the data,
                # this seems an issue from pysoundfile or soundfile lib itself
                # we try to read from
                # time-offset to the end of the file, and remove the extra frames later,
                # this solves the problem in most cases
                logging.info(
                    (
                        "error-1 reading at keys={} offset={} "
                        "retrying reading until end-of-file ..."
                    ).format(keys, time_offset)
                )
                x, fs = self._read(keys, time_offset=time_offset)
                for i in range(len(x)):
                    end_sample = int(time_durs[i] * fs[i])
                    x[i] = x[i][:end_sample]
            except:
                # try to read the full file
                logging.info(
                    (
                        "error-2 reading at key={}, " "retrying reading full file ..."
                    ).format(keys)
                )
                x, fs = self._read(keys)
                for i in range(len(x)):
                    start_sample = int(time_offset[i] * fs[i])
                    end_sample = start_sample + int(time_durs[i] * fs[i])
                    x[i] = x[i][start_sample:end_sample]

        return x, fs

    @staticmethod
    def filter_args(**kwargs):
        valid_args = ("wav_scale",)
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
        else:
            p1 = "--" + prefix + "."

        parser.add_argument(
            p1 + "wav-scale",
            default=2 ** 15 - 1,
            type=float,
            help=("multiplicative factor for waveform"),
        )

    add_argparse_args = add_class_args
