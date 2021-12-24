"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import os
import re
import soundfile as sf

import numpy as np

from ..utils.kaldi_io_funcs import is_token
from .audio_reader import valid_ext
from .audio_writer import subtype_to_npdtype


class PackedAudioWriter(object):
    """Class to pack multiple audio files into a single audio file.
       It will produce a single audio file (packed oudio file)
       plus an scp file with the
       time-stamps indicating the location of the original files in
       packed audio file

    Attributes:
      audio_path: output data file path.
      script_path: optional output scp file.
      audio_format:   audio file format
      subtype: subtype of audio in [PCM_16, PCM_32, FLOAT, DOUBLE, ...],
               if None, it uses soundfile defaults (recommended)
      fs: sampling freq.
      scp_sep: Separator for scp files (default ' ').
    """

    def __init__(
        self,
        audio_path,
        script_path=None,
        audio_format="wav",
        audio_subtype=None,
        fs=16000,
        wav_scale=1,
        scp_sep=" ",
    ):
        self.audio_path = audio_path
        self.script_path = script_path
        self.audio_format = audio_format
        self.scp_sep = scp_sep
        self.fs = int(fs)
        self.wav_scale = wav_scale
        self.cur_pos = 0

        assert "." + self.audio_format in valid_ext
        if audio_subtype is None:
            self.subtype = sf.default_subtype(self.audio_format)
        else:
            self.subtype = audio_subtype
            assert sf.check_format(self.audio_format, self.subtype)

        assert self.subtype in subtype_to_npdtype
        self.audio_dtype = subtype_to_npdtype[self.subtype]

        if script_path is not None:
            self.f_script = open(script_path, "w")
        else:
            self.f_script = None

        self.f_audio = sf.SoundFile(
            audio_path,
            mode="w",
            samplerate=self.fs,
            subtype=self.subtype,
            format=audio_format,
            channels=1,
        )

    def __enter__(self):
        """Function required when entering contructions of type

        with PackedAudioWriter('./output_file.flac', './audio_file.scp',
                               audio_format='flac') as f:
           f.write(key, data)
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Function required when exiting from contructions of type

        with PackedAudioWriter('./output_file.flac', './audio_file.scp',
                               audio_format='flac') as f:
           f.write(key, data)

        """
        self.close()

    def close(self):
        """Closes the script file if open"""
        self.f_audio.close()
        if self.f_script is not None:
            self.f_script.close()

    def write(self, keys, data):
        """Writes waveform to packed audio file.

        Args:
          key: List of recodings names.
          data: List of waveforms
        """
        if isinstance(keys, str):
            keys = [keys]
            data = [data]

        for i, key_i in enumerate(keys):
            assert is_token(key_i), "Token %s not valid" % key_i
            data_i = data[i] / self.wav_scale
            data_i = data_i.astype(self.audio_dtype, copy=False)
            num_samples = len(data_i)
            self.f_audio.write(data_i)
            self.f_audio.flush()

            if self.f_script is not None:
                self.f_script.write(
                    "%s%s%s:%d[0:%d]\n"
                    % (
                        key_i,
                        self.scp_sep,
                        self.audio_path,
                        self.cur_pos,
                        num_samples - 1,
                    )
                )
                self.f_script.flush()
            self.cur_pos += num_samples

    @staticmethod
    def filter_args(**kwargs):
        valid_args = (
            "output_fs",
            "output_wav_scale",
            "output_audio_format",
            "output_audio_subtype",
        )
        return dict(
            (re.sub("output_", "", k), kwargs[k]) for k in valid_args if k in kwargs
        )

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
        else:
            p1 = "--" + prefix + "."

        parser.add_argument(
            p1 + "output-wav-scale",
            default=1,
            type=float,
            help=("scale to divide the waveform before writing"),
        )

        parser.add_argument(
            p1 + "output-audio-format",
            default="flac",
            choices=["flac", "ogg", "wav"],
            help=("ouput audio format"),
        )

        parser.add_argument(
            p1 + "output-audio-subtype",
            default=None,
            choices=["pcm_16", "pcm_24", "float", "double", "vorbis"],
            help=("coding format for audio file"),
        )

        parser.add_argument(
            p1 + "output-fs", default=16000, type=int, help=("output sample frequency")
        )

    add_argparse_args = add_class_args
