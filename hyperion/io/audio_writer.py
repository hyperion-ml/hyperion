"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os
import re
import soundfile as sf

import numpy as np

from ..hyp_defs import float_cpu
from ..utils.scp_list import SCPList
from ..utils.kaldi_io_funcs import is_token
from .audio_reader import valid_ext

subtype_to_npdtype = {
    "PCM_32": "int32",
    "ALAW": "int16",
    "IMA_ADPCM": "int16",
    "FLOAT": "float32",
    "PCM_16": "int16",
    "DOUBLE": "float64",
    "MS_ADPCM": "int16",
    "ULAW": "int16",
    "PCM_U8": "uint8",
    "PCM_S8": "int8",
    "VORBIS": "float32",
    "GSM610": "int16",
    "G721_32": "int16",
    "PCM_24": "int24",
}


class AudioWriter(object):
    """Abstract base class to write audio files.

    Attributes:
      output_path: output data file path.
      script_path: optional output scp file.
      audio_format:   audio file format
      audio_subtype: subtype of audio in [PCM_16, PCM_32, FLOAT, DOUBLE, ...],
               if None, it uses soundfile defaults (recommended)
      scp_sep: Separator for scp files (default ' ').
    """

    def __init__(
        self,
        output_path,
        script_path=None,
        audio_format="wav",
        audio_subtype=None,
        scp_sep=" ",
    ):
        self.output_path = output_path
        self.script_path = script_path
        self.audio_format = audio_format
        self.scp_sep = scp_sep

        assert "." + self.audio_format in valid_ext
        if audio_subtype is None:
            self.subtype = sf.default_subtype(self.audio_format)
        else:
            self.subtype = audio_subtype
            assert sf.check_format(self.audio_format, self.subtype)

        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except FileExistsError:
                pass

        if script_path is not None:
            self.f_script = open(script_path, "w")
        else:
            self.f_script = None

    def __enter__(self):
        """Function required when entering contructions of type

        with AudioWriter('./path') as f:
           f.write(key, data)
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Function required when exiting from contructions of type

        with AudioWriter('./path') as f:
           f.write(key, data)
        """
        self.close()

    def close(self):
        """Closes the script file if open"""
        if self.f_script is not None:
            self.f_script.close()

    def write(self, keys, data, fs):
        """Writes waveform to audio file.

        Args:
          key: List of recodings names.
          data: List of waveforms
          fs:
        """
        if isinstance(keys, str):
            keys = [keys]
            data = [data]

        fs_is_list = isinstance(fs, (list, np.ndarray))
        assert self.subtype in subtype_to_npdtype
        dtype = subtype_to_npdtype[self.subtype]
        output_files = []
        for i, key_i in enumerate(keys):
            assert is_token(key_i), "Token %s not valid" % key_i
            file_basename = re.sub("/", "-", key_i)
            output_file = "%s/%s.%s" % (
                self.output_path,
                file_basename,
                self.audio_format,
            )
            fs_i = fs[i] if fs_is_list else fs
            data_i = data[i].astype(dtype, copy=False)
            sf.write(output_file, data_i, fs_i, subtype=self.subtype)

            output_files.append(output_file)

            if self.f_script is not None:
                self.f_script.write("%s%s%s\n" % (key_i, self.scp_sep, output_file))
                self.f_script.flush()

        return output_files

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

        # parser.add_argument(p1+'output-wav-scale', default=1, type=float,
        #                      help=('scale to divide the waveform before writing'))

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

        # parser.add_argument(p1+'output-fs', default=16000, type=int,
        #                      help=('output sample frequency'))

    add_argparse_args = add_class_args
