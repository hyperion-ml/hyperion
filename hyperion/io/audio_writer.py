"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os
import re

import numpy as np
import soundfile as sf
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from typing import Union, Optional, List
from pathlib import Path

from ..hyp_defs import float_cpu
from ..utils.kaldi_io_funcs import is_token
from ..utils import PathLike
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
    "PCM_S8": "int16",
    "VORBIS": "float32",
    "GSM610": "int16",
    "G721_32": "int16",
    "PCM_24": "int32",
}

scale_32 = 2 ** 31 - 1
scale_24 = 2 ** 23 - 1
scale_16 = 2 ** 15 - 1
scale_8 = 2 ** 7 - 1


subtype_to_scale = {
    "PCM_32": scale_32,
    "ALAW": scale_16,
    "IMA_ADPCM": scale_16,
    "FLOAT": 1,
    "PCM_16": scale_16,
    "DOUBLE": 1,
    "MS_ADPCM": scale_16,
    "ULAW": scale_16,
    "PCM_S8": scale_8,
    "VORBIS": 1,
    "GSM610": scale_16,
    "G721_32": scale_16,
    "PCM_24": scale_24,
}


class AudioWriter(object):
    """Abstract base class to write audio files.

    Attributes:
      output_path: output data file path.
      script_path: optional output kaldi .scp or pandas .csv file.
      audio_format:   audio file format
      audio_subtype: subtype of audio in [PCM_16, PCM_32, FLOAT, DOUBLE, ...],
               if None, it uses soundfile defaults (recommended)
      wav_scale: scale of the input waveform
    """

    def __init__(
        self,
        output_path: PathLike,
        script_path: Optional[PathLike] = None,
        audio_format: str = "wav",
        audio_subtype: Optional[str] = None,
        wav_scale: float = 1.0,
    ):
        self.output_path = Path(output_path)
        self.script_path = Path(script_path) if script_path is not None else None
        self.audio_format = audio_format
        self.output_path.mkdir(exist_ok=True, parents=True)

        assert "." + self.audio_format in valid_ext
        if audio_subtype is None:
            self.subtype = sf.default_subtype(self.audio_format)
        else:
            self.subtype = audio_subtype.upper()
            assert sf.check_format(self.audio_format, self.subtype)

        self._dtype = subtype_to_npdtype[self.subtype]

        self.wav_scale = wav_scale
        # we multiply the audio for this number before saving it.
        self._output_wav_scale = subtype_to_scale[self.subtype] / wav_scale

        self.script_is_scp = False
        self.script_sep = None
        self.f_script = None
        if script_path is not None:
            self.script_path.parent.mkdir(exist_ok=True, parents=True)
            script_ext = self.script_path.suffix
            self.script_is_scp = script_ext == ".scp"

            if self.script_is_scp:
                self.f_script = open(self.script_path, "w")
            else:
                self.script_sep = "," if script_ext == ".csv" else "\t"
                self.f_script = open(self.script_path, "w", encoding="utf-8")
                row = self.script_sep.join(
                    ["id", "storage_path", "duration", "sample_freq"]
                )
                self.f_script.write(f"{row}\n")

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

    def write(
        self,
        keys: Union[str, List[str], np.array],
        data: Union[np.array, List[np.array]],
        fs: Union[int, float, List[int], List[float], np.array],
    ):
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

        output_files = []
        for i, key_i in enumerate(keys):
            assert is_token(key_i), "Token %s not valid" % key_i
            file_basename = re.sub("/", "-", key_i)
            output_file = "%s/%s.%s" % (
                self.output_path,
                file_basename,
                self.audio_format,
            )
            fs_i = int(fs[i]) if fs_is_list else fs
            data_i = (self._output_wav_scale * data[i]).astype(self._dtype, copy=False)
            sf.write(output_file, data_i, fs_i, subtype=self.subtype)

            output_files.append(output_file)

            if self.f_script is not None:
                if self.script_is_scp:
                    self.f_script.write(f"{key_i} {output_file}\n")
                else:
                    duration_i = data_i.shape[-1] / fs_i
                    row = self.script_sep.join(
                        [key_i, output_file, str(duration_i), str(fs_i)]
                    )
                    self.f_script.write(f"{row}\n")
                self.f_script.flush()

        return output_files

    @staticmethod
    def filter_args(**kwargs):
        valid_args = (
            "wav_scale",
            "audio_format",
            "audio_subtype",
        )
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--audio-format",
            default="flac",
            choices=["flac", "ogg", "wav"],
            help=("ouput audio format"),
        )

        parser.add_argument(
            "--audio-subtype",
            default=None,
            choices=["pcm_16", "pcm_24", "pcm_32", "float", "double", "vorbis"],
            help=("coding format for audio file"),
        )

        try:
            parser.add_argument(
                "--wav-scale", default="1.0", help=("input waveform scale wrt 1"),
            )
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix, action=ActionParser(parser=parser),
            )

    add_argparse_args = add_class_args
