"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import soundfile as sf
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ..hyp_defs import float_cpu
from ..utils import PathLike
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
    "PCM_S8": "int16",
    "VORBIS": "float32",
    "GSM610": "int16",
    "G721_32": "int16",
    "PCM_24": "int32",
}

scale_32 = 2**31 - 1
scale_24 = 2**23 - 1
scale_16 = 2**15 - 1
scale_8 = 2**7 - 1


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


class AudioWriter:
    """Write audio arrays to disk and optionally create an output manifest.

    Attributes:
      output_path: Directory where audio files are saved.
      script_path: Optional output Kaldi ``.scp`` or table file (``.csv``/``.tsv``).
      audio_format: Output audio container format.
      audio_subtype: Audio encoding subtype (e.g., ``PCM_16``, ``FLOAT``).
      wav_scale: Scale of the input waveform.
      channels_first: If True, interprets 2-D inputs as ``(channels, num_samples)``.
      always_2d: If True, always writes 2-channel output audio.

    Example:
      >>> import numpy as np
      >>> with AudioWriter(
      ...     "./audio_out",
      ...     script_path="./audio_out/recordings.csv",
      ...     audio_format="wav",
      ...     audio_subtype="pcm_16",
      ...     channels_first=True,
      ... ) as w:
      ...     x = np.random.randn(1, 16000).astype("float32")  # (channels, samples)
      ...     files = w.write("utt1", x, 16000)
      >>> files[0]
      './audio_out/utt1.wav'
    """

    def __init__(
        self,
        output_path: PathLike,
        script_path: Optional[PathLike] = None,
        audio_format: str = "wav",
        audio_subtype: Optional[str] = None,
        wav_scale: float = 1.0,
        channels_first: bool = True,
        always_2d: bool = False,
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
        self.channels_first = channels_first
        self.always_2d = always_2d
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
           f.write(key, data, fs)
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Function required when exiting from contructions of type

        with AudioWriter('./path') as f:
           f.write(key, data, fs)
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
        """Write one or more waveforms to audio files.

        Args:
          keys: Recording key or list of recording keys.
          data: Single waveform array or list of waveform arrays.
          fs: Sample rate scalar or list of sample rates aligned with ``keys``.

        Returns:
          List with the output file paths written for each key.

        Raises:
          ValueError: If key/data/fs lengths are inconsistent.
        """
        if isinstance(keys, str):
            keys = [keys]
        else:
            keys = list(keys)

        if isinstance(data, np.ndarray):
            if len(keys) != 1:
                raise ValueError(
                    "data is a single audio array but keys contains multiple items"
                )
            data = [data]
        else:
            data = list(data)

        if isinstance(fs, (int, float, np.integer, np.floating)):
            fs = [int(fs)] * len(keys)
        else:
            fs = [int(v) for v in fs]

        if not (len(keys) == len(data) == len(fs)):
            raise ValueError(
                f"keys/data/fs length mismatch: {len(keys)}/{len(data)}/{len(fs)}"
            )

        output_files = []
        for i, key_i in enumerate(keys):
            assert is_token(key_i), "Token %s not valid" % key_i
            file_basename = re.sub("/", "-", key_i)
            output_file = "%s/%s.%s" % (
                self.output_path,
                file_basename,
                self.audio_format,
            )
            fs_i = fs[i]
            data_i = self._prepare_audio(data[i])
            data_i = (self._output_wav_scale * data_i).astype(self._dtype, copy=False)
            sf.write(output_file, data_i, fs_i, subtype=self.subtype)

            output_files.append(output_file)

            if self.f_script is not None:
                if self.script_is_scp:
                    self.f_script.write(f"{key_i} {output_file}\n")
                else:
                    duration_i = data_i.shape[0] / fs_i
                    if self.script_sep in key_i:
                        key_i = '"' + key_i + '"'

                    if self.script_sep in output_file:
                        output_file = '"' + output_file + '"'
                    row = self.script_sep.join(
                        [key_i, output_file, str(duration_i), str(fs_i)]
                    )
                    self.f_script.write(f"{row}\n")
                self.f_script.flush()

        return output_files

    def _prepare_audio(self, x: np.array) -> np.array:
        """Convert input waveform to soundfile layout.

        Returns audio as either ``(num_samples,)`` or ``(num_samples, channels)``,
        honoring ``channels_first`` and ``always_2d``.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            pass
        elif x.ndim == 2:
            if self.channels_first:
                x = x.T
        else:
            raise ValueError(
                f"Audio tensor for writing must be 1-D or 2-D, got shape={x.shape}"
            )

        if self.always_2d:
            if x.ndim == 1:
                x = np.stack((x, x), axis=1)
            elif x.shape[1] == 1:
                x = np.repeat(x, 2, axis=1)
            elif x.shape[1] > 2:
                x = x[:, :2]

        return x

    @staticmethod
    def filter_args(**kwargs):
        valid_args = (
            "wav_scale",
            "audio_format",
            "audio_subtype",
            "channels_first",
            "always_2d",
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
                "--wav-scale",
                type=float,
                default=1.0,
                help=("input waveform scale wrt 1"),
            )
            parser.add_argument(
                "--channels-first",
                default=True,
                action=ActionYesNo,
                help=("if true, interpret 2-D input as (channels, num_samples)"),
            )
            parser.add_argument(
                "--always-2d",
                default=False,
                action=ActionYesNo,
                help=("if true, force saved waveform to have exactly 2 channels"),
            )
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    add_argparse_args = add_class_args
