"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from ....np.preprocessing import ResamplerToTargetFreq
from ....utils.misc import PathLike


class UTMOSV2:
    """Wrapper around the official implementation of UTMOS v2 model.

    Attributes:
      tmp_dir: temporal dir to store audios, since the UTMOS code only allows
        to get audio from disk
    """

    def __init__(self, tmp_dir: Optional[PathLike] = None):

        if tmp_dir is None:
            tmp_dir = Path("./utmos_cache")

        self.model = None
        pid = os.getpid()
        self.tmp_dir = Path(tmp_dir) / f"utmos.{pid}"
        self.delete_tmp_dir()
        self.make_tmp_dir()
        self.tmp_audio_ids = []
        self.resampler = ResamplerToTargetFreq(16000.0)

    def delete_tmp_dir(self):
        if self.tmp_dir.is_dir():
            shutil.rmtree(self.tmp_dir)

    def make_tmp_dir(self):
        self.tmp_dir.mkdir(exist_ok=True, parents=True)

    def add_audios(
        self,
        audios: Union[List[torch.Tensor], List[np.ndarray]],
        audio_fs: Union[torch.Tensor, np.ndarray, List[float]],
        audio_ids: Union[List[str], np.ndarray, None] = None,
    ):
        """Append audios to a temporal directory.
           compute_mos computes MOS for all audios in the temporal directory.

        Arguments:
          audios: List of audio np.ndarray or torch.Tensor
          audio_fs: List of audios sample freq
          audio_ids: List of audio ids, if None, it uses integers from 0 to num_audios-1
        """

        if audio_ids is None:
            audio_ids = [str(i) for i in range(len(audios))]

        for audio, fs, audio_id in zip(audios, audio_fs, audio_ids):
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()

            audio, fs = self.resampler(audio, fs)
            file_path = self.tmp_dir / f"{audio_id}.flac"
            sf.write(str(file_path), audio, 16000)
            self.tmp_audio_ids.append(audio_id)

    def compute_mos(
        self,
        audios: Union[List[torch.Tensor], List[np.ndarray], None] = None,
        audio_fs: Union[torch.Tensor, np.ndarray, List[float], None] = None,
        audio_ids: Union[List[str], np.ndarray, None] = None,
    ):
        """Computes MOS for all audios in the temporal directory.
           If audios and audio_fs are not None, it compute MOS for these input audios.

        Arguments:
          audios: List of audio np.ndarray or torch.Tensor
          audio_fs: List of audios sample freq
          audio_ids: List of audio ids, if None, it uses integers from 0 to num_audios-1
        """

        if audios is not None and audio_fs is not None:
            self.add_audios(audios, audio_fs, audio_ids)

        if self.model is None:
            import utmosv2

            self.model = utmosv2.create_model(pretrained=True)

        preds = self.model.predict(input_dir=self.tmp_dir)
        pred_ids = []
        for pred in preds:
            pred_id = Path(pred["file_path"]).stem
            pred_ids.append(pred_id)
            del pred["file_path"]

        preds = pd.DataFrame(preds)
        preds["id"] = pred_ids
        preds.set_index("id", inplace=True)
        pred_mos = preds.loc[self.tmp_audio_ids].values
        audio_ids = self.tmp_audio_ids

        self.delete_tmp_dir()
        self.make_tmp_dir()
        self.tmp_audio_ids = []
        return audio_ids, pred_mos

    def delete_model(self):
        """Releases the model memory"""
        del self.model
        self.model = None

    def __call__(
        self,
        audios: Union[List[torch.Tensor], List[np.ndarray], None] = None,
        audio_fs: Union[torch.Tensor, np.ndarray, List[float], None] = None,
        audio_ids: Union[List[str], np.ndarray, None] = None,
    ):
        """Computes MOS for all audios in the temporal directory.
           If audios and audio_fs are not None, it compute MOS for these input audios.

        Arguments:
          audios: List of audio np.ndarray or torch.Tensor
          audio_fs: List of audios sample freq
          audio_ids: List of audio ids, if None, it uses integers from 0 to num_audios-1
        """

        return self.compute_mos(audios, audio_fs, audio_ids)
