"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import List, Optional, Union

import numpy as np
import torch

from ....np.preprocessing import ResamplerToTargetFreq
from ....utils.misc import PathLike


class UTMOSV2:
    """Wrapper around the Hyperion-maintained implementation of UTMOS v2.

    Attributes:
      model: UTMOS model instance, initialized lazily on the first prediction.
      audios: Resampled audio kept in memory until ``compute_mos`` is called.
      tmp_audio_ids: IDs corresponding to the audio in ``audios``.
    """

    def __init__(self, tmp_dir: Optional[PathLike] = None):
        # Keep tmp_dir in the signature for compatibility with older callers.
        # The current UTMOS API accepts audio arrays directly, so it is unused.
        self.model = None
        self.audios = []
        self.tmp_audio_ids = []
        self.resampler = ResamplerToTargetFreq(16000.0)

    def add_audios(
        self,
        audios: Union[List[torch.Tensor], List[np.ndarray]],
        audio_fs: Union[torch.Tensor, np.ndarray, List[float]],
        audio_ids: Union[List[str], np.ndarray, None] = None,
    ):
        """Append audios to the in-memory prediction queue.

        ``compute_mos`` predicts each queued utterance directly from memory.

        Arguments:
          audios: List of audio np.ndarray or torch.Tensor
          audio_fs: List of audios sample freq
          audio_ids: List of audio ids, if None, it uses integers from 0 to num_audios-1
        """

        if audio_ids is None:
            audio_ids = [str(i) for i in range(len(audios))]

        for audio, fs, audio_id in zip(audios, audio_fs, audio_ids):
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()

            audio, fs = self.resampler(audio, fs)
            self.audios.append(audio)
            self.tmp_audio_ids.append(audio_id)

    def compute_mos(
        self,
        audios: Union[List[torch.Tensor], List[np.ndarray], None] = None,
        audio_fs: Union[torch.Tensor, np.ndarray, List[float], None] = None,
        audio_ids: Union[List[str], np.ndarray, None] = None,
    ):
        """Compute MOS for queued or newly provided in-memory audios.

        The current UTMOS API accepts a single waveform through ``data``. Audio
        is predicted one utterance at a time because utterances can have
        different lengths.

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

        if not self.audios:
            raise ValueError("No audios have been added for UTMOS prediction")

        pred_mos = []
        for audio in self.audios:
            pred = self.model.predict(data=audio, sr=16000)
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            pred_mos.append(np.asarray(pred).reshape(-1)[0])

        audio_ids = self.tmp_audio_ids.copy()
        pred_mos = np.asarray(pred_mos)[:, np.newaxis]

        self.audios = []
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
