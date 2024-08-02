"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
import multiprocessing
import re
import time
from copy import deepcopy
from enum import Enum
from typing import Dict, List, Union

import numpy as np
import soundfile as sf
import torch
import yaml
from torchaudio.io import AudioEffector, CodecConfig

from ...hyp_defs import float_cpu
from ..preprocessing.resampler import ResamplerFromInputFreq, ResamplerToTargetFreq


class CodecAugment:
    """Class to augment speech with codecs.

    Attributes:
      codec_prob: probability of applying codec perturbation.
      codec_types: list of possible codecs.
      codec_choice_prob: list with the probability of each codec or "uniform"
      mp3_qscale: mp3 quality [min_qscale, max_qscale] #lower the better
                  0-3 will normally produce transparent results, 4 (default) should be close to perceptual transparency, and 6 produces an "acceptable" quality
      mp3_compression: interval of compression levels for MP3 [min_compression_level, max_compression_level] # lower better
      vorbis_compression: interval of compression levels for OGG Vorbis [min_compression_level, max_compression_level] # higher better (default:3)
      opus_compression: interval of compression levels for OGG Opus [min_compression_level, max_compression_level] # higher better (default:10)
      random_seed: random seed for random number generator.
      rng:     Random number generator returned by
               np.random.default_rng (optional).
    """

    def __init__(
        self,
        codec_prob: float,
        codec_types: List[str] = [
            "mulaw",
            "alaw",
            "723_1",
            "726",
            "g722",
            "ac3",
            "mp3",
            "vorbis",
            "opus",
        ],
        codec_choice_prob: Union[List[str], str] = "uniform",
        mp3_vbr_prob: float = 1.0,
        mp3_cbr: List[int] = [8, 320],
        mp3_qscale: List[int] = [0, 9],
        mp3_compression: List[int] = [0, 9],
        vorbis_compression: List[int] = [-1, 10],
        opus_compression: List[int] = [0, 10],
        random_seed=112358,
        rng=None,
    ):
        logging.info(
            f"init codec augment with prob={codec_prob} {codec_types=} {codec_choice_prob=}"
        )
        self.codec_prob = codec_prob
        self.codec_types = codec_types
        if codec_choice_prob == "uniform":
            codec_choice_prob = np.ones((len(codec_types),))
        self.codec_choice_prob = np.asarray(codec_choice_prob)
        self.codec_choice_prob /= self.codec_choice_prob.sum()
        self.mp3_vbr_prob = mp3_vbr_prob
        valid_cbrs = [
            8,
            16,
            24,
            32,
            40,
            48,
            64,
            80,
            96,
            112,
            128,
            160,
            192,
            224,
            256,
            320,
        ]
        self.mp3_cbrs = [
            int(cbr * 1000)
            for cbr in valid_cbrs
            if cbr >= mp3_cbr[0] and cbr <= mp3_cbr[1]
        ]
        assert (
            mp3_qscale[0] >= 0 and mp3_qscale[1] <= 9 and mp3_qscale[0] <= mp3_qscale[1]
        )
        assert (
            mp3_compression[0] >= 0
            and mp3_compression[1] <= 9
            and mp3_compression[0] <= mp3_compression[1]
        )
        assert (
            vorbis_compression[0] >= -1
            and vorbis_compression[1] <= 10
            and vorbis_compression[0] <= vorbis_compression[1]
        )
        assert (
            opus_compression[0] >= 0
            and opus_compression[1] <= 10
            and opus_compression[0] <= opus_compression[1]
        )
        self.mp3_qscale = mp3_qscale
        self.mp3_compression = mp3_compression
        self.vorbis_compression = vorbis_compression
        self.opus_compression = opus_compression
        if rng is None:
            self.rng = np.random.default_rng(seed=random_seed)
        else:
            self.rng = deepcopy(rng)

        self.valid_tel_codecs = ["alaw", "mulaw", "g723_1", "g726"]
        if any([codec in self.codec_types for codec in self.valid_tel_codecs]):
            self.resampler_to_tel = ResamplerToTargetFreq(8000.0)
            self.resampler_from_tel = ResamplerFromInputFreq(8000.0)
        else:
            self.resampler_to_tel = None
            self.resampler_from_tel = None

        self.tel_mask = np.asarray(
            [True if c in self.valid_tel_codecs else False for c in self.codec_types]
        )
        self.media_mask = np.logical_not(self.tel_mask)

    @classmethod
    def create(cls, cfg, random_seed=112358, rng=None):
        """Creates a SpeedAugment object from options dictionary or YAML file.

        Args:
          cfg: YAML file path or dictionary with  options.
          rng: Random number generator returned by
               np.random.default_rng (optional).

        Returns:
          CodecAugment object.
        """
        if isinstance(cfg, str):
            with open(cfg, "r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

        assert isinstance(cfg, dict), f"wrong object type for cfg={cfg}"

        return cls(
            **cfg,
            random_seed=random_seed,
            rng=rng,
        )

    def _get_tel_filter(self):
        poles = self.rng.integers(low=1, high=3)
        id = self.rng.integers(low=0, high=3)
        if id == 0:
            highpass = f"highpass=frequency=300:poles={poles}"
        elif id == 1:
            highpass = f"highpass=frequency=200:poles={poles}"
        else:
            highpass = f"highpass=frequency=100:poles={poles}"

        id = self.rng.integers(low=0, high=3)
        if id == 0:
            lowpass = f"lowpass=frequency=3400:poles={poles}"
        elif id == 1:
            lowpass = f"lowpass=frequency=3700:poles={poles}"
        else:
            lowpass = f"lowpass=frequency=3900:poles={poles}"

        filter = ",".join([highpass, lowpass])
        return filter

    def _get_codec_type(self, enable_tel_codecs: bool, enable_media_codecs: bool):
        if not enable_media_codecs and not enable_tel_codecs:
            return None

        codec_choice_prob = self.codec_choice_prob.copy()
        if not enable_tel_codecs:
            codec_choice_prob[self.tel_mask] = 0.0

        if not enable_media_codecs:
            codec_choice_prob[self.media_mask] = 0.0

        prob_acc = codec_choice_prob.sum()
        if prob_acc == 0:
            return None

        codec_choice_prob /= prob_acc
        codec_type = self.rng.choice(self.codec_types, p=self.codec_choice_prob)
        return codec_type

    def forward(
        self,
        x: np.ndarray,
        sample_freq: float,
        enable_tel_codecs: bool = True,
        enable_media_codecs: bool = True,
    ):
        """Apply codec to the signal,

        Args:
          x: original speech signal.

        Returns:
          Augmented signal.
          Dictionary containing information about the codec type and codec options.
        """
        # decide whether to add noise or not
        x = x.astype("float32", copy=False)
        p = self.rng.random()
        if p > self.codec_prob:
            # we don't add codec
            info = {"codec_type": None}
            return x, info

        # id = self.rng.integers(low=0, high=100000)
        # sf.write(f"audios/{id}.flac", x, samplerate=sample_freq)

        codec_type = self._get_codec_type(enable_tel_codecs, enable_media_codecs)
        info = {"codec_type": codec_type}
        if codec_type is None:
            return x, info

        tel_filter = None
        tel_resampler = False
        if codec_type == "alaw":
            effect_config = {"format": "wav", "encoder": "pcm_alaw"}
            tel_filter = self._get_tel_filter()
        elif codec_type == "mulaw":
            effect_config = {"format": "wav", "encoder": "pcm_mulaw"}
            tel_filter = self._get_tel_filter()
        # elif codec_type == "gsm":
        #     effect_config = {"format": "gsm"}
        #     tel_resampler = True
        # elif codec_type == "g711":
        #     effect_config = {"format": "g711"}
        #     tel_resampler = True
        elif codec_type == "g723_1":
            effect_config = {"format": "g723_1"}
            tel_resampler = True
            tel_filter = self._get_tel_filter()
        elif codec_type == "g726":
            effect_config = {"format": "g726"}
            tel_resampler = True
            tel_filter = self._get_tel_filter()
        # elif codec_type == "g729":
        #     effect_config = {"format": "g729"}
        #     tel_resampler = True
        # elif codec_type == "amr_nb":
        #     effect_config = {"format": "amr_nb"}
        #     tel_resampler = True
        # elif codec_type == "amrnb":
        #     effect_config = {"format": "amrnb"}
        #     tel_resampler = True
        # elif codec_type == "amr":
        #     effect_config = {"format": "amr"}
        #     tel_resampler = True
        elif codec_type == "g722":
            effect_config = {"format": "g722"}
        elif codec_type == "ac3":
            effect_config = {"format": "ac3"}
        # elif codec_type == "ac4":
        #     effect_config = {"format": "ac4"}
        # elif codec_type == "aac":
        #     effect_config = {"format": "aac"}
        elif codec_type == "mp3":
            compression_level = self.rng.integers(
                low=self.mp3_compression[0], high=self.mp3_compression[1] + 1
            )
            info["compression_level"] = compression_level
            p = self.rng.random()
            if p > self.mp3_vbr_prob:
                # we do variable bit rate
                qscale = self.rng.integers(
                    low=self.mp3_qscale[0], high=self.mp3_qscale[1] + 1
                )
                codec_config = CodecConfig(
                    compression_level=compression_level, qscale=qscale
                )
                info["vbr"] = True
                info["qscale"] = qscale
            else:
                cbr = self.rng.choice(self.mp3_cbrs)
                codec_config = CodecConfig(
                    compression_level=compression_level, bit_rate=cbr
                )
                info["vbr"] = False
                info["bit_rate"] = cbr

            effect_config = {"format": "mp3", "codec_config": codec_config}
        elif codec_type == "vorbis":
            compression_level = self.rng.integers(
                low=self.vorbis_compression[0], high=self.vorbis_compression[1] + 1
            )
            codec_config = CodecConfig(compression_level=compression_level)
            info["compression_level"] = compression_level
            effect_config = {
                "format": "ogg",
                "encoder": "vorbis",
                "codec_config": codec_config,
            }
        elif codec_type == "opus":
            compression_level = self.rng.integers(
                low=self.opus_compression[0], high=self.opus_compression[1] + 1
            )
            codec_config = CodecConfig(compression_level=compression_level)
            info["compression_level"] = compression_level
            effect_config = {
                "format": "ogg",
                "encoder": "opus",
                "codec_config": codec_config,
            }

        # print("codec:", str(effect_config), "tel_filter:", tel_filter, flush=True)
        # t1 = time.time()
        if tel_resampler:
            try:
                x, effector_sample_freq = self.resampler_to_tel(x, sample_freq)
            except:
                print(f"xr1 {x.dtype} {x} {sample_freq}", flush=True)
        else:
            effector_sample_freq = sample_freq

        x = torch.from_numpy(x).unsqueeze(1)
        if tel_filter is not None:
            effector = AudioEffector(effect=tel_filter)
            x = effector.apply(x, int(effector_sample_freq))

        effector = AudioEffector(**effect_config)
        y = effector.apply(x, sample_rate=int(effector_sample_freq))
        y = y.squeeze(1).numpy()
        if tel_resampler:
            try:
                y, _ = self.resampler_from_tel(y, sample_freq)
            except:
                print(f"xr2 {x.dtype} {x} {sample_freq}", flush=True)
        # sinfo = re.sub(r"[{}':\. ]", "", str(info))
        # print(f"codec-time {sinfo} dt={time.time()-t1}", flush=True)
        # sf.write(f"audios/{id}-{sinfo}.flac", y, samplerate=sample_freq)
        # avg proc times
        # mulaw t=0.014
        # alaw t=0.018
        # g722 t=0.016
        # g723_1 t=0.16
        # g726 t=0.04
        # vorbis t=0.08
        # opus t=0.046
        # mp3 t=0.032
        # ac3 t=0.024
        return y, info

    def __call__(
        self, x, sample_freq=None, enable_tel_codecs=True, enable_media_codecs=True
    ):
        return self.forward(x, sample_freq, enable_tel_codecs, enable_media_codecs)


# class CodecAugmentation(SerializableObject):
#     def __init__(
#         self,
#         codec_prob: float,
#         codecs: List[Dict[str, Union[str, int]]] = [
#             {"format": "wav", "encoding": "ULAW", "bits_per_sample": 8},  # 8 bit mu-law
#             {"format": "gsm"},  # GSM-FR
#             {"format": "mp3", "max_compression": -9, "min_compression": -6},  # mp3
#             {
#                 "format": "vorbis",
#                 "max_compression": -1,
#                 "min_compression": 10,
#             },  # vorbis
#         ],
#         p: float = 0.0,
#     ):
#         self.p = p
#         self.codecs = codecs

#     def __call__(self, a, sample_rate):
#         if random.random() > self.p:
#             return a
#         config = random.choice(self.codecs)
#         a = torch.from_numpy(a).unsqueeze(0)
#         if config["format"] == "gsm":
#             codec_sr = 8000
#             a = torchaudio.transforms.Resample(sample_rate, codec_sr)(a)
#         else:
#             codec_sr = sample_rate
#         if "max_compression" in config and "min_compression" in config:
#             config = copy.deepcopy(config)
#             config["compression"] = random.randint(
#                 config.pop("max_compression"), config.pop("min_compression")
#             )
#         a = F.apply_codec(a, codec_sr, **config)
#         if config["format"] == "gsm":
#             a = torchaudio.transforms.Resample(codec_sr, sample_rate)(a)
#         a = a.squeeze(0).numpy()
#         return a
