"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yaml
from torchaudio.io import AudioEffector, CodecConfig

from ..preprocessing.resampler import ResamplerFromInputFreq, ResamplerToTargetFreq


class CodecAugment:
    """Augments speech waveforms with random codec perturbations.

    Attributes:
      codec_prob: Probability of applying codec perturbation to an input utterance.
      codec_types: Sequence of codec names that can be sampled.
      codec_choice_prob: Sampling probabilities aligned with ``codec_types``.
      mp3_vbr_prob: Probability of using MP3 variable bitrate mode.
      mp3_cbrs: Valid MP3 constant bitrate values (in bps) allowed by configuration.
      mp3_qscale: MP3 VBR quality range ``[min_qscale, max_qscale]`` where lower is better.
      mp3_compression: MP3 compression level range ``[min_level, max_level]`` where lower is better.
      vorbis_compression: OGG Vorbis compression range ``[min_level, max_level]`` where higher is better.
      opus_compression: OGG Opus compression range ``[min_level, max_level]`` where higher is better.
      rng: Random number generator used for codec and parameter sampling.
      valid_tel_codecs: Codecs considered telephony-style codecs.
      resampler_to_tel: Resampler that downsamples to 8 kHz for telephony codecs.
      resampler_from_tel: Resampler that restores original sample rate after telephony codecs.
      tel_mask: Boolean mask selecting telephony codecs in ``codec_types``.
      media_mask: Boolean mask selecting non-telephony codecs in ``codec_types``.
    """

    SUPPORTED_CODEC_TYPES = (
        "alaw",
        "mulaw",
        "g723_1",
        "g726",
        "g722",
        "ac3",
        "mp3",
        "vorbis",
        "opus",
    )

    def __init__(
        self,
        codec_prob: float,
        codec_types: Sequence[str] = (
            "mulaw",
            "alaw",
            "g723_1",
            "g726",
            "g722",
            "ac3",
            "mp3",
            "vorbis",
            "opus",
        ),
        codec_choice_prob: Union[Sequence[float], str] = "uniform",
        mp3_vbr_prob: float = 1.0,
        mp3_cbr: Sequence[int] = (8, 320),
        mp3_qscale: Sequence[int] = (0, 9),
        mp3_compression: Sequence[int] = (0, 9),
        vorbis_compression: Sequence[int] = (-1, 10),
        opus_compression: Sequence[int] = (0, 10),
        random_seed: int = 112358,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        logging.info(
            f"init codec augment with prob={codec_prob} {codec_types=} {codec_choice_prob=}"
        )
        if not np.isscalar(codec_prob):
            raise TypeError(
                f"codec_prob must be a scalar value, got {type(codec_prob)}"
            )
        codec_prob = float(codec_prob)
        if not np.isfinite(codec_prob) or codec_prob < 0 or codec_prob > 1:
            raise ValueError(f"codec_prob must be in [0, 1], got {codec_prob}")

        if not np.isscalar(mp3_vbr_prob):
            raise TypeError(
                f"mp3_vbr_prob must be a scalar value, got {type(mp3_vbr_prob)}"
            )
        mp3_vbr_prob = float(mp3_vbr_prob)
        if not np.isfinite(mp3_vbr_prob) or mp3_vbr_prob < 0 or mp3_vbr_prob > 1:
            raise ValueError(f"mp3_vbr_prob must be in [0, 1], got {mp3_vbr_prob}")

        def _validate_int_pair(
            name: str,
            values: Sequence[int],
            min_allowed: Optional[int] = None,
            max_allowed: Optional[int] = None,
        ) -> Tuple[int, int]:
            if isinstance(values, (str, bytes)):
                raise TypeError(f"{name} must be a sequence of two integer values")
            try:
                values = list(values)
            except TypeError as err:
                raise TypeError(
                    f"{name} must be a sequence of two integer values"
                ) from err

            if len(values) != 2:
                raise ValueError(
                    f"{name} must contain exactly two values, got {values}"
                )

            out = []
            for v in values:
                if not np.isscalar(v):
                    raise TypeError(
                        f"{name} entries must be scalar values, got {type(v)}"
                    )
                v_f = float(v)
                if not np.isfinite(v_f) or not v_f.is_integer():
                    raise ValueError(
                        f"{name} entries must be finite integers, got {values}"
                    )
                out.append(int(v_f))

            low, high = out
            if low > high:
                raise ValueError(
                    f"{name} lower bound must be <= upper bound, got ({low}, {high})"
                )
            if min_allowed is not None and low < min_allowed:
                raise ValueError(
                    f"{name} lower bound must be >= {min_allowed}, got {low}"
                )
            if max_allowed is not None and high > max_allowed:
                raise ValueError(
                    f"{name} upper bound must be <= {max_allowed}, got {high}"
                )
            return low, high

        mp3_cbr = _validate_int_pair("mp3_cbr", mp3_cbr, min_allowed=0)
        mp3_qscale = _validate_int_pair("mp3_qscale", mp3_qscale, 0, 9)
        mp3_compression = _validate_int_pair("mp3_compression", mp3_compression, 0, 9)
        vorbis_compression = _validate_int_pair(
            "vorbis_compression", vorbis_compression, -1, 10
        )
        opus_compression = _validate_int_pair(
            "opus_compression", opus_compression, 0, 10
        )

        self.codec_prob = codec_prob
        self.codec_types = list(codec_types)
        if not self.codec_types:
            raise ValueError("codec_types must contain at least one codec name")
        invalid_codecs = sorted(set(self.codec_types) - set(self.SUPPORTED_CODEC_TYPES))
        if invalid_codecs:
            raise ValueError(
                "Unsupported codec types in codec_types: "
                f"{invalid_codecs}. Supported codecs are {self.SUPPORTED_CODEC_TYPES}"
            )

        if isinstance(codec_choice_prob, str) and codec_choice_prob == "uniform":
            codec_choice_prob = np.ones((len(self.codec_types),))
        elif isinstance(codec_choice_prob, str):
            raise ValueError(
                f"codec_choice_prob='{codec_choice_prob}' is not supported. "
                "Use 'uniform' or a sequence of probabilities."
            )

        self.codec_choice_prob = np.asarray(codec_choice_prob, dtype=float)
        if self.codec_choice_prob.ndim != 1:
            raise ValueError("codec_choice_prob must be a 1-D sequence")
        if len(self.codec_choice_prob) != len(self.codec_types):
            raise ValueError(
                "codec_choice_prob length does not match codec_types: "
                f"{len(self.codec_choice_prob)} != {len(self.codec_types)}"
            )
        if not np.all(np.isfinite(self.codec_choice_prob)):
            raise ValueError("codec_choice_prob must contain only finite values")
        if np.any(self.codec_choice_prob < 0):
            raise ValueError("codec_choice_prob must contain non-negative values")

        prob_sum = self.codec_choice_prob.sum()
        if prob_sum <= 0:
            raise ValueError("codec_choice_prob sum must be > 0")
        self.codec_choice_prob /= prob_sum
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
        if "mp3" in self.codec_types and not self.mp3_cbrs:
            raise ValueError(
                "mp3_cbr range does not include any supported CBR values. "
                f"Got mp3_cbr={tuple(mp3_cbr)}; supported values (kbps) are {valid_cbrs}."
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
    def create(
        cls,
        cfg: Union[str, Dict[str, Any]],
        random_seed: int = 112358,
        rng: Optional[np.random.Generator] = None,
    ) -> "CodecAugment":
        """Creates a ``CodecAugment`` object from a config dictionary or YAML file.

        Args:
          cfg: YAML file path or dictionary with codec augmentation options.
          random_seed: Seed used when creating a new random generator.
          rng: Optional pre-created random generator.

        Returns:
          Configured codec augmenter instance.
        """
        if isinstance(cfg, str):
            with open(cfg, "r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

        if not isinstance(cfg, dict):
            raise TypeError(f"wrong object type for cfg={cfg}")

        return cls(
            **cfg,
            random_seed=random_seed,
            rng=rng,
        )

    def _get_tel_filter(self) -> str:
        """Builds a random telephony bandpass filter for FFmpeg effects.

        Args:
          None.

        Returns:
          Comma-separated effect string with one random highpass and one lowpass filter.
        """
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

    def _get_codec_type(
        self, enable_tel_codecs: bool, enable_media_codecs: bool
    ) -> Optional[str]:
        """Samples a codec name under telephony/media enable constraints.

        Args:
          enable_tel_codecs: If ``True``, telephony codecs are eligible for sampling.
          enable_media_codecs: If ``True``, media codecs are eligible for sampling.

        Returns:
          Selected codec type, or ``None`` when no codec type is allowed.
        """
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
        codec_type = self.rng.choice(self.codec_types, p=codec_choice_prob)
        return codec_type

    def forward(
        self,
        x: np.ndarray,
        sample_freq: Optional[float],
        enable_tel_codecs: bool = True,
        enable_media_codecs: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Applies a randomly sampled codec to the signal.

        Args:
          x: Original speech waveform.
          sample_freq: Waveform sample rate in Hz. Required when codec processing is enabled.
          enable_tel_codecs: Enables sampling telephony codecs.
          enable_media_codecs: Enables sampling media codecs.

        Returns:
          Augmented waveform after codec processing.
          Dictionary with codec metadata (codec type and sampled codec options).
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

        if sample_freq is None:
            raise ValueError(
                "sample_freq must be provided when codec augmentation is applied"
            )
        if not np.isscalar(sample_freq):
            raise TypeError(
                f"sample_freq must be a scalar value, got {type(sample_freq)}"
            )
        sample_freq = float(sample_freq)
        if not np.isfinite(sample_freq) or sample_freq <= 0:
            raise ValueError(
                f"sample_freq must be a positive finite value, got {sample_freq}"
            )

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
            if p < self.mp3_vbr_prob:
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
        else:
            raise ValueError(f"Unsupported codec_type sampled: {codec_type}")

        # print("codec:", str(effect_config), "tel_filter:", tel_filter, flush=True)
        # t1 = time.time()
        did_tel_resample = False
        if tel_resampler:
            try:
                x, effector_sample_freq = self.resampler_to_tel(x, sample_freq)
                did_tel_resample = True
            except Exception as err:
                logging.warning(
                    "Codec %s pre-resample error: %s. Using original sample rate.",
                    codec_type,
                    str(err),
                )
                effector_sample_freq = sample_freq
        else:
            effector_sample_freq = sample_freq

        x = torch.from_numpy(x).unsqueeze(1)
        if tel_filter is not None:
            try:
                effector = AudioEffector(effect=tel_filter)
                x = effector.apply(x, int(effector_sample_freq))
            except Exception as err:
                logging.warning("Codec %s tel-filter error: %s", codec_type, str(err))

        effector = AudioEffector(**effect_config)
        try:
            y = effector.apply(x, sample_rate=int(effector_sample_freq))
        except Exception as err:
            logging.warning("Codec %s error: %s", codec_type, str(err))
            y = x

        y = y.squeeze(1).numpy()
        if did_tel_resample:
            try:
                y, _ = self.resampler_from_tel(y, sample_freq)
            except Exception as err:
                logging.warning(
                    "Codec %s post-resample error: %s. Returning unresampled output.",
                    codec_type,
                    str(err),
                )
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
        self,
        x: np.ndarray,
        sample_freq: Optional[float] = None,
        enable_tel_codecs: bool = True,
        enable_media_codecs: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Runs codec augmentation using callable-style syntax.

        Args:
          x: Original speech waveform.
          sample_freq: Waveform sample rate in Hz. Required when codec processing is enabled.
          enable_tel_codecs: Enables sampling telephony codecs.
          enable_media_codecs: Enables sampling media codecs.

        Returns:
          Augmented waveform after codec processing.
          Dictionary with codec metadata (codec type and sampled codec options).
        """
        return self.forward(x, sample_freq, enable_tel_codecs, enable_media_codecs)

    def reseed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Reseeds the internal RNG."""
        self.rng = np.random.default_rng(seed=seed)
