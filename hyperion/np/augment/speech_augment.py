"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import yaml

from .codec_augment import CodecAugment
from .noise_augment import NoiseAugment
from .reverb_augment import ReverbAugment
from .speed_augment import SpeedAugment


class SpeechAugment:
    """Applies a configurable chain of speech augmentations.

    Attributes:
       speed_aug: Optional speed augmenter.
       reverb_aug: Optional reverb augmenter.
       noise_aug: Optional additive noise augmenter.
       codec_aug: Optional codec augmenter.
       transcodec_aug: Optional second codec augmenter applied after ``codec_aug``.
    """

    def __init__(
        self,
        speed_aug: Optional[SpeedAugment] = None,
        reverb_aug: Optional[ReverbAugment] = None,
        noise_aug: Optional[NoiseAugment] = None,
        codec_aug: Optional[CodecAugment] = None,
        transcodec_aug: Optional[CodecAugment] = None,
    ) -> None:
        """Initializes a speech augmentation pipeline.

        Args:
          speed_aug: Optional speed augmenter.
          reverb_aug: Optional reverb augmenter.
          noise_aug: Optional additive noise augmenter.
          codec_aug: Optional codec augmenter.
          transcodec_aug: Optional second codec augmenter applied conditionally.

        Returns:
          None.
        """
        self.speed_aug = speed_aug
        self.reverb_aug = reverb_aug
        self.noise_aug = noise_aug
        self.codec_aug = codec_aug
        self.transcodec_aug = transcodec_aug

    @classmethod
    def create(
        cls,
        cfg: Union[str, Dict[str, Any]],
        random_seed: int = 112358,
        rng: Optional[np.random.Generator] = None,
    ) -> "SpeechAugment":
        """Creates a SpeechAugment object from options dictionary or YAML file.

        Args:
          cfg: YAML file path or dictionary with augmentation options.
          random_seed: Seed passed to sub-augmenters when they create RNGs.
          rng: Optional pre-created random generator.

        Returns:
          Configured speech augmenter instance.
        """
        if isinstance(cfg, str):
            with open(cfg, "r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

        if not isinstance(cfg, dict):
            raise TypeError(f"wrong object type for cfg={cfg}")
        if rng is None:
            root_seed = np.random.SeedSequence(random_seed)
        else:
            seed_rng = deepcopy(rng)
            entropy = seed_rng.integers(
                0, np.iinfo(np.uint32).max, size=5, dtype=np.uint32
            )
            root_seed = np.random.SeedSequence(entropy.tolist())
        child_seeds = root_seed.spawn(5)

        speed_aug = None
        if "speed_aug" in cfg:
            speed_aug = SpeedAugment.create(
                cfg["speed_aug"],
                rng=np.random.default_rng(seed=child_seeds[0]),
            )

        reverb_aug = None
        if "reverb_aug" in cfg:
            reverb_aug = ReverbAugment.create(
                cfg["reverb_aug"],
                rng=np.random.default_rng(seed=child_seeds[1]),
            )

        noise_aug = None
        if "noise_aug" in cfg:
            noise_aug = NoiseAugment.create(
                cfg["noise_aug"],
                rng=np.random.default_rng(seed=child_seeds[2]),
            )

        codec_aug = None
        if "codec_aug" in cfg:
            codec_aug = CodecAugment.create(
                cfg["codec_aug"],
                rng=np.random.default_rng(seed=child_seeds[3]),
            )

        transcodec_aug = None
        if "transcodec_aug" in cfg:
            transcodec_aug = CodecAugment.create(
                cfg["transcodec_aug"],
                rng=np.random.default_rng(seed=child_seeds[4]),
            )

        return cls(
            speed_aug=speed_aug,
            reverb_aug=reverb_aug,
            noise_aug=noise_aug,
            codec_aug=codec_aug,
            transcodec_aug=transcodec_aug,
        )

    @property
    def max_reverb_context(self) -> int:
        """Returns the maximum reverb context required by the pipeline.

        Args:
          None.

        Returns:
          Maximum left context in samples required by reverb augmentation.
        """
        if self.reverb_aug is None:
            return 0

        return self.reverb_aug.max_reverb_context

    def reseed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Reseeds all stochastic sub-augmenters with independent child streams."""
        augmenters = [
            self.speed_aug,
            self.reverb_aug,
            self.noise_aug,
            self.codec_aug,
            self.transcodec_aug,
        ]
        root_seed = (
            seed
            if isinstance(seed, np.random.SeedSequence)
            else np.random.SeedSequence(seed)
        )
        child_seeds = root_seed.spawn(len(augmenters))
        for i, augmenter in enumerate(augmenters):
            if augmenter is not None and hasattr(augmenter, "reseed"):
                augmenter.reseed(child_seeds[i])

    def forward(
        self,
        x: np.ndarray,
        sample_freq: Optional[float] = None,
        enable_tel_codecs: bool = True,
        enable_media_codecs: bool = True,
        enable_transcodec: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Adds speed augment, noise and reverberation to signal,
        speed multiplier, noise type, SNR, room type and RIRs are chosen randomly.

        Args:
          x: Clean speech signal.
          sample_freq: Sampling rate in Hz used by codec-based augmenters.
          enable_tel_codecs: Enables telephony codecs in ``codec_aug``.
          enable_media_codecs: Enables media codecs in ``codec_aug``.
          enable_transcodec: Enables second-stage codec augmentation.

        Returns:
          Augmented signal.
          Dictionary containing augmentation metadata for each enabled stage.
        """

        info = {}
        if self.speed_aug is not None:
            x, speed_info = self.speed_aug(x)
            info["speed"] = speed_info

        x_speed = x

        if self.reverb_aug is not None:
            x, reverb_info = self.reverb_aug(x)
            info["reverb"] = reverb_info
        else:
            info["reverb"] = {"rir_type": None, "srr": 100, "h_max": 1, "h_delay": 0}

        if self.noise_aug is not None:
            x, noise_info = self.noise_aug(x)
            info["noise"] = noise_info
        else:
            info["noise"] = {"noise_type": None, "snr": 100}

        if self.noise_aug is None:
            info["sdr"] = info["reverb"]["srr"]
        elif self.reverb_aug is None:
            info["sdr"] = info["noise"]["snr"]
        else:
            # we calculate SNR(dB) of the combined reverb + noise
            scale = info["reverb"]["h_max"]
            delay = info["reverb"]["h_delay"]
            info["sdr"] = ReverbAugment.sdr(x_speed, x, scale, delay)

        if self.codec_aug is not None:
            x, codec_info = self.codec_aug(
                x,
                sample_freq,
                enable_tel_codecs=enable_tel_codecs,
                enable_media_codecs=enable_media_codecs,
            )
            info["codec"] = codec_info
        else:
            info["codec"] = {"codec_type": None}

        if (
            self.transcodec_aug is not None
            and info["codec"]["codec_type"] is not None
            and enable_transcodec
        ):
            x, codec_info = self.transcodec_aug(x, sample_freq)
            info["transcodec"] = codec_info
        else:
            info["transcodec"] = {"codec_type": None}

        return x, info

    def __call__(
        self,
        x: np.ndarray,
        sample_freq: Optional[float] = None,
        enable_tel_codecs: bool = True,
        enable_media_codecs: bool = True,
        enable_transcodec: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Runs the augmentation pipeline using callable-style syntax.

        Args:
          x: Clean speech signal.
          sample_freq: Sampling rate in Hz used by codec-based augmenters.
          enable_tel_codecs: Enables telephony codecs in ``codec_aug``.
          enable_media_codecs: Enables media codecs in ``codec_aug``.
          enable_transcodec: Enables second-stage codec augmentation.

        Returns:
          Augmented signal.
          Dictionary containing augmentation metadata for each enabled stage.
        """
        return self.forward(
            x, sample_freq, enable_tel_codecs, enable_media_codecs, enable_transcodec
        )
