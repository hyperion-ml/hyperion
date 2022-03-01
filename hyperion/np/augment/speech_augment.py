"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
import yaml

import numpy as np

from ...hyp_defs import float_cpu

from .noise_augment import NoiseAugment
from .reverb_augment import ReverbAugment
from .speed_augment import SpeedAugment


class SpeechAugment(object):
    """Class to change speedd, add noise and reverberation
       on-the-fly when training nnets.

    Attributes:
       speed_aug: SpeedAugment object.
       reverb_aug: ReverbAugment object.
       noise_aug: NoiseAugment object.
    """

    def __init__(self, speed_aug=None, reverb_aug=None, noise_aug=None):
        self.speed_aug = speed_aug
        self.reverb_aug = reverb_aug
        self.noise_aug = noise_aug

    @classmethod
    def create(cls, cfg, random_seed=112358, rng=None):
        """Creates a SpeechAugment object from options dictionary or YAML file.

        Args:
          cfg: YAML file path or dictionary with noise options.
          rng: Random number generator returned by
               np.random.RandomState (optional).

        Returns:
          SpeechAugment object.
        """
        if isinstance(cfg, str):
            with open(cfg, "r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

        assert isinstance(cfg, dict), "wrong object type for cfg={}".format(cfg)

        speed_aug = None
        if "speed_aug" in cfg:
            speed_aug = SpeedAugment.create(cfg["speed_aug"], random_seed=random_seed)

        reverb_aug = None
        if "reverb_aug" in cfg:
            reverb_aug = ReverbAugment.create(
                cfg["reverb_aug"], random_seed=random_seed
            )

        noise_aug = None
        if "noise_aug" in cfg:
            noise_aug = NoiseAugment.create(cfg["noise_aug"], random_seed=random_seed)

        return cls(speed_aug=speed_aug, reverb_aug=reverb_aug, noise_aug=noise_aug)

    @property
    def max_reverb_context(self):
        """Maximum length of the RIRs."""
        if self.reverb_aug is None:
            return 0

        return self.reverb_aug.max_reverb_context

    def forward(self, x):
        """Adds speed augment, noise and reverberation to signal,
        speed multiplier, noise type, SNR, room type and RIRs are chosen randomly.

        Args:
          x: clean speech signal.

        Returns:
          Augmented signal
          Dictionary containing information of noise type, rir_type, SNR(dB), SDR(dB), speed.
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

        return x, info

    def __call__(self, x):
        return self.forward(x)
