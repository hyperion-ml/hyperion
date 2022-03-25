"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from copy import deepcopy
import yaml
import numpy as np
from librosa.effects import time_stretch

from ...hyp_defs import float_cpu


class SpeedAugment(object):
    """Class to augment speech with speed perturbation.

    Attributes:
      speed_prob: probability of applying speed perturbation.
      speed_ratios: list of speed pertubation ratios.
      keep_length: applies padding or cropping to keep the lenght of the signal.
      random_seed: random seed for random number generator.
      rng:     Random number generator returned by
               np.random.RandomState (optional).
    """

    def __init__(
        self,
        speed_prob,
        speed_ratios=[0.9, 1.1],
        keep_length=False,
        random_seed=112358,
        rng=None,
    ):
        logging.info(
            "init speed augment with prob={}, speed_ratios={}, keep_length={}".format(
                speed_prob, speed_ratios, keep_length
            )
        )
        self.speed_prob = speed_prob
        self.speed_ratios = speed_ratios
        self.keep_length = keep_length

        if rng is None:
            self.rng = np.random.RandomState(seed=random_seed)
        else:
            self.rng = deepcopy(rng)

    @classmethod
    def create(cls, cfg, random_seed=112358, rng=None):
        """Creates a SpeedAugment object from options dictionary or YAML file.

        Args:
          cfg: YAML file path or dictionary with noise options.
          rng: Random number generator returned by
               np.random.RandomState (optional).

        Returns:
          NoiseAugment object.
        """
        if isinstance(cfg, str):
            with open(cfg, "r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

        assert isinstance(cfg, dict), "wrong object type for cfg={}".format(cfg)

        return cls(
            speed_prob=cfg["speed_prob"],
            speed_ratios=cfg["speed_ratios"],
            keep_length=cfg["keep_length"],
            random_seed=random_seed,
            rng=rng,
        )

    def forward(self, x):
        """Change the speed of the signal,
           the multiplication factor is chosen randomly.

        Args:
          x: clean speech signal.

        Returns:
          Augmented signal.
          Dictionary containing speed ratio applied.
        """

        # decide whether to add noise or not
        p = self.rng.random_sample()
        if p > self.speed_prob:
            # we don't add speed perturbation
            info = {"speed_ratio": 1}
            return x, info

        speed_idx = self.rng.choice(len(self.speed_ratios))
        # change speed
        r = self.speed_ratios[speed_idx]
        info = {"speed_ratio": r}
        y = time_stretch(x, r)
        # print(f"1 r={r} {x.shape} {y.shape}", flush=True)
        if self.keep_length:
            if r > 1:
                dither = np.max(x) / 2 ** 15  # we add some dither in the padding
                pad_y = dither * np.ones((x.shape[-1] - y.shape[-1],), dtype=y.dtype)
                y = np.concatenate((y, pad_y), axis=-1)
            elif r < 1:
                y = y[: x.shape[-1]]

        # print(f"2 r={r} {x.shape} {y.shape}", flush=True)
        return y, info

    def __call__(self, x):
        return self.forward(x)
