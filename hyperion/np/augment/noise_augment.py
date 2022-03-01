"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
import multiprocessing
import yaml
from copy import deepcopy

import numpy as np

from ...hyp_defs import float_cpu
from ...io import RandomAccessAudioReader as AR


class SingleNoiseAugment(object):
    """Class to augment speech with additive noise of a single type,
        e.g., music, babble, ...

    Attributes:
      noise_type: string label indicating the noise type.
      noise_path: path to Kaldi style wav.scp file indicating the path
                  to the noise wav files.
      min_snr: mininimum SNR(dB) to sample from.
      max_snr: maximum SNR(dB) to sample from.
      rng:     Random number generator returned by
               np.random.RandomState (optional).
    """

    def __init__(
        self, noise_type, noise_path, min_snr, max_snr, random_seed=112358, rng=None
    ):
        logging.info(
            "init noise_augment with noise={} noise_path={} snr={}-{}".format(
                noise_type, noise_path, min_snr, max_snr
            )
        )

        self.noise_type = noise_type
        self.r = AR(noise_path)
        self.noise_keys = self.r.keys
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.cache = None
        self.lock = multiprocessing.Lock()
        if rng is None:
            self.rng = np.random.RandomState(seed=random_seed)
        else:
            self.rng = deepcopy(rng)

        logging.info("init noise_augment with noise={} done".format(noise_type))

    @staticmethod
    def _power(x):
        """Computes power of x in dB."""
        return 10 * np.log10((x ** 2).sum())

    @staticmethod
    def snr(x, n):
        """Computes SNR in dB.

        Args:
          x: clean speech signal.
          n: noise signal.
        """
        return SingleNoiseAugment._power(x) - SingleNoiseAugment._power(n)

    @staticmethod
    def _compute_noise_scale(x, n, target_snr):
        snr = SingleNoiseAugment.snr(x, n)
        return 10 ** ((snr - target_snr) / 20)

    def forward(self, x):
        """Adds noise to signal, SNR is chosen randomly.

        Args:
          x: clean speech signal.

        Returns:
          Noisy signal.
          Dictionary containing information of noise type and SNR(dB).
        """
        num_samples = x.shape[0]
        with self.lock:
            if self.cache is not None:
                if self.cache.shape[0] > num_samples:
                    noise = self.cache[:num_samples]
                    self.cache = self.cache[num_samples:]
                else:
                    noise = self.cache
                    self.cache = None
            else:
                noise = None

        while noise is None or noise.shape[0] < num_samples:
            with self.lock:
                noise_idx = self.rng.randint(len(self.noise_keys))
                key = self.noise_keys[noise_idx]
                noise_k, fs_k = self.r.read([key])
                noise_k = noise_k[0]

            if noise is None:
                need_samples = min(x.shape[0], noise_k.shape[0])
                noise = noise_k[:need_samples]
            else:
                need_samples = min(x.shape[0] - noise.shape[0], noise_k.shape[0])
                noise = np.concatenate((noise, noise_k[:need_samples]))

            if need_samples < noise_k.shape[0]:
                with self.lock:
                    self.cache = noise_k[need_samples:]

        with self.lock:
            target_snr = self.rng.uniform(self.min_snr, self.max_snr)
        scale = self._compute_noise_scale(x, noise, target_snr)

        info = {"noise_type": self.noise_type, "snr": target_snr}
        return x + scale * noise, info

    def __call__(self, x):
        return self.forward(x)


class NoiseAugment(object):
    """Class to augment speech with additive noise from multiple types,
        e.g., music, babble, ...
        It will randomly choose which noise type to add.

    Attributes:
      noise_prob: probability of adding noise.
      noise_types: dictionary of options with one entry per noise-type,
                  Each entry is also a dictiory with the following entries:
                  weight, max_snr, min_snr, noise_path. The weight parameter
                  is proportional to how often we want to sample a given noise
                  type.
      rng:     Random number generator returned by
               np.random.RandomState (optional).
    """

    def __init__(self, noise_prob, noise_types, random_seed=112358, rng=None):
        logging.info("init noise augment")
        self.noise_prob = noise_prob
        assert isinstance(noise_types, dict)
        # num_noise_types = len(noise_types)

        augmenters = []
        self.weights = np.zeros((len(noise_types),))
        count = 0
        for key, opts in noise_types.items():
            self.weights[count] = opts["weight"]
            aug = SingleNoiseAugment(
                key,
                opts["noise_path"],
                opts["min_snr"],
                opts["max_snr"],
                random_seed=random_seed,
                rng=rng,
            )
            augmenters.append(aug)
            count += 1

        self.weights /= np.sum(self.weights)
        self.augmenters = augmenters

        self.lock = multiprocessing.Lock()
        if rng is None:
            self.rng = np.random.RandomState(seed=random_seed)
        else:
            self.rng = deepcopy(rng)

    @classmethod
    def create(cls, cfg, random_seed=112358, rng=None):
        """Creates a NoiseAugment object from options dictionary or YAML file.

        Args:
          cfg: YAML file path or dictionary with noise options.
          rng: Random number generator returned by
               np.random.RandomState (optional).

        Returns:
          NoiseAugment object
        """
        if isinstance(cfg, str):
            with open(cfg, "r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

        assert isinstance(cfg, dict), "wrong object type for cfg={}".format(cfg)

        return cls(
            noise_prob=cfg["noise_prob"],
            noise_types=cfg["noise_types"],
            random_seed=random_seed,
            rng=rng,
        )

    def forward(self, x):
        """Adds noise to signal, noise type and SNR are chosen randomly.

        Args:
          x: clean speech signal.

        Returns:
          Noisy signal.
          Dictionary containing information of noise type and SNR(dB).
        """

        # decide whether to add noise or not
        with self.lock:
            p = self.rng.random_sample()

        if p > self.noise_prob:
            # we don't add noise
            info = {"noise_type": None, "snr": 100}
            return x, info

        # decide the noise type
        with self.lock:
            noise_idx = self.rng.choice(len(self.weights), p=self.weights)

        # add noise
        x, info = self.augmenters[noise_idx](x)
        return x, info

    def __call__(self, x):
        return self.forward(x)
