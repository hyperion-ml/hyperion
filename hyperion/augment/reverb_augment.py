"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import time
import logging
import math
import multiprocessing
import yaml
from copy import deepcopy
from enum import Enum

import numpy as np
from scipy import signal

from ..hyp_defs import float_cpu
from ..io import RandomAccessDataReaderFactory as DRF


class RIRNormType(Enum):
    """normalization type to apply to RIR."""

    NONE = 0  # none
    MAX = 1  # max ray normalized to 1
    ENERGY = 2  # energy of RIR normalized to 1


class SingleReverbAugment(object):
    """Class to augment speech with reverberation using RIR from a
        single type, e.g., small room, medium room, large room.

    Attributes:
      rir_type: string label indicating the RIR type.
      rir_path: Kaldi style rspecifier to Ark or H5 file containing RIRs.
      rir_norm: RIR normalization method between None, 'max' or 'energy'.
      comp_delay: compensate the delay introduced by the RIR if any,
                  this delay will happen if the maximum of the RIR is not in
                  its first sample.
      preload_rirs: if True all RIRS are loaded into RAM.
      rng:     Random number generator returned by
               np.random.RandomState (optional).
    """

    def __init__(
        self,
        rir_type,
        rir_path,
        rir_norm=None,
        comp_delay=True,
        preload_rirs=True,
        random_seed=112358,
        rng=None,
    ):
        self.rir_type = rir_type
        logging.info(
            (
                "init reverb_augment with RIR={} rir_path={} "
                "rir_norm={} comp_delay={}"
            ).format(rir_type, rir_path, rir_norm, comp_delay)
        )
        self.r = DRF.create(rir_path)
        # logging.info('init reverb_augment with RIR={} read RIR lengths'.format(rir_type))
        self.rir_keys = self.r.keys
        self.preload_rirs = preload_rirs
        if preload_rirs:
            self.rirs = self.r.read(self.rir_keys)
            self.r.close()
        else:
            self.rirs = None

        if rir_norm is None:
            self.rir_norm = RIRNormType.NONE
        elif rir_norm == "max":
            self.rir_norm = RIRNormType.MAX
        elif rir_norm == "energy":
            self.rir_norm = RIRNormType.ENERGY

        self.comp_delay = comp_delay

        self.lock = multiprocessing.Lock()
        if rng is None:
            self.rng = np.random.RandomState(seed=random_seed)
        else:
            self.rng = deepcopy(rng)

        logging.info("init reverb_augment with RIR={} done".format(rir_type))

    @staticmethod
    def _power(x):
        """Computes power of x in dB."""
        return 10 * np.log10((x ** 2).sum() + 1e-5)

    @staticmethod
    def sdr(x, y, scale, delay):
        """Computes SDR in DB.

        Args:
          x: clean speech signal.
          y: reverberant speech signal.
          scale: linear gain of the RIR.
          delay: delay introduced by the RIR.
        """

        x = scale * x
        n = y[delay:] - x
        return SingleReverbAugment._power(x) - SingleReverbAugment._power(n)

    def _norm_rir(self, h):
        """Normalizes RIR by max value or power"""
        if self.rir_norm == RIRNormType.NONE:
            return h
        if self.rir_norm == RIRNormType.MAX:
            idx = np.argmax(np.abs(h))
            return h / h[idx]

        return h / np.sum(h ** 2)

    def forward(self, x):
        """Adds reverberation to signal, RIR is chosen randomly.

        Args:
          x: clean speech signal.

        Returns:
          Noisy signal.
          Dictionary containing information of RIR type, Signal reverb ratio (dB), linear gain and delay introduced by RIR.
        """

        num_samples = x.shape[0]
        with self.lock:
            rir_idx = self.rng.randint(len(self.rir_keys))

        if self.preload_rirs:
            h = self.rirs[rir_idx]
        else:
            key = self.rir_keys[rir_idx]
            h = self.r.read([key])[0]

        h = self._norm_rir(h)
        h_delay = np.argmax(np.abs(h))
        h_max = h[h_delay]
        y = signal.fftconvolve(x, h)
        if self.comp_delay:
            y = y[h_delay : num_samples + h_delay]
            h_delay = 0
        else:
            y = y[: num_samples + h_delay]

        srr = self.sdr(x, y, h_max, h_delay)
        # logging.info('rt={} {} {} {} {}'.format(t2-t1, t3-t2, t4-t3, t5-t4, t6-t5))
        info = {
            "rir_type": self.rir_type,
            "srr": srr,
            "h_max": h_max,
            "h_delay": h_delay,
        }
        return y, info

    def __call__(self, x):
        return self.forward(x)


class ReverbAugment(object):
    """Class to augment speech with reverberation with RIRS from multiple types,
        e.g., small room, medium room, large room.
        It will randomly choose which RIR type to add.

    Attributes:
      reverb_prob: probability of adding reverberation.
      rir_types: dictionary of options with one entry per RIR-type,
                  Each entry is also a dictiory with the following entries:
                  weight, rir_norm, comp_delay, rir_path. The weight parameter
                  is proportional to how often we want to sample a given RIR
                  type.
      max_reverb_context: number of samples required as left context
                          for the convolution operation.
      rng:     Random number generator returned by
               np.random.RandomState (optional).
    """

    def __init__(
        self, reverb_prob, rir_types, max_reverb_context=0, random_seed=112358, rng=None
    ):

        logging.info("init reverb_augment")
        self.reverb_prob = reverb_prob
        assert isinstance(rir_types, dict)
        num_rir_types = len(rir_types)

        augmenters = []
        self.weights = np.zeros((len(rir_types),))
        count = 0
        val_opts = ("rir_path", "rir_norm", "comp_delay", "preload_rirs")
        for key, opts in rir_types.items():
            self.weights[count] = opts["weight"]

            opts_i = {}
            for opt_key in val_opts:
                if opt_key in opts:
                    opts_i[opt_key] = opts[opt_key]

            aug = SingleReverbAugment(key, **opts_i, random_seed=random_seed, rng=rng)
            augmenters.append(aug)
            count += 1

        self.max_reverb_context = max_reverb_context
        self.weights /= np.sum(self.weights)
        self.augmenters = augmenters

        self.lock = multiprocessing.Lock()
        if rng is None:
            self.rng = np.random.RandomState(seed=random_seed)
        else:
            self.rng = deepcopy(rng)

    @classmethod
    def create(cls, cfg, random_seed=112358, rng=None):
        """Creates a ReverbAugment object from options dictionary or YAML file.

        Args:
          cfg: YAML file path or dictionary with reverb options.
          rng: Random number generator returned by
               np.random.RandomState (optional).

        Returns:
          ReverbAugment object.
        """

        if isinstance(cfg, str):
            with open(cfg, "r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

        assert isinstance(cfg, dict), "wrong object type for cfg={}".format(cfg)

        return cls(
            reverb_prob=cfg["reverb_prob"],
            rir_types=cfg["rir_types"],
            max_reverb_context=cfg["max_reverb_context"],
            random_seed=random_seed,
            rng=rng,
        )

    @staticmethod
    def sdr(x, y, scale, delay):
        """Computes SDR in DB.

        Args:
          x: clean speech signal.
          y: reverberant speech signal.
          scale: linear gain of the RIR.
          delay: delay introduced by the RIR.
        """
        return SingleReverbAugment.sdr(x, y, scale, delay)

    def forward(self, x):
        """Adds reverberation to signal, Room type is choosen randomly,
           RIR is chosen randomly.

        Args:
          x: clean speech signal.

        Returns:
          Noisy signal.
          Dictionary containing information of RIR type, Signal reverb ratio (dB), linear gain and delay introduced by RIR.
        """

        # decide whether to add reverb or not
        with self.lock:
            p = self.rng.random_sample()

        if p > self.reverb_prob:
            # we don't add reverb
            info = {"rir_type": None, "srr": 100, "h_max": 1, "h_delay": 0}
            return x, info

        # decide the RIR type
        with self.lock:
            rir_idx = self.rng.choice(len(self.weights), p=self.weights)

        # add reverb
        x, info = self.augmenters[rir_idx](x)
        return x, info

    def __call__(self, x):
        return self.forward(x)
