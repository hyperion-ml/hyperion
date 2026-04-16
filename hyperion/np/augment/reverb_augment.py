"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import yaml
from scipy import signal

from ...io import RandomAccessDataReaderFactory as DRF


class RIRNormType(str, Enum):
    """Normalization type applied to room impulse responses (RIRs)."""

    NONE = "none"
    MAX = "max"
    ENERGY = "energy"


class SingleReverbAugment:
    """Augments speech with reverberation from a single RIR category.

    Attributes:
      rir_type: RIR category label (for example, small room, medium room).
      r: Random-access reader for RIR waveforms.
      rir_keys: Keys available in the RIR reader.
      preload_rirs: If ``True``, all RIRs are loaded into memory at initialization.
      rirs: In-memory RIR cache when ``preload_rirs`` is enabled.
      rir_norm: Normalization mode for selected RIRs.
      comp_delay: If ``True``, compensates RIR peak delay after convolution.
      rng: Random number generator used for RIR sampling.
    """

    def __init__(
        self,
        rir_type: str,
        rir_path: str,
        rir_norm: Optional[Union[str, RIRNormType]] = None,
        comp_delay: bool = True,
        preload_rirs: bool = True,
        random_seed: int = 112358,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Initializes a single-reverb augmenter.

        Args:
          rir_type: RIR category label.
          rir_path: Kaldi-style rspecifier to an Ark or H5 container with RIRs.
          rir_norm: RIR normalization mode (``None``, ``"none"``, ``"max"``, or ``"energy"``).
          comp_delay: If ``True``, removes delay introduced by the RIR peak location.
          preload_rirs: If ``True``, loads all RIRs into memory.
          random_seed: Seed used when creating a new random generator.
          rng: Optional pre-created random generator.

        Returns:
          None.
        """
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
        if len(self.rir_keys) == 0:
            self.r.close()
            raise ValueError(
                f"rir_path={rir_path} contains no RIRs for rir_type={rir_type}"
            )
        self.preload_rirs = preload_rirs
        if preload_rirs:
            self.rirs = self.r.read(self.rir_keys)
            self.r.close()
        else:
            self.rirs = None

        if rir_norm is None:
            self.rir_norm = RIRNormType.NONE
        elif isinstance(rir_norm, RIRNormType):
            self.rir_norm = rir_norm
        elif isinstance(rir_norm, str):
            try:
                self.rir_norm = RIRNormType(rir_norm.strip().lower())
            except ValueError as err:
                valid = [v.value for v in RIRNormType]
                raise ValueError(
                    f"Invalid rir_norm='{rir_norm}'. Expected one of {valid} or None."
                ) from err
        else:
            raise TypeError(
                f"rir_norm must be None, str, or RIRNormType, got {type(rir_norm)}"
            )

        self.comp_delay = comp_delay

        if rng is None:
            self.rng = np.random.default_rng(seed=random_seed)
        else:
            self.rng = deepcopy(rng)

        logging.info("init reverb_augment with RIR={} done".format(rir_type))

    @staticmethod
    def _power(x: np.ndarray) -> float:
        """Computes signal power in dB.

        Args:
          x: Input waveform.

        Returns:
          Signal power in dB.
        """
        return 10 * np.log10((x**2).sum() + 1e-5)

    @staticmethod
    def sdr(x: np.ndarray, y: np.ndarray, scale: float, delay: int) -> float:
        """Computes SDR in DB.

        Args:
          x: Clean speech signal.
          y: Reverberant speech signal.
          scale: Linear gain of the RIR.
          delay: Delay introduced by the RIR.

        Returns:
          Signal-to-distortion ratio in dB.
        """

        x = scale * x
        n = y[delay:] - x
        return SingleReverbAugment._power(x) - SingleReverbAugment._power(n)

    def _norm_rir(self, h: np.ndarray) -> np.ndarray:
        """Normalizes an RIR according to the configured normalization mode.

        Args:
          h: Input room impulse response waveform.

        Returns:
          Normalized room impulse response.
        """
        if h.size == 0:
            raise ValueError(f"RIR type '{self.rir_type}' has empty impulse response")

        if self.rir_norm == RIRNormType.NONE:
            return h
        if self.rir_norm == RIRNormType.MAX:
            idx = np.argmax(np.abs(h))
            h_max = h[idx]
            if np.abs(h_max) < 1e-12:
                raise ValueError(
                    f"RIR type '{self.rir_type}' has near-zero peak; cannot apply max normalization"
                )
            return h / h_max

        energy = np.sum(h**2)
        if energy < 1e-12:
            raise ValueError(
                f"RIR type '{self.rir_type}' has near-zero energy; cannot apply energy normalization"
            )
        return h / np.sqrt(energy)

    def forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Union[str, float, int]]]:
        """Adds reverberation to signal, RIR is chosen randomly.

        Args:
          x: Clean speech signal.

        Returns:
          Reverberant signal.
          Dictionary containing RIR type, signal-to-reverb ratio (dB), linear gain, and delay.
        """
        if x.ndim != 1:
            raise ValueError(
                f"SingleReverbAugment expects a 1-D waveform, got shape={x.shape}"
            )
        num_samples = x.shape[0]
        rir_idx = self.rng.integers(len(self.rir_keys))

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
        info = {
            "rir_type": self.rir_type,
            "srr": srr,
            "h_max": h_max,
            "h_delay": h_delay,
        }
        # avg proc time 0.004 secs
        return y, info

    def __call__(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Union[str, float, int]]]:
        """Runs reverberation augmentation using callable-style syntax.

        Args:
          x: Clean speech signal.

        Returns:
          Reverberant signal.
          Dictionary containing RIR type, signal-to-reverb ratio (dB), linear gain, and delay.
        """
        return self.forward(x)

    def reseed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Reseeds the internal RNG."""
        self.rng = np.random.default_rng(seed=seed)


class ReverbAugment:
    """Augments speech with reverberation sampled from multiple RIR categories.

    Attributes:
      reverb_prob: Probability of applying reverberation to an input utterance.
      max_reverb_context: Required left-context (in seconds) for convolution.
      weights: Sampling weights associated with ``augmenters``.
      augmenters: Per-RIR-type augmenters.
      rng: Random number generator used for augmentation decisions.
    """

    def __init__(
        self,
        reverb_prob: float,
        rir_types: Dict[str, Dict[str, Any]],
        max_reverb_context: float = 0.0,
        random_seed: int = 112358,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Initializes a multi-reverb augmenter.

        Args:
          reverb_prob: Probability of applying reverberation.
          rir_types: Dictionary from RIR type name to configuration dictionary.
          max_reverb_context: Required left-context (in seconds) for convolution.
          random_seed: Seed used when creating a new random generator.
          rng: Optional pre-created random generator.

        Returns:
          None.
        """

        logging.info("init reverb_augment")
        if not np.isscalar(reverb_prob):
            raise TypeError(
                f"reverb_prob must be a scalar value, got {type(reverb_prob)}"
            )
        reverb_prob = float(reverb_prob)
        if not np.isfinite(reverb_prob) or reverb_prob < 0 or reverb_prob > 1:
            raise ValueError(f"reverb_prob must be in [0, 1], got {reverb_prob}")

        if not isinstance(rir_types, dict):
            raise TypeError(
                f"rir_types must be a dict from RIR type to options, got {type(rir_types)}"
            )
        if len(rir_types) == 0:
            raise ValueError("rir_types must contain at least one RIR type")

        if not np.isscalar(max_reverb_context):
            raise TypeError(
                "max_reverb_context must be a scalar duration in seconds, "
                f"got {type(max_reverb_context)}"
            )
        max_reverb_context = float(max_reverb_context)
        if not np.isfinite(max_reverb_context) or max_reverb_context < 0:
            raise ValueError(
                "max_reverb_context must be a finite value >= 0 seconds, "
                f"got {max_reverb_context}"
            )

        self.reverb_prob = reverb_prob

        if rng is None:
            root_seed = np.random.SeedSequence(random_seed)
        else:
            seed_rng = deepcopy(rng)
            entropy = seed_rng.integers(
                0, np.iinfo(np.uint32).max, size=4, dtype=np.uint32
            )
            root_seed = np.random.SeedSequence(entropy.tolist())
        child_seeds = root_seed.spawn(len(rir_types) + 1)
        self.rng = np.random.default_rng(seed=child_seeds[0])

        augmenters = []
        self.weights = np.zeros((len(rir_types),))
        count = 0
        val_opts = ("rir_path", "rir_norm", "comp_delay", "preload_rirs")
        for key, opts in rir_types.items():
            if not isinstance(opts, dict):
                raise TypeError(f"rir_types['{key}'] must be a dict, got {type(opts)}")
            required_keys = ("weight", "rir_path")
            for cfg_key in required_keys:
                if cfg_key not in opts:
                    raise KeyError(
                        f"rir_types['{key}'] is missing required key '{cfg_key}'"
                    )

            weight = opts["weight"]
            if not np.isscalar(weight):
                raise TypeError(
                    f"rir_types['{key}']['weight'] must be a scalar value, got {type(weight)}"
                )
            weight = float(weight)
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(
                    f"rir_types['{key}']['weight'] must be a non-negative finite value, got {weight}"
                )
            self.weights[count] = weight

            opts_i = {}
            for opt_key in val_opts:
                if opt_key in opts:
                    opts_i[opt_key] = opts[opt_key]

            child_rng = np.random.default_rng(seed=child_seeds[count + 1])
            aug = SingleReverbAugment(key, **opts_i, rng=child_rng)
            augmenters.append(aug)
            count += 1

        self.max_reverb_context = max_reverb_context
        weights_sum = np.sum(self.weights)
        if weights_sum <= 0:
            raise ValueError("rir_types weights must sum to a value > 0")
        self.weights /= weights_sum
        self.augmenters = augmenters

    @classmethod
    def create(
        cls,
        cfg: Union[str, Dict[str, Any]],
        random_seed: int = 112358,
        rng: Optional[np.random.Generator] = None,
    ) -> "ReverbAugment":
        """Creates a ReverbAugment object from options dictionary or YAML file.

        Args:
          cfg: YAML file path or dictionary with reverb options.
          random_seed: Seed used when creating a new random generator.
          rng: Optional pre-created random generator.

        Returns:
          Configured reverb augmenter instance.
        """

        if isinstance(cfg, str):
            with open(cfg, "r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

        if not isinstance(cfg, dict):
            raise TypeError(f"wrong object type for cfg={cfg}")

        return cls(
            reverb_prob=cfg["reverb_prob"],
            rir_types=cfg["rir_types"],
            max_reverb_context=cfg.get("max_reverb_context", 0.0),
            random_seed=random_seed,
            rng=rng,
        )

    @staticmethod
    def sdr(x: np.ndarray, y: np.ndarray, scale: float, delay: int) -> float:
        """Computes SDR in DB.

        Args:
          x: Clean speech signal.
          y: Reverberant speech signal.
          scale: Linear gain of the RIR.
          delay: Delay introduced by the RIR.

        Returns:
          Signal-to-distortion ratio in dB.
        """
        return SingleReverbAugment.sdr(x, y, scale, delay)

    def forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Union[Optional[str], float, int]]]:
        """Adds reverberation to signal, Room type is choosen randomly,
           RIR is chosen randomly.

        Args:
          x: Clean speech signal.

        Returns:
          Reverberant signal.
          Dictionary containing RIR type, signal-to-reverb ratio (dB), linear gain, and delay.
        """

        # decide whether to add reverb or not
        p = self.rng.random()

        if p > self.reverb_prob:
            # we don't add reverb
            info = {"rir_type": None, "srr": 100, "h_max": 1, "h_delay": 0}
            return x, info

        # decide the RIR type
        rir_idx = self.rng.choice(len(self.weights), p=self.weights)

        # add reverb
        x, info = self.augmenters[rir_idx](x)
        return x, info

    def __call__(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Union[Optional[str], float, int]]]:
        """Runs reverberation augmentation using callable-style syntax.

        Args:
          x: Clean speech signal.

        Returns:
          Reverberant signal.
          Dictionary containing RIR type, signal-to-reverb ratio (dB), linear gain, and delay.
        """
        return self.forward(x)

    def reseed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Reseeds this augmenter and all child augmenters."""
        root_seed = (
            seed
            if isinstance(seed, np.random.SeedSequence)
            else np.random.SeedSequence(seed)
        )
        child_seeds = root_seed.spawn(len(self.augmenters) + 1)
        self.rng = np.random.default_rng(seed=child_seeds[0])
        for i, augmenter in enumerate(self.augmenters):
            augmenter.reseed(child_seeds[i + 1])


# class SingleReverbAugment(object):
#     """Augments speech with reverberation from a single RIR category.

#     Attributes:
#       rir_type: RIR category label (for example, small room, medium room).
#       r: Random-access reader for RIR waveforms.
#       rir_keys: Keys available in the RIR reader.
#       preload_rirs: If ``True``, all RIRs are loaded into memory at initialization.
#       rirs: In-memory RIR cache when ``preload_rirs`` is enabled.
#       rir_norm: Normalization mode for selected RIRs.
#       comp_delay: If ``True``, compensates RIR peak delay after convolution.
#       lock: Mutex used to protect RNG access.
#       rng: Random number generator used for RIR sampling.
#     """

#     def __init__(
#         self,
#         rir_type: str,
#         rir_path: str,
#         rir_norm: Optional[str] = None,
#         comp_delay: bool = True,
#         preload_rirs: bool = True,
#         random_seed: int = 112358,
#         rng: Optional[np.random.Generator] = None,
#     ) -> None:
#         """Initializes a single-reverb augmenter.

#         Args:
#           rir_type: RIR category label.
#           rir_path: Kaldi-style rspecifier to an Ark or H5 container with RIRs.
#           rir_norm: RIR normalization mode (``None``, ``"max"``, or ``"energy"``).
#           comp_delay: If ``True``, removes delay introduced by the RIR peak location.
#           preload_rirs: If ``True``, loads all RIRs into memory.
#           random_seed: Seed used when creating a new random generator.
#           rng: Optional pre-created random generator.

#         Returns:
#           None.
#         """
#         self.rir_type = rir_type
#         logging.info(
#             (
#                 "init reverb_augment with RIR={} rir_path={} "
#                 "rir_norm={} comp_delay={}"
#             ).format(rir_type, rir_path, rir_norm, comp_delay)
#         )
#         self.r = DRF.create(rir_path)
#         # logging.info('init reverb_augment with RIR={} read RIR lengths'.format(rir_type))
#         self.rir_keys = self.r.keys
#         self.preload_rirs = preload_rirs
#         if preload_rirs:
#             self.rirs = self.r.read(self.rir_keys)
#             self.r.close()
#         else:
#             self.rirs = None

#         if rir_norm is None:
#             self.rir_norm = RIRNormType.NONE
#         elif rir_norm == "max":
#             self.rir_norm = RIRNormType.MAX
#         elif rir_norm == "energy":
#             self.rir_norm = RIRNormType.ENERGY

#         self.comp_delay = comp_delay

#         self.lock = multiprocessing.Lock()
#         if rng is None:
#             self.rng = np.random.default_rng(seed=random_seed)
#         else:
#             self.rng = deepcopy(rng)

#         logging.info("init reverb_augment with RIR={} done".format(rir_type))

#     @staticmethod
#     def _power(x: np.ndarray) -> float:
#         """Computes signal power in dB.

#         Args:
#           x: Input waveform.

#         Returns:
#           Signal power in dB.
#         """
#         return 10 * np.log10((x**2).sum() + 1e-5)

#     @staticmethod
#     def sdr(x: np.ndarray, y: np.ndarray, scale: float, delay: int) -> float:
#         """Computes SDR in DB.

#         Args:
#           x: Clean speech signal.
#           y: Reverberant speech signal.
#           scale: Linear gain of the RIR.
#           delay: Delay introduced by the RIR.

#         Returns:
#           Signal-to-distortion ratio in dB.
#         """

#         x = scale * x
#         n = y[delay:] - x
#         return SingleReverbAugment._power(x) - SingleReverbAugment._power(n)

#     def _norm_rir(self, h: np.ndarray) -> np.ndarray:
#         """Normalizes an RIR according to the configured normalization mode.

#         Args:
#           h: Input room impulse response waveform.

#         Returns:
#           Normalized room impulse response.
#         """
#         if self.rir_norm == RIRNormType.NONE:
#             return h
#         if self.rir_norm == RIRNormType.MAX:
#             idx = np.argmax(np.abs(h))
#             return h / h[idx]

#         return h / np.sum(h**2)

#     def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Union[str, float, int]]]:
#         """Adds reverberation to signal, RIR is chosen randomly.

#         Args:
#           x: Clean speech signal.

#         Returns:
#           Reverberant signal.
#           Dictionary containing RIR type, signal-to-reverb ratio (dB), linear gain, and delay.
#         """
#         t1 = time.time()
#         num_samples = x.shape[0]
#         with self.lock:
#             rir_idx = self.rng.integers(len(self.rir_keys))

#         if self.preload_rirs:
#             h = self.rirs[rir_idx]
#         else:
#             key = self.rir_keys[rir_idx]
#             h = self.r.read([key])[0]

#         h = self._norm_rir(h)
#         h_delay = np.argmax(np.abs(h))
#         h_max = h[h_delay]
#         y = signal.fftconvolve(x, h)
#         if self.comp_delay:
#             y = y[h_delay : num_samples + h_delay]
#             h_delay = 0
#         else:
#             y = y[: num_samples + h_delay]

#         srr = self.sdr(x, y, h_max, h_delay)
#         info = {
#             "rir_type": self.rir_type,
#             "srr": srr,
#             "h_max": h_max,
#             "h_delay": h_delay,
#         }
#         # avg proc time 0.004 secs
#         return y, info

#     def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Union[str, float, int]]]:
#         """Runs reverberation augmentation using callable-style syntax.

#         Args:
#           x: Clean speech signal.

#         Returns:
#           Reverberant signal.
#           Dictionary containing RIR type, signal-to-reverb ratio (dB), linear gain, and delay.
#         """
#         return self.forward(x)

#     def reseed(self, seed: Union[int, np.random.SeedSequence]) -> None:
#         """Reseeds the internal RNG."""
#         self.rng = np.random.default_rng(seed=seed)


# class ReverbAugment(object):
#     """Augments speech with reverberation sampled from multiple RIR categories.

#     Attributes:
#       reverb_prob: Probability of applying reverberation to an input utterance.
#       max_reverb_context: Required left-context (in samples) for convolution.
#       weights: Sampling weights associated with ``augmenters``.
#       augmenters: Per-RIR-type augmenters.
#       lock: Mutex used to protect RNG access.
#       rng: Random number generator used for augmentation decisions.
#     """

#     def __init__(
#         self,
#         reverb_prob: float,
#         rir_types: Mapping[str, Mapping[str, Any]],
#         max_reverb_context: int = 0,
#         random_seed: int = 112358,
#         rng: Optional[np.random.Generator] = None,
#     ) -> None:
#         """Initializes a multi-reverb augmenter.

#         Args:
#           reverb_prob: Probability of applying reverberation.
#           rir_types: Mapping from RIR type name to configuration dictionary.
#           max_reverb_context: Required left-context (in samples) for convolution.
#           random_seed: Seed used when creating a new random generator.
#           rng: Optional pre-created random generator.

#         Returns:
#           None.
#         """

#         logging.info("init reverb_augment")
#         self.reverb_prob = reverb_prob
#         assert isinstance(rir_types, dict)
#         num_rir_types = len(rir_types)

#         augmenters = []
#         self.weights = np.zeros((len(rir_types),))
#         count = 0
#         val_opts = ("rir_path", "rir_norm", "comp_delay", "preload_rirs")
#         for key, opts in rir_types.items():
#             self.weights[count] = opts["weight"]

#             opts_i = {}
#             for opt_key in val_opts:
#                 if opt_key in opts:
#                     opts_i[opt_key] = opts[opt_key]

#             aug = SingleReverbAugment(key, **opts_i, random_seed=random_seed, rng=rng)
#             augmenters.append(aug)
#             count += 1

#         self.max_reverb_context = max_reverb_context
#         self.weights /= np.sum(self.weights)
#         self.augmenters = augmenters

#         self.lock = multiprocessing.Lock()
#         if rng is None:
#             self.rng = np.random.default_rng(seed=random_seed)
#         else:
#             self.rng = deepcopy(rng)

#     @classmethod
#     def create(
#         cls,
#         cfg: Union[str, Mapping[str, Any]],
#         random_seed: int = 112358,
#         rng: Optional[np.random.Generator] = None,
#     ) -> "ReverbAugment":
#         """Creates a ReverbAugment object from options dictionary or YAML file.

#         Args:
#           cfg: YAML file path or dictionary with reverb options.
#           random_seed: Seed used when creating a new random generator.
#           rng: Optional pre-created random generator.

#         Returns:
#           Configured reverb augmenter instance.
#         """

#         if isinstance(cfg, str):
#             with open(cfg, "r") as f:
#                 cfg = yaml.load(f, Loader=yaml.FullLoader)

#         assert isinstance(cfg, dict), "wrong object type for cfg={}".format(cfg)

#         return cls(
#             reverb_prob=cfg["reverb_prob"],
#             rir_types=cfg["rir_types"],
#             max_reverb_context=cfg["max_reverb_context"],
#             random_seed=random_seed,
#             rng=rng,
#         )

#     @staticmethod
#     def sdr(x: np.ndarray, y: np.ndarray, scale: float, delay: int) -> float:
#         """Computes SDR in DB.

#         Args:
#           x: Clean speech signal.
#           y: Reverberant speech signal.
#           scale: Linear gain of the RIR.
#           delay: Delay introduced by the RIR.

#         Returns:
#           Signal-to-distortion ratio in dB.
#         """
#         return SingleReverbAugment.sdr(x, y, scale, delay)

#     def forward(
#         self, x: np.ndarray
#     ) -> Tuple[np.ndarray, Dict[str, Union[Optional[str], float, int]]]:
#         """Adds reverberation to signal, Room type is choosen randomly,
#            RIR is chosen randomly.

#         Args:
#           x: Clean speech signal.

#         Returns:
#           Reverberant signal.
#           Dictionary containing RIR type, signal-to-reverb ratio (dB), linear gain, and delay.
#         """

#         # decide whether to add reverb or not
#         with self.lock:
#             p = self.rng.random()

#         if p > self.reverb_prob:
#             # we don't add reverb
#             info = {"rir_type": None, "srr": 100, "h_max": 1, "h_delay": 0}
#             return x, info

#         # decide the RIR type
#         with self.lock:
#             rir_idx = self.rng.choice(len(self.weights), p=self.weights)

#         # add reverb
#         x, info = self.augmenters[rir_idx](x)
#         return x, info

#     def __call__(
#         self, x: np.ndarray
#     ) -> Tuple[np.ndarray, Dict[str, Union[Optional[str], float, int]]]:
#         """Runs reverberation augmentation using callable-style syntax.

#         Args:
#           x: Clean speech signal.

#         Returns:
#           Reverberant signal.
#           Dictionary containing RIR type, signal-to-reverb ratio (dB), linear gain, and delay.
#         """
#         return self.forward(x)

#     def reseed(self, seed: Union[int, np.random.SeedSequence]) -> None:
#         """Reseeds this augmenter and all child augmenters."""
#         child_seeds = np.random.SeedSequence(seed).spawn(len(self.augmenters) + 1)
#         self.rng = np.random.default_rng(seed=child_seeds[0])
#         for i, augmenter in enumerate(self.augmenters):
#             augmenter.reseed(child_seeds[i + 1])
