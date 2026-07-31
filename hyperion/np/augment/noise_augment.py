"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import yaml

from ...io import RandomAccessAudioReader as AR


class SingleNoiseAugment:
    """Augments speech with additive noise from a single noise category.

    Attributes:
      noise_type: Noise category label (for example, music or babble).
      r: Random-access audio reader for the configured noise source.
      noise_keys: Keys available in the noise reader.
      min_snr: Minimum sampled SNR value in dB.
      max_snr: Maximum sampled SNR value in dB.
      cache: Buffered tail from the last sampled noise recording.
      rng: Random number generator used for sampling noise and SNR.
    """

    def __init__(
        self,
        noise_type: str,
        noise_path: str,
        min_snr: float,
        max_snr: float,
        random_seed: int = 112358,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Initializes a single-noise augmenter.

        Args:
          noise_type: Noise category label.
          noise_path: Kaldi-style ``wav.scp`` path for noise recordings.
          min_snr: Minimum target SNR (dB) used during mixing.
          max_snr: Maximum target SNR (dB) used during mixing.
          random_seed: Seed used when creating a new random generator.
          rng: Optional pre-created random generator.

        Returns:
          None.
        """
        logging.info(
            "init noise_augment with noise={} noise_path={} snr={}-{}".format(
                noise_type, noise_path, min_snr, max_snr
            )
        )

        if not np.isscalar(min_snr):
            raise TypeError(f"min_snr must be a scalar value, got {type(min_snr)}")
        if not np.isscalar(max_snr):
            raise TypeError(f"max_snr must be a scalar value, got {type(max_snr)}")
        min_snr = float(min_snr)
        max_snr = float(max_snr)
        if not np.isfinite(min_snr) or not np.isfinite(max_snr):
            raise ValueError(
                f"min_snr and max_snr must be finite values, got ({min_snr}, {max_snr})"
            )
        if min_snr > max_snr:
            raise ValueError(f"min_snr must be <= max_snr, got ({min_snr}, {max_snr})")

        self.noise_type = noise_type
        self.r = AR(recordings=noise_path)
        self.noise_keys = self.r.keys
        if len(self.noise_keys) == 0:
            raise ValueError(
                f"noise_path={noise_path} contains no recordings for noise_type={noise_type}"
            )
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.cache = None
        if rng is None:
            self.rng = np.random.default_rng(seed=random_seed)
        else:
            self.rng = deepcopy(rng)

        logging.info("init noise_augment with noise={} done".format(noise_type))

    @staticmethod
    def _power(x: np.ndarray) -> float:
        """Computes signal power in dB.

        Args:
          x: Input waveform.

        Returns:
          Signal power in dB.
        """
        return 10 * np.log10((x**2).sum() + 1e-10)

    @staticmethod
    def snr(x: np.ndarray, n: np.ndarray) -> float:
        """Computes SNR in dB.

        Args:
          x: Clean speech signal.
          n: Noise signal.

        Returns:
          Signal-to-noise ratio in dB.
        """
        return SingleNoiseAugment._power(x) - SingleNoiseAugment._power(n)

    @staticmethod
    def _compute_noise_scale(x: np.ndarray, n: np.ndarray, target_snr: float) -> float:
        """Computes the linear scale factor to reach the target SNR.

        Args:
          x: Clean speech signal.
          n: Noise signal.
          target_snr: Desired SNR in dB.

        Returns:
          Linear scaling factor applied to the noise signal.
        """
        snr = SingleNoiseAugment.snr(x, n)
        return 10 ** ((snr - target_snr) / 20)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Union[str, float]]]:
        """Adds noise to signal, SNR is chosen randomly.

        Args:
          x: Clean speech signal.

        Returns:
          Noisy signal.
          Dictionary containing information of noise type and SNR(dB).
        """
        if x.ndim != 1:
            raise ValueError(
                f"SingleNoiseAugment expects a 1-D waveform, got shape={x.shape}"
            )
        num_samples = x.shape[0]
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
            noise_idx = self.rng.integers(len(self.noise_keys))
            key = self.noise_keys[noise_idx]
            noise_k, _ = self.r.read([key])
            noise_k = noise_k[0]

            if noise is None:
                need_samples = min(x.shape[0], noise_k.shape[0])
                noise = noise_k[:need_samples]
            else:
                need_samples = min(x.shape[0] - noise.shape[0], noise_k.shape[0])
                noise = np.concatenate((noise, noise_k[:need_samples]))

            if need_samples < noise_k.shape[0]:
                self.cache = noise_k[need_samples:]

        num_zeros = np.sum(noise == 0)
        # add dither for noises files with many 0s.
        if num_zeros > len(noise) // 3:
            noise += 0.0001 * self.rng.standard_normal(noise.shape, dtype=noise.dtype)

        target_snr = self.rng.uniform(self.min_snr, self.max_snr)

        scale = self._compute_noise_scale(x, noise, target_snr)

        info = {"noise_type": self.noise_type, "snr": target_snr}
        y = x + scale * noise
        # avg proc time 0.0091 secs
        return y, info

    def __call__(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Union[str, float]]]:
        """Runs additive noise augmentation using callable-style syntax.

        Args:
          x: Clean speech signal.

        Returns:
          Noisy signal.
          Dictionary containing information of noise type and SNR(dB).
        """
        return self.forward(x)

    def reseed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Reseeds the internal RNG and clears cached noise tail."""
        self.rng = np.random.default_rng(seed=seed)
        self.cache = None


class NoiseAugment:
    """Augments speech with additive noise sampled from multiple noise categories.

    Attributes:
      noise_prob: Probability of applying additive noise to an input utterance.
      weights: Sampling weights associated with ``augmenters``.
      augmenters: Per-noise-type augmenters.
      rng: Random number generator used for augmentation decisions.
    """

    def __init__(
        self,
        noise_prob: float,
        noise_types: Dict[str, Dict[str, Any]],
        random_seed: int = 112358,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Initializes a multi-noise augmenter.

        Args:
          noise_prob: Probability of applying additive noise.
          noise_types: Dictionary from noise type name to its configuration dictionary.
          random_seed: Seed used when creating a new random generator.
          rng: Optional pre-created random generator.

        Returns:
          None.
        """
        logging.info("init noise augment")
        if not np.isscalar(noise_prob):
            raise TypeError(
                f"noise_prob must be a scalar value, got {type(noise_prob)}"
            )
        noise_prob = float(noise_prob)
        if not np.isfinite(noise_prob) or noise_prob < 0 or noise_prob > 1:
            raise ValueError(f"noise_prob must be in [0, 1], got {noise_prob}")

        if not isinstance(noise_types, dict):
            raise TypeError(
                f"noise_types must be a dict from noise type to options, got {type(noise_types)}"
            )
        if len(noise_types) == 0:
            raise ValueError("noise_types must contain at least one noise type")

        self.noise_prob = noise_prob
        if rng is None:
            root_seed = np.random.SeedSequence(random_seed)
        else:
            seed_rng = deepcopy(rng)
            entropy = seed_rng.integers(
                0, np.iinfo(np.uint32).max, size=4, dtype=np.uint32
            )
            root_seed = np.random.SeedSequence(entropy.tolist())
        child_seeds = root_seed.spawn(len(noise_types) + 1)
        self.rng = np.random.default_rng(seed=child_seeds[0])

        augmenters = []
        self.weights = np.zeros((len(noise_types),))
        count = 0
        for key, opts in noise_types.items():
            if not isinstance(opts, dict):
                raise TypeError(
                    f"noise_types['{key}'] must be a dict, got {type(opts)}"
                )
            required_keys = ("weight", "noise_path", "min_snr", "max_snr")
            for cfg_key in required_keys:
                if cfg_key not in opts:
                    raise KeyError(
                        f"noise_types['{key}'] is missing required key '{cfg_key}'"
                    )

            weight = opts["weight"]
            if not np.isscalar(weight):
                raise TypeError(
                    f"noise_types['{key}']['weight'] must be a scalar value, got {type(weight)}"
                )
            weight = float(weight)
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(
                    f"noise_types['{key}']['weight'] must be a non-negative finite value, got {weight}"
                )
            self.weights[count] = weight
            aug = SingleNoiseAugment(
                key,
                opts["noise_path"],
                opts["min_snr"],
                opts["max_snr"],
                rng=np.random.default_rng(seed=child_seeds[count + 1]),
            )
            augmenters.append(aug)
            count += 1

        weights_sum = np.sum(self.weights)
        if weights_sum <= 0:
            raise ValueError("noise_types weights must sum to a value > 0")
        self.weights /= weights_sum
        self.augmenters = augmenters

    @classmethod
    def create(
        cls,
        cfg: Union[str, Dict[str, Any]],
        random_seed: int = 112358,
        rng: Optional[np.random.Generator] = None,
    ) -> "NoiseAugment":
        """Creates a NoiseAugment object from options dictionary or YAML file.

        Args:
          cfg: YAML file path or dictionary with noise options.
          random_seed: Seed used when creating a new random generator.
          rng: Optional pre-created random generator.

        Returns:
          Configured noise augmenter instance.
        """
        if isinstance(cfg, str):
            with open(cfg, "r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

        if not isinstance(cfg, dict):
            raise TypeError(f"wrong object type for cfg={cfg}")

        return cls(
            noise_prob=cfg["noise_prob"],
            noise_types=cfg["noise_types"],
            random_seed=random_seed,
            rng=rng,
        )

    def forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Union[Optional[str], float]]]:
        """Adds noise to signal, noise type and SNR are chosen randomly.

        Args:
          x: Clean speech signal.

        Returns:
          Noisy signal.
          Dictionary containing information of noise type and SNR(dB).
        """

        # decide whether to add noise or not
        p = self.rng.random()

        if p > self.noise_prob:
            # we don't add noise
            info = {"noise_type": None, "snr": 100}
            return x, info

        # decide the noise type
        noise_idx = self.rng.choice(len(self.weights), p=self.weights)

        # add noise
        x, info = self.augmenters[noise_idx](x)
        return x, info

    def __call__(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Union[Optional[str], float]]]:
        """Runs additive noise augmentation using callable-style syntax.

        Args:
          x: Clean speech signal.

        Returns:
          Noisy signal.
          Dictionary containing information of noise type and SNR(dB).
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


# class SingleNoiseAugment:
#     """Augments speech with additive noise from a single noise category.

#     Attributes:
#       noise_type: Noise category label (for example, music or babble).
#       r: Random-access audio reader for the configured noise source.
#       noise_keys: Keys available in the noise reader.
#       min_snr: Minimum sampled SNR value in dB.
#       max_snr: Maximum sampled SNR value in dB.
#       cache: Buffered tail from the last sampled noise recording.
#       lock: Mutex used to protect shared cache and RNG access.
#       rng: Random number generator used for sampling noise and SNR.
#     """

#     def __init__(
#         self,
#         noise_type: str,
#         noise_path: str,
#         min_snr: float,
#         max_snr: float,
#         random_seed: int = 112358,
#         rng: Optional[np.random.Generator] = None,
#     ) -> None:
#         """Initializes a single-noise augmenter.

#         Args:
#           noise_type: Noise category label.
#           noise_path: Kaldi-style ``wav.scp`` path for noise recordings.
#           min_snr: Minimum target SNR (dB) used during mixing.
#           max_snr: Maximum target SNR (dB) used during mixing.
#           random_seed: Seed used when creating a new random generator.
#           rng: Optional pre-created random generator.

#         Returns:
#           None.
#         """
#         logging.info(
#             "init noise_augment with noise={} noise_path={} snr={}-{}".format(
#                 noise_type, noise_path, min_snr, max_snr
#             )
#         )

#         if not np.isscalar(min_snr):
#             raise TypeError(f"min_snr must be a scalar value, got {type(min_snr)}")
#         if not np.isscalar(max_snr):
#             raise TypeError(f"max_snr must be a scalar value, got {type(max_snr)}")
#         min_snr = float(min_snr)
#         max_snr = float(max_snr)
#         if not np.isfinite(min_snr) or not np.isfinite(max_snr):
#             raise ValueError(
#                 f"min_snr and max_snr must be finite values, got ({min_snr}, {max_snr})"
#             )
#         if min_snr > max_snr:
#             raise ValueError(f"min_snr must be <= max_snr, got ({min_snr}, {max_snr})")

#         self.noise_type = noise_type
#         self.r = AR(recordings=noise_path)
#         self.noise_keys = self.r.keys
#         if len(self.noise_keys) == 0:
#             raise ValueError(
#                 f"noise_path={noise_path} contains no recordings for noise_type={noise_type}"
#             )
#         self.min_snr = min_snr
#         self.max_snr = max_snr
#         self.cache = None
#         self.lock = multiprocessing.Lock()
#         if rng is None:
#             self.rng = np.random.default_rng(seed=random_seed)
#         else:
#             self.rng = deepcopy(rng)

#         logging.info("init noise_augment with noise={} done".format(noise_type))

#     @staticmethod
#     def _power(x: np.ndarray) -> float:
#         """Computes signal power in dB.

#         Args:
#           x: Input waveform.

#         Returns:
#           Signal power in dB.
#         """
#         return 10 * np.log10((x**2).sum() + 1e-10)

#     @staticmethod
#     def snr(x: np.ndarray, n: np.ndarray) -> float:
#         """Computes SNR in dB.

#         Args:
#           x: Clean speech signal.
#           n: Noise signal.

#         Returns:
#           Signal-to-noise ratio in dB.
#         """
#         return SingleNoiseAugment._power(x) - SingleNoiseAugment._power(n)

#     @staticmethod
#     def _compute_noise_scale(x: np.ndarray, n: np.ndarray, target_snr: float) -> float:
#         """Computes the linear scale factor to reach the target SNR.

#         Args:
#           x: Clean speech signal.
#           n: Noise signal.
#           target_snr: Desired SNR in dB.

#         Returns:
#           Linear scaling factor applied to the noise signal.
#         """
#         snr = SingleNoiseAugment.snr(x, n)
#         return 10 ** ((snr - target_snr) / 20)

#     def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Union[str, float]]]:
#         """Adds noise to signal, SNR is chosen randomly.

#         Args:
#           x: Clean speech signal.

#         Returns:
#           Noisy signal.
#           Dictionary containing information of noise type and SNR(dB).
#         """
#         if x.ndim != 1:
#             raise ValueError(
#                 f"SingleNoiseAugment expects a 1-D waveform, got shape={x.shape}"
#             )
#         num_samples = x.shape[0]
#         with self.lock:
#             if self.cache is not None:
#                 if self.cache.shape[0] > num_samples:
#                     noise = self.cache[:num_samples]
#                     self.cache = self.cache[num_samples:]
#                 else:
#                     noise = self.cache
#                     self.cache = None
#             else:
#                 noise = None

#         while noise is None or noise.shape[0] < num_samples:
#             with self.lock:
#                 noise_idx = self.rng.integers(len(self.noise_keys))
#                 key = self.noise_keys[noise_idx]
#                 noise_k, _ = self.r.read([key])
#                 noise_k = noise_k[0]

#             if noise is None:
#                 need_samples = min(x.shape[0], noise_k.shape[0])
#                 noise = noise_k[:need_samples]
#             else:
#                 need_samples = min(x.shape[0] - noise.shape[0], noise_k.shape[0])
#                 noise = np.concatenate((noise, noise_k[:need_samples]))

#             if need_samples < noise_k.shape[0]:
#                 with self.lock:
#                     self.cache = noise_k[need_samples:]

#         num_zeros = np.sum(noise == 0)
#         with self.lock:
#             # add dither for noises files with many 0s.
#             if num_zeros > len(noise) // 3:
#                 noise += 0.0001 * self.rng.standard_normal(
#                     noise.shape, dtype=noise.dtype
#                 )

#             target_snr = self.rng.uniform(self.min_snr, self.max_snr)

#         scale = self._compute_noise_scale(x, noise, target_snr)

#         info = {"noise_type": self.noise_type, "snr": target_snr}
#         y = x + scale * noise
#         # avg proc time 0.0091 secs
#         return y, info

#     def __call__(
#         self, x: np.ndarray
#     ) -> Tuple[np.ndarray, Dict[str, Union[str, float]]]:
#         """Runs additive noise augmentation using callable-style syntax.

#         Args:
#           x: Clean speech signal.

#         Returns:
#           Noisy signal.
#           Dictionary containing information of noise type and SNR(dB).
#         """
#         return self.forward(x)


# class NoiseAugment:
#     """Augments speech with additive noise sampled from multiple noise categories.

#     Attributes:
#       noise_prob: Probability of applying additive noise to an input utterance.
#       weights: Sampling weights associated with ``augmenters``.
#       augmenters: Per-noise-type augmenters.
#       lock: Mutex used to protect RNG access.
#       rng: Random number generator used for augmentation decisions.
#     """

#     def __init__(
#         self,
#         noise_prob: float,
#         noise_types: Dict[str, Dict[str, Any]],
#         random_seed: int = 112358,
#         rng: Optional[np.random.Generator] = None,
#     ) -> None:
#         """Initializes a multi-noise augmenter.

#         Args:
#           noise_prob: Probability of applying additive noise.
#           noise_types: Dictionary from noise type name to its configuration dictionary.
#           random_seed: Seed used when creating a new random generator.
#           rng: Optional pre-created random generator.

#         Returns:
#           None.
#         """
#         logging.info("init noise augment")
#         if not np.isscalar(noise_prob):
#             raise TypeError(
#                 f"noise_prob must be a scalar value, got {type(noise_prob)}"
#             )
#         noise_prob = float(noise_prob)
#         if not np.isfinite(noise_prob) or noise_prob < 0 or noise_prob > 1:
#             raise ValueError(f"noise_prob must be in [0, 1], got {noise_prob}")

#         if not isinstance(noise_types, dict):
#             raise TypeError(
#                 f"noise_types must be a dict from noise type to options, got {type(noise_types)}"
#             )
#         if len(noise_types) == 0:
#             raise ValueError("noise_types must contain at least one noise type")

#         self.noise_prob = noise_prob

#         augmenters = []
#         self.weights = np.zeros((len(noise_types),))
#         count = 0
#         for key, opts in noise_types.items():
#             if not isinstance(opts, dict):
#                 raise TypeError(
#                     f"noise_types['{key}'] must be a dict, got {type(opts)}"
#                 )
#             required_keys = ("weight", "noise_path", "min_snr", "max_snr")
#             for cfg_key in required_keys:
#                 if cfg_key not in opts:
#                     raise KeyError(
#                         f"noise_types['{key}'] is missing required key '{cfg_key}'"
#                     )

#             weight = opts["weight"]
#             if not np.isscalar(weight):
#                 raise TypeError(
#                     f"noise_types['{key}']['weight'] must be a scalar value, got {type(weight)}"
#                 )
#             weight = float(weight)
#             if not np.isfinite(weight) or weight < 0:
#                 raise ValueError(
#                     f"noise_types['{key}']['weight'] must be a non-negative finite value, got {weight}"
#                 )
#             self.weights[count] = weight
#             aug = SingleNoiseAugment(
#                 key,
#                 opts["noise_path"],
#                 opts["min_snr"],
#                 opts["max_snr"],
#                 random_seed=random_seed,
#                 rng=rng,
#             )
#             augmenters.append(aug)
#             count += 1

#         weights_sum = np.sum(self.weights)
#         if weights_sum <= 0:
#             raise ValueError("noise_types weights must sum to a value > 0")
#         self.weights /= weights_sum
#         self.augmenters = augmenters

#         self.lock = multiprocessing.Lock()
#         if rng is None:
#             self.rng = np.random.default_rng(seed=random_seed)
#         else:
#             self.rng = deepcopy(rng)

#     @classmethod
#     def create(
#         cls,
#         cfg: Union[str, Dict[str, Any]],
#         random_seed: int = 112358,
#         rng: Optional[np.random.Generator] = None,
#     ) -> "NoiseAugment":
#         """Creates a NoiseAugment object from options dictionary or YAML file.

#         Args:
#           cfg: YAML file path or dictionary with noise options.
#           random_seed: Seed used when creating a new random generator.
#           rng: Optional pre-created random generator.

#         Returns:
#           Configured noise augmenter instance.
#         """
#         if isinstance(cfg, str):
#             with open(cfg, "r") as f:
#                 cfg = yaml.load(f, Loader=yaml.FullLoader)

#         if not isinstance(cfg, dict):
#             raise TypeError(f"wrong object type for cfg={cfg}")

#         return cls(
#             noise_prob=cfg["noise_prob"],
#             noise_types=cfg["noise_types"],
#             random_seed=random_seed,
#             rng=rng,
#         )

#     def forward(
#         self, x: np.ndarray
#     ) -> Tuple[np.ndarray, Dict[str, Union[Optional[str], float]]]:
#         """Adds noise to signal, noise type and SNR are chosen randomly.

#         Args:
#           x: Clean speech signal.

#         Returns:
#           Noisy signal.
#           Dictionary containing information of noise type and SNR(dB).
#         """

#         # decide whether to add noise or not
#         with self.lock:
#             p = self.rng.random()

#         if p > self.noise_prob:
#             # we don't add noise
#             info = {"noise_type": None, "snr": 100}
#             return x, info

#         # decide the noise type
#         with self.lock:
#             noise_idx = self.rng.choice(len(self.weights), p=self.weights)

#         # add noise
#         x, info = self.augmenters[noise_idx](x)
#         return x, info

#     def __call__(
#         self, x: np.ndarray
#     ) -> Tuple[np.ndarray, Dict[str, Union[Optional[str], float]]]:
#         """Runs additive noise augmentation using callable-style syntax.

#         Args:
#           x: Clean speech signal.

#         Returns:
#           Noisy signal.
#           Dictionary containing information of noise type and SNR(dB).
#         """
#         return self.forward(x)
