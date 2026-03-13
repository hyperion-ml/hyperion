"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import yaml
from librosa.effects import time_stretch


class SpeedAugment:
    """Augments speech by applying random speed perturbation.

    Attributes:
      speed_prob: Probability of applying speed perturbation to an utterance.
      speed_ratios: Candidate speed ratios used for sampling.
      keep_length: If ``True``, pads or crops output to match input length.
      rng: Random number generator used for augmentation decisions.
    """

    def __init__(
        self,
        speed_prob: float,
        speed_ratios: Sequence[float] = (0.9, 1.1),
        keep_length: bool = False,
        random_seed: int = 112358,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Initializes a speed augmenter.

        Args:
          speed_prob: Probability of applying speed perturbation.
          speed_ratios: Candidate speed ratios for random sampling.
          keep_length: If ``True``, keeps the output duration equal to input duration.
          random_seed: Seed used when creating a new random generator.
          rng: Optional pre-created random generator.

        Returns:
          None.
        """
        logging.info(
            "init speed augment with prob={}, speed_ratios={}, keep_length={}".format(
                speed_prob, speed_ratios, keep_length
            )
        )
        if not np.isscalar(speed_prob):
            raise TypeError(
                f"speed_prob must be a scalar value, got {type(speed_prob)}"
            )
        speed_prob = float(speed_prob)
        if not np.isfinite(speed_prob) or speed_prob < 0 or speed_prob > 1:
            raise ValueError(f"speed_prob must be in [0, 1], got {speed_prob}")
        self.speed_prob = speed_prob

        if isinstance(speed_ratios, (str, bytes)):
            raise TypeError("speed_ratios must be a sequence of positive values")
        try:
            speed_ratios = list(speed_ratios)
        except TypeError as err:
            raise TypeError(
                "speed_ratios must be a sequence of positive values"
            ) from err
        if len(speed_ratios) == 0:
            raise ValueError("speed_ratios must contain at least one ratio")

        self.speed_ratios = []
        for r in speed_ratios:
            if not np.isscalar(r):
                raise TypeError(
                    f"speed_ratios entries must be scalar values, got {type(r)}"
                )
            r = float(r)
            if not np.isfinite(r) or r <= 0:
                raise ValueError(
                    f"speed_ratios entries must be positive finite values, got {r}"
                )
            self.speed_ratios.append(r)

        if not isinstance(keep_length, (bool, np.bool_)):
            raise TypeError(
                f"keep_length must be a boolean value, got {type(keep_length)}"
            )
        self.keep_length = bool(keep_length)

        if rng is None:
            self.rng = np.random.default_rng(seed=random_seed)
        else:
            self.rng = deepcopy(rng)

    @classmethod
    def create(
        cls,
        cfg: Union[str, Dict[str, Any]],
        random_seed: int = 112358,
        rng: Optional[np.random.Generator] = None,
    ) -> "SpeedAugment":
        """Creates a SpeedAugment object from options dictionary or YAML file.

        Args:
          cfg: YAML file path or dictionary with speed perturb. options.
          random_seed: Seed used when creating a new random generator.
          rng: Optional pre-created random generator.

        Returns:
          Configured speed augmenter instance.
        """
        if isinstance(cfg, str):
            with open(cfg, "r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

        if not isinstance(cfg, dict):
            raise TypeError(f"wrong object type for cfg={cfg}")

        return cls(
            speed_prob=cfg["speed_prob"],
            speed_ratios=cfg["speed_ratios"],
            keep_length=cfg["keep_length"] if "keep_length" in cfg else False,
            random_seed=random_seed,
            rng=rng,
        )

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Union[int, float]]]:
        """Change the speed of the signal,
           the multiplication factor is chosen randomly.

        Args:
          x: Clean speech signal.

        Returns:
          Augmented signal.
          Dictionary containing speed ratio applied.
        """
        if x.ndim != 1:
            raise ValueError(
                f"SpeedAugment expects a 1-D waveform, got shape={x.shape}"
            )
        if x.shape[0] == 0:
            return x, {"speed_ratio": 1}

        # decide whether to add speed perturbation or not
        p = self.rng.random()
        if p > self.speed_prob:
            # we don't add speed perturbation
            info = {"speed_ratio": 1}
            return x, info

        speed_idx = self.rng.choice(len(self.speed_ratios))
        # change speed
        r = self.speed_ratios[speed_idx]
        info = {"speed_ratio": r}
        y = time_stretch(x, rate=r)
        # print(f"1 r={r} {x.shape} {y.shape}", flush=True)
        if self.keep_length:
            if r > 1:
                pad_len = max(0, x.shape[-1] - y.shape[-1])
                if pad_len > 0:
                    noise_std = max(np.max(np.abs(x)), 1e-8) / (2**15)
                    pad_y = noise_std * self.rng.standard_normal(pad_len, dtype=y.dtype)
                    y = np.concatenate((y, pad_y), axis=-1)
                y = y[: x.shape[-1]]
            elif r < 1:
                y = y[: x.shape[-1]]

        return y, info

    def __call__(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Union[int, float]]]:
        """Runs speed augmentation using callable-style syntax.

        Args:
          x: Clean speech signal.

        Returns:
          Augmented signal.
          Dictionary containing speed ratio applied.
        """
        return self.forward(x)

    def reseed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Reseeds the internal RNG."""
        self.rng = np.random.default_rng(seed=seed)
        self.rng = np.random.default_rng(seed=seed)
