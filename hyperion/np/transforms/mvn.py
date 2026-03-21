"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional

import h5py
import numpy as np

from ..hyper_np_model import HyperNPModel


class MVN(HyperNPModel):
    """Class to do global mean and variance normalization.

    Attributes:
      mu: data mean vector
      s: standard deviation vector.

    Example:
      ```python
      import numpy as np
      from hyperion.np.transforms import MVN

      rng = np.random.default_rng(1234)
      x = rng.standard_normal((2000, 80))

      mvn = MVN()
      mvn.fit(x)
      x_mvn = mvn.predict(x)
      print(x_mvn.shape)  # (2000, 80)
      ```

    """

    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        s: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mu = mu
        self.s = s

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Applies the transformation to the data.

        Args:
          x: data samples.

        Returns:
          Transformed data samples.
        """
        return self.predict(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Applies the transformation to the data.

        Args:
          x: data samples.

        Returns:
          Transformed data samples.
        """
        return self.predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Applies the transformation to the data.

        Args:
          x: data samples.

        Returns:
          Transformed data samples.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be a 2D array, got shape={x.shape}")
        if self.mu is not None:
            if self.mu.ndim != 1:
                raise ValueError(f"mu must be a 1D array, got shape={self.mu.shape}")
            if self.mu.shape[0] != x.shape[1]:
                raise ValueError(
                    "mu dimension must match x feature dimension, "
                    f"got {self.mu.shape[0]} and {x.shape[1]}"
                )
        if self.s is not None:
            if self.s.ndim != 1:
                raise ValueError(f"s must be a 1D array, got shape={self.s.shape}")
            if self.s.shape[0] != x.shape[1]:
                raise ValueError(
                    "s dimension must match x feature dimension, "
                    f"got {self.s.shape[0]} and {x.shape[1]}"
                )

        if self.mu is not None:
            x = x - self.mu
        if self.s is not None:
            x = x / np.maximum(self.s, 1e-10)
        return x

    def fit(self, x: np.ndarray) -> None:
        """Trains the model.

        Args:
          x: training data samples with shape (num_samples, x_dim).
        """
        if x.ndim != 2:
            raise ValueError(f"x must be a 2D array, got shape={x.shape}")
        if x.shape[0] == 0:
            raise ValueError("x must have at least one sample")
        if x.shape[1] == 0:
            raise ValueError("x must have at least one feature")

        self.mu = np.mean(x, axis=0)
        self.s = np.maximum(np.std(x, axis=0), 1e-10)

    def save_params(self, f: h5py.File) -> None:
        """Saves the model parameters into the file.

        Args:
          f: file handle.
        """
        params = {"mu": self.mu, "s": self.s}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f: h5py.File, config: Dict[str, Any]) -> "MVN":
        """Initializes the model from the configuration and loads the model
        parameters from file.

        Args:
          f: file handle.
          config: configuration dictionary.

        Returns:
          Model object.
        """
        param_list = ["mu", "s"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(mu=params["mu"], s=params["s"], name=config["name"])
