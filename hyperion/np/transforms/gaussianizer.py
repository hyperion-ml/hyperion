"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Optional

import h5py
import numpy as np
from jsonargparse import ActionParser, ArgumentParser
from scipy.special import erfinv

from ...utils.misc import PathLike
from ..hyper_np_model import HyperNPModel


class Gaussianizer(HyperNPModel):
    """Class to make i/x-vector distribution standard Normal.

    Args:
      max_vectors: maximum number of background vectors needed to
        compute the Gaussianization.
      r: background vector matrix obtained by fit function.

    Example:
      ```python
      import numpy as np
      from hyperion.np.transforms import Gaussianizer

      rng = np.random.default_rng(1234)
      x_train = rng.standard_normal((10000, 256))
      x_eval = rng.standard_normal((500, 256))

      gauss = Gaussianizer(max_vectors=4000)
      gauss.fit(x_train)
      x_eval_g = gauss.predict(x_eval)
      print(x_eval_g.shape)  # (500, 256)
      ```
    """

    def __init__(
        self,
        max_vectors: Optional[int] = None,
        r: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_vectors = max_vectors
        self.r = r

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
        if self.r is None:
            raise ValueError(
                "Gaussianizer is not initialized: call fit() first or provide r."
            )
        if x.ndim != 2:
            raise ValueError(f"x must be a 2D array, got shape={x.shape}")
        if self.r.ndim != 2:
            raise ValueError(f"self.r must be a 2D array, got shape={self.r.shape}")
        if x.shape[1] != self.r.shape[1]:
            raise ValueError(
                "x and r must have the same feature dimension, "
                f"got {x.shape[1]} and {self.r.shape[1]}"
            )

        # px_cum = np.linspace(0, 1, self.r.shape[0] + 2)[1:-1]
        px_cum = np.linspace(0, 1, self.r.shape[0] + 3)[1:-1]
        y_map = erfinv(2 * px_cum - 1) * np.sqrt(2)

        # r = self.r[1:]
        r = self.r
        y = np.zeros_like(x)
        for i in range(x.shape[1]):
            y_index = np.searchsorted(r[:, i], x[:, i])
            logging.debug(y_index)
            y[:, i] = y_map[y_index]

        return y

    def fit(self, x: np.ndarray) -> None:
        """Trains the model.

        Args:
          x: training data samples with shape (num_samples, x_dim).
        """
        if x.ndim != 2:
            raise ValueError(f"x must be a 2D array, got shape={x.shape}")
        if x.shape[0] == 0:
            raise ValueError("x must contain at least one sample")
        if x.shape[1] == 0:
            raise ValueError("x must contain at least one feature")

        if self.max_vectors is not None:
            if self.max_vectors <= 0:
                raise ValueError(
                    f"max_vectors must be > 0 when provided, got {self.max_vectors}"
                )
            if x.shape[0] < self.max_vectors:
                max_vectors = x.shape[0]
            else:
                max_vectors = self.max_vectors
        else:
            max_vectors = x.shape[0]

        r = np.sort(x, axis=0, kind="heapsort")

        if r.shape[0] > max_vectors:
            index = np.round(
                np.linspace(0, r.shape[0] - 1, max_vectors, dtype=float)
            ).astype(int)
            r = r[index, :]

        self.r = r

    def get_config(self) -> Dict[str, Any]:
        """Returns the model configuration dict."""
        config = {"max_vectors": self.max_vectors}

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f: h5py.File) -> None:
        """Saves the model paramters into the file.

        Args:
          f: file handle.
        """
        params = {"r": self.r}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f: h5py.File, config: Dict[str, Any]) -> "Gaussianizer":
        """Initializes the model from the configuration and loads the model
        parameters from file.

        Args:
          f: file handle.
          config: configuration dictionary.

        Returns:
          Model object.
        """
        param_list = ["r"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(
            r=params["r"], max_vectors=config["max_vectors"], name=config["name"]
        )

    @classmethod
    def load_mat(cls, file_path: PathLike) -> "Gaussianizer":
        with h5py.File(file_path, "r") as f:
            r = np.asarray(f["r"], dtype="float32")
            return cls(r=r)

    def save_mat(self, file_path: PathLike) -> None:
        with h5py.File(file_path, "w") as f:
            f.create_dataset("r", data=self.r)

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        valid_args = ("max_vectors", "name")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--max-vectors",
            default=None,
            type=int,
            help=("maximum number of background vectors"),
        )

        parser.add_argument("--name", default="gauss")
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    add_argparse_args = add_class_args
