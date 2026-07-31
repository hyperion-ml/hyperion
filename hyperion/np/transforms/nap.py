"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional

import h5py
import numpy as np
import scipy.linalg as la

from ...utils.misc import PathLike
from ..hyper_np_model import HyperNPModel


class NAP(HyperNPModel):
    """Class to do nuisance attribute projection (NAP).

    NAP removes low-rank nuisance directions estimated from class-centered data:
    `x_hat = x - (x U^T) U`, where rows of `U` are nuisance basis vectors.

    Attributes:
      U: Nuisance subspace basis with shape `(U_dim, x_dim)`.
      U_dim: Number of nuisance directions to estimate/use.

    Example:
      ```python
      import numpy as np
      from hyperion.np.transforms import NAP

      rng = np.random.default_rng(1234)
      x = rng.standard_normal((3000, 256))
      class_ids = rng.integers(0, 100, size=(3000,))

      nap = NAP(U_dim=50)
      nap.fit(x, class_ids)
      x_nap = nap.predict(x)
      print(x_nap.shape)  # (3000, 256)
      ```
    """

    def __init__(
        self, U: Optional[np.ndarray] = None, U_dim: Optional[int] = None, **kwargs: Any
    ) -> None:
        """Initializes the NAP transform.

        Args:
          U: Optional nuisance basis matrix `(U_dim, x_dim)`.
          U_dim: Number of nuisance directions to estimate in `fit`.
          **kwargs: Additional arguments forwarded to :class:`HyperNPModel`.
        """
        super().__init__(**kwargs)
        self.U = U
        if U is None:
            self.U_dim = U_dim
        else:
            self.U_dim = U.shape[0]

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
        if self.U is None:
            raise ValueError(
                "NAP projection matrix U is None. Fit the model or load "
                "precomputed parameters before calling predict."
            )
        return x - np.dot(np.dot(x, self.U.T), self.U)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Trains the model.

        Args:
          x: training data samples with shape (num_samples, x_dim).
          y: class labels with shape (num_samples,).
        """
        u_ids = np.unique(y)
        M = np.sqrt(len(u_ids))
        xx = np.empty_like(x, dtype=np.float64)
        for i in u_ids:
            idx = y == i
            N = np.sqrt(np.sum(idx))
            mu_i = np.mean(x[idx, :], axis=0)
            xx[idx, :] = (x[idx, :] - mu_i) / N
        xx /= M
        _, s, Vt = la.svd(xx, full_matrices=False, overwrite_a=True)
        idx = (np.argsort(s)[::-1])[: self.U_dim]
        self.U = Vt[idx, :]

    def get_config(self) -> Dict[str, Any]:
        """Returns the model configuration dict."""
        config = {
            "U_dim": self.U_dim,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f: h5py.File) -> None:
        """Saves the model parameters into the file.

        Args:
          f: file handle.
        """
        params = {"U": self.U}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f: h5py.File, config: Dict[str, Any]) -> "NAP":
        """Initializes the model from the configuration and loads the model
        parameters from file.

        Args:
          f: file handle.
          config: configuration dictionary.

        Returns:
          Model object.
        """
        param_list = ["U"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(U=params["U"], name=config["name"])

    @classmethod
    def load_mat(cls, file_path: PathLike) -> "NAP":
        """Loads NAP parameters from an HDF5 matrix file.

        Args:
          file_path: Path to the file containing dataset `U`.

        Returns:
          Loaded :class:`NAP` instance.
        """
        with h5py.File(file_path, "r") as f:
            U = np.asarray(f["U"], dtype="float32")
            return cls(U)

    def save_mat(self, file_path: PathLike) -> None:
        """Saves NAP parameters to an HDF5 matrix file.

        Args:
          file_path: Output file path.
        """
        with h5py.File(file_path, "w") as f:
            f.create_dataset("U", data=self.U)
