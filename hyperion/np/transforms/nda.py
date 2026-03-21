"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional

import h5py
import numpy as np
import scipy.linalg as la

from ...hyp_defs import float_cpu
from ...utils.misc import PathLike
from ..hyper_np_model import HyperNPModel
from .sb_sw import NSbSw


class NDA(HyperNPModel):
    """Class to do nearest-neighbors discriminant analysis

    Attributes:
      mu: data mean vector
      T: NDA projection.

    Example:
      ```python
      import numpy as np
      from hyperion.np.transforms import NDA

      rng = np.random.default_rng(1234)
      x = rng.standard_normal((2500, 256))
      y = rng.integers(0, 120, size=(2500,))

      nda = NDA(nda_dim=150, update_mu=True, update_T=True)
      nda.fit(x, y)
      x_nda = nda.predict(x)
      print(x_nda.shape)  # (2500, 150)
      ```
    """

    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        nda_dim: Optional[int] = None,
        update_mu: bool = True,
        update_T: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes an NDA transform.

        Args:
          mu: Optional centering vector with shape (x_dim,).
          T: Optional projection matrix with shape (x_dim, nda_dim). If provided,
            `nda_dim` is inferred from `T.shape[1]`.
          nda_dim: Target dimensionality to keep when estimating `T` in `fit`.
          update_mu: If True, updates `mu` during `fit`.
          update_T: If True, updates `T` during `fit`.
          **kwargs: Additional arguments forwarded to :class:`HyperNPModel`.
        """
        super().__init__(**kwargs)
        self.mu = mu
        self.T = T
        if T is None:
            self.nda_dim = nda_dim
        else:
            self.nda_dim = T.shape[1]
        self.update_mu = update_mu
        self.update_T = update_T

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
        if self.T is None:
            raise ValueError(
                "NDA projection matrix T is None. Fit the model with update_T=True "
                "or load precomputed parameters before calling predict."
            )
        if x.ndim != 2:
            raise ValueError(f"x must be a 2D array, got shape={x.shape}")
        if x.shape[1] != self.T.shape[0]:
            raise ValueError(
                "x feature dimension must match T input dimension, "
                f"got {x.shape[1]} and {self.T.shape[0]}"
            )
        if self.mu is not None and self.mu.shape[0] != x.shape[1]:
            raise ValueError(
                "mu dimension must match x feature dimension, "
                f"got {self.mu.shape[0]} and {x.shape[1]}"
            )
        if self.mu is not None:
            x = x - self.mu
        return np.dot(x, self.T)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mu: Optional[np.ndarray] = None,
        Sb: Optional[np.ndarray] = None,
        Sw: Optional[np.ndarray] = None,
    ) -> None:
        """Trains the model.

        Args:
          x: training data samples with shape (num_samples, x_dim).
          y: training labels as integers in [0, num_classes-1] with shape (num_samples,)
          mu: precomputed mean.
          Sb: precomputed between-class covariance.
          Sw: precomputed within-class covariance.
        """
        if mu is None or Sb is None or Sw is None:
            sbsw = NSbSw()
            sbsw.fit(x, y)
            mu = sbsw.mu
            Sb = sbsw.Sb
            Sw = sbsw.Sw

        if self.update_mu:
            self.mu = mu

        if not self.update_T:
            return

        if Sb.shape != Sw.shape:
            raise ValueError(
                "Sb and Sw must have the same shape, " f"got {Sb.shape} and {Sw.shape}"
            )

        try:
            d, V = la.eigh(Sb, Sw)
        except la.LinAlgError:
            alpha = 1e-2 * np.max(np.diag(Sw))
            d, V = la.eigh(Sb, alpha * np.eye(Sw.shape[0]) + Sw)
        V = np.fliplr(V)

        p = V[0, :] < 0
        V[:, p] *= -1

        if self.nda_dim is not None:
            if self.nda_dim <= 0:
                raise ValueError(f"nda_dim must be > 0, got {self.nda_dim}")
            if self.nda_dim > V.shape[1]:
                raise ValueError(
                    "nda_dim must be <= input feature dimension, "
                    f"got nda_dim={self.nda_dim} and x_dim={V.shape[1]}"
                )
            V = V[:, : self.nda_dim]

        self.T = V

    def get_config(self) -> Dict[str, Any]:
        """Returns the model configuration dict."""
        config = {
            "nda_dim": self.nda_dim,
            "update_mu": self.update_mu,
            "update_T": self.update_T,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f: h5py.File) -> None:
        """Saves the model paramters into the file.

        Args:
          f: file handle.
        """
        params = {"mu": self.mu, "T": self.T}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f: h5py.File, config: Dict[str, Any]) -> "NDA":
        """Initializes the model from the configuration and loads the model
        parameters from file.

        Args:
          f: file handle.
          config: configuration dictionary.

        Returns:
          Model object.
        """
        param_list = ["mu", "T"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(mu=params["mu"], T=params["T"], **config)

    @classmethod
    def load_mat(cls, file_path: PathLike) -> "NDA":
        """Loads NDA parameters from a Kaldi-style HDF5 matrix file.

        Args:
          file_path: Path to the file containing datasets `mu` and `T`.

        Returns:
          Loaded :class:`NDA` instance.
        """
        with h5py.File(file_path, "r") as f:
            mu = np.asarray(f["mu"], dtype="float32")
            T = np.asarray(f["T"], dtype="float32")
            return cls(mu, T)

    def save_mat(self, file_path: PathLike) -> None:
        """Saves NDA parameters to a Kaldi-style HDF5 matrix file.

        Args:
          file_path: Output file path.
        """
        with h5py.File(file_path, "w") as f:
            f.create_dataset("mu", data=self.mu)
            f.create_dataset("T", data=self.T)
