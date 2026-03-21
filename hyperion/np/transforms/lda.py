"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional

import h5py
import numpy as np
import scipy.linalg as la
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import PathLike
from ..hyper_np_model import HyperNPModel
from .sb_sw import SbSw


class LDA(HyperNPModel):
    """Class to do linear discriminant analysis.

    Attributes:
      mu: data mean vector
      T: LDA projection.
      lda_dim: LDA dimension.
      update_mu: whether or not to update the mean when training.
      update_T: wheter or not to update T when training.

    Example:
      ```python
      import numpy as np
      from hyperion.np.transforms import LDA

      rng = np.random.default_rng(1234)
      x = rng.standard_normal((2000, 256))
      y = rng.integers(0, 100, size=(2000,))

      lda = LDA(lda_dim=150, update_mu=True, update_T=True)
      lda.fit(x, y)
      x_lda = lda.predict(x)
      print(x_lda.shape)  # (2000, 150)
      ```
    """

    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        lda_dim: Optional[int] = None,
        update_mu: bool = True,
        update_T: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes an LDA transform.

        Args:
          mu: Optional centering vector with shape (x_dim,).
          T: Optional projection matrix with shape (x_dim, lda_dim). If provided,
            `lda_dim` is inferred from `T.shape[1]`.
          lda_dim: Target dimensionality to keep when estimating `T` in `fit`.
          update_mu: If True, updates `mu` during `fit`.
          update_T: If True, updates `T` during `fit`.
          **kwargs: Additional arguments forwarded to :class:`HyperNPModel`.
        """
        super().__init__(**kwargs)
        self.mu = mu
        self.T = T
        if T is None:
            self.lda_dim = lda_dim
        else:
            self.lda_dim = T.shape[1]
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
                "LDA projection matrix T is None. Fit the model with update_T=True "
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
            sbsw = SbSw()
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
                "Sb and Sw must have the same shape, "
                f"got {Sb.shape} and {Sw.shape}"
            )

        try:
            d, V = la.eigh(Sb, Sw)
        except la.LinAlgError:
            alpha = 1e-2 * np.max(np.diag(Sw))
            d, V = la.eigh(Sb, alpha * np.eye(Sw.shape[0]) + Sw)
        V = np.fliplr(V)

        p = V[0, :] < 0
        V[:, p] *= -1

        if self.lda_dim is not None:
            if self.lda_dim <= 0:
                raise ValueError(f"lda_dim must be > 0, got {self.lda_dim}")
            if self.lda_dim > V.shape[1]:
                raise ValueError(
                    "lda_dim must be <= input feature dimension, "
                    f"got lda_dim={self.lda_dim} and x_dim={V.shape[1]}"
                )
            V = V[:, : self.lda_dim]

        self.T = V

    def get_config(self) -> Dict[str, Any]:
        """Returns the model configuration dict."""
        config = {
            "lda_dim": self.lda_dim,
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
    def load_params(cls, f: h5py.File, config: Dict[str, Any]) -> "LDA":
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
    def load_mat(cls, file_path: PathLike) -> "LDA":
        """Loads LDA parameters from a Kaldi-style HDF5 matrix file.

        Args:
          file_path: Path to the file containing datasets `mu` and `T`.

        Returns:
          Loaded :class:`LDA` instance.
        """
        with h5py.File(file_path, "r") as f:
            mu = np.asarray(f["mu"], dtype="float32")
            T = np.asarray(f["T"], dtype="float32")
            return cls(mu, T)

    def save_mat(self, file_path: PathLike) -> None:
        """Saves LDA parameters to a Kaldi-style HDF5 matrix file.

        Args:
          file_path: Output file path.
        """
        with h5py.File(file_path, "w") as f:
            f.create_dataset("mu", data=self.mu)
            f.create_dataset("T", data=self.T)

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters a kwargs dictionary to only LDA constructor arguments.

        Args:
          **kwargs: Arbitrary keyword arguments.

        Returns:
          Dictionary containing only supported LDA arguments.
        """
        valid_args = ("update_mu", "update_T", "name", "lda_dim")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Adds LDA command-line arguments to an argument parser.

        Args:
          parser: Destination argument parser.
          prefix: Optional argument namespace prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--update-mu",
            default=True,
            action=ActionYesNo,
            help=("updates centering parameter"),
        )
        parser.add_argument(
            "--update-T",
            default=True,
            action=ActionYesNo,
            help=("updates projection parameter"),
        )

        parser.add_argument(
            "--lda-dim", default=1, type=int, help=("output dimension of LDA")
        )

        parser.add_argument("--name", dest="name", default="lda")
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )
