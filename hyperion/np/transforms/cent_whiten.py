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


class CentWhiten(HyperNPModel):
    """Class to do centering and whitening of i-vectors.

    Attributes:
      mu: data mean vector
      T: whitening projection.
      update_mu: whether or not to update the mean when training.
      update_T: wheter or not to update T when training.
    """

    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        update_mu: bool = True,
        update_T: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mu = mu
        self.T = T
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
        if self.mu is not None:
            x = x - self.mu
        if self.T is not None:
            if self.T.ndim == 1:
                x = x * self.T
            else:
                x = np.dot(x, self.T)
        return x

    def fit(
        self,
        x: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        mu: Optional[np.ndarray] = None,
        S: Optional[np.ndarray] = None,
    ) -> None:
        """Trains the model.

        Args:
          x: training data samples with shape (num_samples, x_dim).
          sample_weight: weight for each training sample.
          mu: precomputed mean (used if x is None).
          S: precomputed convariances (used if x is None).
        """
        if x is not None:
            if x.shape[0] > x.shape[1]:
                # Lazy import to avoid transform<->pdf package import cycles.
                from ..pdfs.core import Normal

                gauss = Normal(x_dim=x.shape[1])
                gauss.fit(x=x, sample_weight=sample_weight)
                mu = gauss.mu
                S = gauss.Sigma
            else:
                mu = np.mean(x, axis=0)
                S = np.eye(x.shape[1])
        elif self.update_T and S is None:
            raise ValueError("S must be provided when x is None and update_T=True.")

        if self.update_mu:
            self.mu = mu

        if self.update_T:
            d, V = la.eigh(S)
            V *= np.sqrt(1 / d)
            V = np.fliplr(V)

            p = V[0, :] < 0
            V[:, p] *= -1

            nonzero = d > 0
            if not np.all(nonzero):
                V = V[:, nonzero[::-1]]

            self.T = V

    def get_config(self) -> Dict[str, Any]:
        """Returns the model configuration dict."""
        config = {"update_mu": self.update_mu, "update_T": self.update_T}
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
    def load_params(cls, f: h5py.File, config: Dict[str, Any]) -> "CentWhiten":
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
    def load_mat(cls, file_path: PathLike) -> "CentWhiten":
        with h5py.File(file_path, "r") as f:
            mu = np.asarray(f["mu"], dtype="float32")
            T = np.asarray(f["T"], dtype="float32")
            return cls(mu, T)

    def save_mat(self, file_path: PathLike) -> None:
        with h5py.File(file_path, "w") as f:
            f.create_dataset("mu", data=self.mu)
            f.create_dataset("T", data=self.T)

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        valid_args = ("update_mu", "update_T", "name")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
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
            help=("updates whitening parameter"),
        )

        parser.add_argument("--name", default="lnorm")
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    add_argparse_args = add_class_args
