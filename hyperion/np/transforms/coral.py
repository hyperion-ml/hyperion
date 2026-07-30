"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional

import h5py
import numpy as np
import scipy.linalg as la

from ..hyper_np_model import HyperNPModel


class CORAL(HyperNPModel):
    """Class to do CORAL.

    https://arxiv.org/abs/1612.01939

    Attributes:
      mu: mean shift between both domains.
      T_col: recoloring projection.
      T_white: whitening projection.
      update_mu: whether or not to update mu when training.
      update_T: wheter or not to update T_col and T_white when training.
      alpha_mu: weight of the in-domain data when computing in-domain mean.
      alpha_T: weight of the in-domain data when computing in-domain covariance.

    Example:
      ```python
      import numpy as np
      from hyperion.np.transforms import CORAL

      rng = np.random.default_rng(1234)
      x_in = rng.standard_normal((500, 256))    # in-domain embeddings
      x_out = rng.standard_normal((800, 256))   # out-domain embeddings

      trn = CORAL(update_mu=True, update_T=True, alpha_mu=1.0, alpha_T=0.7)
      trn.fit(x=x_in, x_out=x_out)

      x_out_adapted = trn.predict(x_out)
      print(x_out_adapted.shape)  # (800, 256)
      ```
    """

    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        T_col: Optional[np.ndarray] = None,
        T_white: Optional[np.ndarray] = None,
        update_mu: bool = True,
        update_T: bool = True,
        alpha_mu: float = 1,
        alpha_T: float = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mu = mu
        self.T_col = T_col
        self.T_white = T_white
        self.T = None
        self.update_mu = update_mu
        self.update_T = update_T
        self.alpha_mu = alpha_mu
        self.alpha_T = alpha_T

    def get_config(self) -> Dict[str, Any]:
        """Returns the model configuration dict."""
        config = {
            "update_mu": self.update_mu,
            "update_T": self.update_T,
            "alpha_mu": self.alpha_mu,
            "alpha_T": self.alpha_T,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _compute_T(self) -> None:
        if self.T_col is not None and self.T_white is not None:
            self.T = np.dot(self.T_white, self.T_col)

    @staticmethod
    def _weighted_mean_cov(
        data: np.ndarray, weights: Optional[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        if data.ndim != 2:
            raise ValueError(f"data must be a 2D array, got shape={data.shape}")
        if data.shape[0] == 0:
            raise ValueError("data must contain at least one sample")

        if weights is None:
            mu = np.mean(data, axis=0)
            delta = data - mu
            cov = np.dot(delta.T, delta) / data.shape[0]
            return mu, cov

        w = np.asarray(weights).reshape(-1)
        if w.shape[0] != data.shape[0]:
            raise ValueError(
                "weights must have same length as data samples, "
                f"got {w.shape[0]} and {data.shape[0]}"
            )
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        w_sum = np.sum(w)
        if w_sum <= 0:
            raise ValueError("sum of weights must be > 0")

        mu = np.sum(data * w[:, None], axis=0) / w_sum
        delta = data - mu
        cov = np.dot((delta * w[:, None]).T, delta) / w_sum
        return mu, cov

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
            self._compute_T()
        if self.mu is not None:
            x = x - self.mu

        if self.T is not None:
            x = np.dot(x, self.T)

        return x

    def fit(
        self,
        x: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        x_out: Optional[np.ndarray] = None,
        sample_weight_out: Optional[np.ndarray] = None,
    ) -> None:
        """Trains the model.

        Args:
          x:  in-domain data samples with shape (num_samples, x_dim).
          sample_weight: weight for each in-domain training sample.
          x_out:  out-domain data samples with shape (num_samples, x_dim).
          sample_weight_out: weight for each out-domain training sample.
        """
        # Invalidate cached transform so it is rebuilt from updated factors.
        self.T = None

        if x.ndim != 2:
            raise ValueError(f"x must be a 2D array, got shape={x.shape}")
        if x_out is not None:
            if x_out.ndim != 2:
                raise ValueError(f"x_out must be a 2D array, got shape={x_out.shape}")
            if x_out.shape[1] != x.shape[1]:
                raise ValueError(
                    "x and x_out must have the same feature dimension, "
                    f"got {x.shape[1]} and {x_out.shape[1]}"
                )

        if x_out is None:
            if sample_weight_out is not None:
                raise ValueError("sample_weight_out was provided but x_out is None.")
            if self.update_mu:
                raise ValueError(
                    "update_mu=True requires x_out to compute out-domain mean."
                )
            if self.update_T:
                if self.T_white is None:
                    raise ValueError(
                        "x_out is required when update_T=True and "
                        "T_white is not initialized."
                    )
                if not np.isclose(self.alpha_T, 1.0):
                    raise ValueError(
                        "alpha_T must be 1 when x_out is None and update_T=True "
                        "(otherwise S_out is required)."
                    )
        else:
            mu_out, S_out = self._weighted_mean_cov(x_out, sample_weight_out)
            if self.update_T:
                # zero-phase component analysis (ZCA)
                d, V = la.eigh(S_out)
                d = np.maximum(d, 1e-10)
                self.T_white = np.dot(V * (1 / np.sqrt(d)), V.T)

        mu_in, S_in = self._weighted_mean_cov(x, sample_weight)
        if self.update_T:
            if self.alpha_T < 1:
                S_in = self.alpha_T * S_in + (1 - self.alpha_T) * S_out
            # zero-phase component analysis (ZCA)
            d, V = la.eigh(S_in)
            d[d < 0] = 0
            self.T_col = np.dot(V * np.sqrt(d), V.T)

        if self.update_mu:
            self.mu = self.alpha_mu * (mu_out - mu_in)

        self._compute_T()

    @classmethod
    def load_params(cls, f: h5py.File, config: Dict[str, Any]) -> "CORAL":
        """Initializes the model from the configuration and loads the model
        parameters from file.

        Args:
          f: file handle.
          config: configuration dictionary.

        Returns:
          Model object.
        """
        param_list = ["mu", "T_col", "T_white"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(
            mu=params["mu"],
            T_col=params["T_col"],
            T_white=params["T_white"],
            **config,
        )

    def save_params(self, f: h5py.File) -> None:
        """Saves the model paramters into the file.

        Args:
          f: file handle.
        """
        params = {
            "mu": self.mu,
            "T_col": self.T_col,
            "T_white": self.T_white,
        }
        self._save_params_from_dict(f, params)
