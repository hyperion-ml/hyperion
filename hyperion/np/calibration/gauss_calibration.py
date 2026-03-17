"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Union

import numpy as np

from ..hyper_np_model import HyperNPModel

ScoreInput = Union[float, np.ndarray]


class GaussCalibration(HyperNPModel):
    """Class for supervised Gaussian calibration.
       The model assumes that targer and non-target score distributions are Gaussians
       with shared covariance.

    Attributes:
      mu1: mean of the target score distribution.
      mu2: mean of the non-target score distribution.
      sigma2: shared variance of the target and non-target score distributions.
      prior: prior prob. for target trials.
    """

    @staticmethod
    def _as_optional_float(value: Optional[float], name: str) -> Optional[float]:
        if value is None:
            return None
        value_arr = np.asarray(value)
        if value_arr.size != 1:
            raise ValueError(f"{name} must be a scalar, got shape={value_arr.shape}")
        return float(value_arr.reshape(()))

    def __init__(
        self,
        mu1: Optional[float] = None,
        mu2: Optional[float] = None,
        sigma2: Optional[float] = None,
        prior: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mu1: Optional[float] = self._as_optional_float(mu1, "mu1")
        self.mu2: Optional[float] = self._as_optional_float(mu2, "mu2")
        self.sigma2: Optional[float] = self._as_optional_float(sigma2, "sigma2")
        self.prior: float = prior
        self.a: Optional[float] = None
        self.b: Optional[float] = None
        if self.is_init():
            self._compute_scale_bias()

    def is_init(self) -> bool:
        """
        Returns:
          True if the model has been initialized.
        """
        return self.mu1 is not None and self.mu2 is not None and self.sigma2 is not None

    def _compute_scale_bias(self) -> None:
        """Computes the scaling and bias of the scores given the Gaussians means and variance."""
        assert self.mu1 is not None and self.mu2 is not None and self.sigma2 is not None

        self.a = (self.mu1 - self.mu2) / self.sigma2
        self.b = 0.5 * (self.mu2**2 - self.mu1**2) / self.sigma2

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Estimates the parameters of the model.

        Args:
          x: score numpy tensor (num_scores,).
          y: trial labels (0,1) numpy tensor (num_scores,).
          sample_weight: weight of each score in the calculation of the Gaussian parameters (num_scores,).
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim != 1:
            raise ValueError(f"x must be 1D, got shape={x.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape={y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of elements")
        if x.size == 0:
            raise ValueError("x and y must be non-empty")

        valid_label = np.logical_or(y == 0, y == 1)
        if not np.all(valid_label):
            raise ValueError("y must contain only binary labels {0, 1}")

        if sample_weight is None:
            sample_weight_eff = np.ones_like(x, dtype=float)
        else:
            sample_weight_eff = np.asarray(sample_weight, dtype=float)
            if sample_weight_eff.ndim != 1:
                raise ValueError(
                    f"sample_weight must be 1D, got shape={sample_weight_eff.shape}"
                )
            if sample_weight_eff.shape[0] != x.shape[0]:
                raise ValueError("sample_weight must have the same length as x")
            if np.any(sample_weight_eff < 0):
                raise ValueError("sample_weight cannot contain negative values")

        tar_mask = y == 1
        non_mask = y == 0
        if not np.any(tar_mask):
            raise ValueError("y must contain at least one target sample with label 1")
        if not np.any(non_mask):
            raise ValueError(
                "y must contain at least one non-target sample with label 0"
            )

        non = x[y == 0]
        tar = x[y == 1]
        sw_tar = sample_weight_eff[tar_mask]
        sw_non = sample_weight_eff[non_mask]
        sw_tar_sum = float(np.sum(sw_tar))
        sw_non_sum = float(np.sum(sw_non))
        sw_sum = float(np.sum(sample_weight_eff))
        if sw_tar_sum <= 0:
            raise ValueError("sum of target sample weights must be > 0")
        if sw_non_sum <= 0:
            raise ValueError("sum of non-target sample weights must be > 0")
        if sw_sum <= 0:
            raise ValueError("sum of sample weights must be > 0")

        self.prior = sw_tar_sum / sw_sum

        self.mu1 = float(np.sum(sw_tar * tar) / sw_tar_sum)
        self.mu2 = float(np.sum(sw_non * non) / sw_non_sum)

        sigma2_num = np.sum(sw_tar * (tar - self.mu1) ** 2) + np.sum(
            sw_non * (non - self.mu2) ** 2
        )
        self.sigma2 = float(sigma2_num / sw_sum)
        if not np.isfinite(self.sigma2) or self.sigma2 <= 0:
            raise ValueError(f"sigma2 must be finite and > 0, got {self.sigma2}")

        self._compute_scale_bias()

    def predict(self, x: ScoreInput) -> ScoreInput:
        """Applies the calibration function.

        Args:
          x: score vector (num_scores,)

        Returns:
          Vector with calibrated scores.
        """
        assert self.a is not None and self.b is not None
        return self.a * x + self.b

    def __call__(self, x: ScoreInput) -> ScoreInput:
        """Applies the calibration function.

        Args:
          x: score vector (num_scores,)

        Returns:
          Vector with calibrated scores.
        """
        return self.predict(x)

    def save_params(self, f: Any) -> None:
        params = {"mu1": self.mu1, "mu2": self.mu2, "sigma2": self.sigma2}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f: Any, config: Dict[str, Any]) -> "GaussCalibration":
        param_list = ["mu1", "mu2", "sigma2"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(
            mu1=cls._as_optional_float(params["mu1"], "mu1"),
            mu2=cls._as_optional_float(params["mu2"], "mu2"),
            sigma2=cls._as_optional_float(params["sigma2"], "sigma2"),
            name=config["name"],
        )
