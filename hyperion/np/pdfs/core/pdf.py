"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict

import numpy as np

from ...np_model import NPModel


class PDF(NPModel):
    """Base class for probability density functions.

    Attributes:
        x_dim: Dimensionality of the data vectors represented by the PDF.
    """

    def __init__(self, x_dim: int = 1, **kwargs: Any) -> None:
        """Initializes a PDF instance.

        Args:
            x_dim: Number of dimensions for each data point.
            **kwargs: Extra keyword arguments propagated to `NPModel`.
        """
        super().__init__(**kwargs)
        self.x_dim: int = x_dim

    def get_config(self) -> Dict[str, Any]:
        """Returns the model configuration dict.

        Returns:
            A dictionary with all configuration parameters, including `x_dim`.
        """
        config = {"x_dim": self.x_dim}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Computes the log probability of each data sample.

        Args:
            x: Array of shape `(num_samples, x_dim)` containing the samples.

        Returns:
            An array of shape `(num_samples,)` with log-probabilities.
        """
        raise NotImplementedError()

    def eval_llk(self, x: np.ndarray) -> np.ndarray:
        """Computes the log-likelihood of the data.

        Args:
            x: Array of samples used to evaluate the likelihood.

        Returns:
            An array with the log-likelihood values for each sample.
        """
        return self.log_prob(x)

    def sample(self, num_samples: int) -> np.ndarray:
        """Draws samples from the data distribution.

        Args:
            num_samples: Number of samples to draw from the PDF.

        Returns:
            A sample matrix of shape `(num_samples, x_dim)`.
        """
        raise NotImplementedError()

    def generate(self, num_samples: int, **kwargs: Any) -> np.ndarray:
        """Draws samples from the data distribution.

        Args:
            num_samples: Number of samples to generate.
            **kwargs: Additional arguments forwarded to `sample`.

        Returns:
            Array of generated samples with shape `(num_samples, x_dim)`.
        """
        return self.sample(num_samples, **kwargs)
