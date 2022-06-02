"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

from ...np_model import NPModel


class PDF(NPModel):
    """Base class for probability density functions.

    Attributes:
      x_dim: data dimension.
    """

    def __init__(self, x_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.x_dim = x_dim

    def get_config(self):
        """Returns the model configuration dict."""
        config = {"x_dim": self.x_dim}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def log_prob(self, x):
        """Computes log probability of the data."""
        raise NotImplementedError()

    def eval_llk(self, x):
        """Computes log likelihood of the data."""
        return self.log_prob(x)

    def sample(self, num_samples):
        """Draws samples from the data distribution."""
        raise NotImplementedError()

    def generate(self, num_samples, **kwargs):
        """Draws samples from the data distribution.
        Args:
          num_samples: number of samples to generate.

        Returns:
          np.array of generated samples with shape=(num_samples, x_dim)
        """
        return self.sample(num_samples, **kwargs)
