"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

from .cent_whiten import CentWhiten


class LNorm(CentWhiten):
    """Class to do length normalization.

    Attributes:
      mu: data mean vector
      T: whitening projection.
      update_mu: whether or not to update the mean when training.
      update_T: wheter or not to update T when training.

    Example:
      ```python
      import numpy as np
      from hyperion.np.transforms import LNorm

      rng = np.random.default_rng(1234)
      x = rng.standard_normal((1000, 256))

      lnorm = LNorm(update_mu=False, update_T=False)
      x_ln = lnorm.predict(x)
      print(x_ln.shape)  # (1000, 256)
      ```
    """

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = super().predict(x)
        mx = np.sqrt(np.sum(x**2, axis=1, keepdims=True)) + 1e-10
        return np.sqrt(x.shape[1]) * x / mx
