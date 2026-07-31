"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

from .cent_whiten_up import CentWhitenUP


class LNormUP(CentWhitenUP):
    """Class to do Lenght Normalization with uncertainty propagation.

    Attributes:
      mu: data mean vector
      T: whitening projection.
      update_mu: whether or not to update the mean when training.
      update_T: wheter or not to update T when training.

    Example:
      ```python
      import numpy as np
      from hyperion.np.transforms import LNormUP

      rng = np.random.default_rng(1234)
      x_dim = 256
      m = rng.standard_normal((800, x_dim))
      s2 = np.abs(rng.standard_normal((800, x_dim)))
      x_up = np.hstack((m, s2))

      lnorm_up = LNormUP(update_mu=False, update_T=False, T=np.eye(x_dim))
      y_up = lnorm_up.predict(x_up)
      print(y_up.shape)  # (800, 512)
      ```
    """

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = super().predict(x)
        x_dim = int(x.shape[-1] / 2)
        m_x = x[:, :x_dim]
        s2_x = x[:, x_dim:]

        mx2 = np.sum(m_x**2, axis=1, keepdims=True) + 1e-10
        m_x /= np.sqrt(mx2)
        s2_x /= mx2

        return np.hstack((m_x, s2_x))
