"""Executable fixture-scale coverage for practical NumPy tutorials."""

from __future__ import annotations

import numpy as np

from hyperion.np.transforms import MVN, PCA


def test_transforms_quickstart() -> None:
    """Fit the normalization and PCA pipeline shown in the transforms tutorial."""
    rng = np.random.default_rng(1234)
    x = rng.standard_normal((300, 32))

    mvn = MVN()
    mvn.fit(x)
    x_mvn = mvn.predict(x)

    pca = PCA(pca_dim=16, whiten=True)
    pca.fit(x_mvn)
    x_pca = pca.predict(x_mvn)

    assert x_pca.shape == (300, 16)
