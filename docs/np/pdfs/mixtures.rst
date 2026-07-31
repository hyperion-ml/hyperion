Mixtures Tutorial (NumPy)
=========================

This tutorial shows practical usage of mixture PDF classes in
``hyperion.np.pdfs.mixtures``:

* :class:`hyperion.np.pdfs.mixtures.gmm.GMM`
* :class:`hyperion.np.pdfs.mixtures.gmm_diag_cov.GMMDiagCov`
* :class:`hyperion.np.pdfs.mixtures.gmm_tied_diag_cov.GMMTiedDiagCov`

See also:

* :doc:`../../numpy`
* :class:`hyperion.np.pdfs.core.normal.Normal`
* :class:`hyperion.np.pdfs.core.normal_diag_cov.NormalDiagCov`

Quick Start
-----------

Train a diagonal-covariance GMM and evaluate log-likelihoods:

.. code-block:: python

   import numpy as np
   from hyperion.np.pdfs.mixtures.gmm_diag_cov import GMMDiagCov

   rng = np.random.default_rng(1234)
   x = rng.standard_normal((2000, 20)).astype("float32")

   gmm = GMMDiagCov(num_comp=8, x_dim=20)
   _ = gmm.fit(x, epochs=3, batch_size=256)

   llk = gmm.log_prob(x[:5])
   print(llk.shape)  # (5,)


Example 1: Full-Covariance GMM
------------------------------

.. code-block:: python

   import numpy as np
   from hyperion.np.pdfs.mixtures.gmm import GMM

   rng = np.random.default_rng(1)
   x = rng.standard_normal((1500, 6)).astype("float32")

   gmm = GMM(num_comp=4, x_dim=6)
   _ = gmm.fit(x, epochs=2)

   post = gmm.compute_pz(x[:10])   # responsibilities, shape (10, 4)
   llk = gmm.log_prob(x[:10])      # shape (10,)


Example 2: Top-N Scoring (Master Mode)
--------------------------------------

Use the model itself to find top components per sample and return both
``log p(x)`` and selected indices.

.. code-block:: python

   import numpy as np
   from hyperion.np.pdfs.mixtures.gmm_diag_cov import GMMDiagCov

   rng = np.random.default_rng(2)
   x = rng.standard_normal((500, 10)).astype("float32")

   gmm = GMMDiagCov(num_comp=16, x_dim=10)
   _ = gmm.fit(x, epochs=2)

   llk, top_idx = gmm.log_prob_nbest(x[:20], mode="std", nbest_mode="master", nbest=3)
   print(llk.shape, top_idx.shape)  # (20,) (20, 3)


Example 3: Top-N Scoring (External N-Best Indices)
--------------------------------------------------

When ``nbest_mode`` is not ``"master"``, pass per-sample component indices with
shape ``(num_samples, nbest)``.

.. code-block:: python

   import numpy as np
   from hyperion.np.pdfs.mixtures.gmm import GMM

   rng = np.random.default_rng(3)
   x = rng.standard_normal((100, 5)).astype("float32")

   gmm = GMM(num_comp=8, x_dim=5)
   _ = gmm.fit(x, epochs=2)

   # Example external shortlist from another model/stage
   nbest_idx = rng.integers(low=0, high=gmm.num_comp, size=(100, 2), dtype=np.intp)
   llk = gmm.log_prob_nbest(x, mode="std", nbest_mode="ubm", nbest=nbest_idx)
   print(llk.shape)  # (100,)


Example 4: Tied-Diagonal Covariance GMM
---------------------------------------

.. code-block:: python

   import numpy as np
   from hyperion.np.pdfs.mixtures.gmm_tied_diag_cov import GMMTiedDiagCov

   rng = np.random.default_rng(4)
   x = rng.standard_normal((1200, 12)).astype("float32")

   gmm = GMMTiedDiagCov(num_comp=6, x_dim=12)
   _ = gmm.fit(x, epochs=3, batch_size=256)

   samples = gmm.sample(num_samples=5, seed=7)
   print(samples.shape)  # (5, 12)


Example 5: Component Splitting
------------------------------

Start with a smaller model and split components to initialize a larger one.

.. code-block:: python

   import numpy as np
   from hyperion.np.pdfs.mixtures.gmm_diag_cov import GMMDiagCov

   rng = np.random.default_rng(5)
   x = rng.standard_normal((1800, 15)).astype("float32")

   gmm = GMMDiagCov(num_comp=4, x_dim=15)
   _ = gmm.fit(x, epochs=2)

   gmm_big = gmm.split_comp(K=2)  # 8 components
   _ = gmm_big.fit(x, epochs=1)


Tips
----

* Use ``GMMDiagCov`` first when you need robust/cheap training.
* Move to ``GMM`` when full covariance is needed and data volume is sufficient.
* For custom top-N selection, ensure non-master ``nbest`` is a 2D integer array.
