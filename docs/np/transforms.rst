Transforms Tutorial (NumPy)
===========================

This tutorial shows practical usage patterns for ``hyperion.np.transforms``.

See also:

* :doc:`../numpy`
* :doc:`mfcc`
* :doc:`speech_augmentation`

Quick Start
-----------

Train a simple normalization + PCA front-end:

.. code-block:: python

   import numpy as np
   from hyperion.np.transforms import MVN, PCA

   rng = np.random.default_rng(1234)
   x = rng.standard_normal((3000, 256))

   mvn = MVN()
   mvn.fit(x)
   x_mvn = mvn.predict(x)

   pca = PCA(pca_dim=128, whiten=True)
   pca.fit(x_mvn)
   x_pca = pca.predict(x_mvn)

   print(x_pca.shape)  # (3000, 128)


Example 1: Centering/Whitening + Length Normalization
-----------------------------------------------------

.. code-block:: python

   import numpy as np
   from hyperion.np.transforms import CentWhiten, LNorm

   rng = np.random.default_rng(1234)
   x = rng.standard_normal((2000, 256))

   cw = CentWhiten()
   cw.fit(x)
   x_cw = cw.predict(x)

   # LNorm can also learn centering/whitening if desired.
   lnorm = LNorm(update_mu=False, update_T=False)
   x_ln = lnorm.predict(x_cw)


Example 2: Supervised Projection with LDA
-----------------------------------------

.. code-block:: python

   import numpy as np
   from hyperion.np.transforms import LDA

   rng = np.random.default_rng(1234)
   x = rng.standard_normal((5000, 256))
   class_ids = rng.integers(0, 200, size=(5000,))

   lda = LDA(lda_dim=150)
   lda.fit(x, class_ids)
   x_lda = lda.predict(x)


Example 3: NAP and NDA (Class-Conditioned Nuisance Handling)
-------------------------------------------------------------

.. code-block:: python

   import numpy as np
   from hyperion.np.transforms import NAP, NDA

   rng = np.random.default_rng(1234)
   x = rng.standard_normal((4000, 256))
   class_ids = rng.integers(0, 150, size=(4000,))

   nap = NAP(U_dim=50)
   nap.fit(x, class_ids)
   x_nap = nap.predict(x)

   nda = NDA(nda_dim=120)
   nda.fit(x, class_ids)
   x_nda = nda.predict(x)


Example 4: CORAL Domain Adaptation
----------------------------------

.. code-block:: python

   import numpy as np
   from hyperion.np.transforms import CORAL

   rng = np.random.default_rng(1234)
   x_in = rng.standard_normal((1200, 256))   # in-domain
   x_out = rng.standard_normal((2000, 256))  # out-domain

   coral = CORAL(update_mu=True, update_T=True, alpha_mu=1.0, alpha_T=0.7)
   coral.fit(x=x_in, x_out=x_out)

   # Adapt out-domain embeddings to in-domain statistics.
   x_out_adapted = coral.predict(x_out)


Example 5: Build a Reusable Pipeline with TransformList
-------------------------------------------------------

.. code-block:: python

   import numpy as np
   from hyperion.np.transforms import MVN, PCA, TransformList

   rng = np.random.default_rng(1234)
   x = rng.standard_normal((3000, 256))

   mvn = MVN()
   mvn.fit(x)
   pca = PCA(pca_dim=128)
   pca.fit(mvn.predict(x))

   pipeline = TransformList([mvn, pca], name="frontend")
   y = pipeline.predict(x)


Example 6: t-SNE Embedding for Visualization
--------------------------------------------

``SklTSNE`` is non-parametric; both ``fit`` and ``predict`` run
``fit_transform`` on the input batch.

.. code-block:: python

   import numpy as np
   from hyperion.np.transforms import SklTSNE

   rng = np.random.default_rng(1234)
   x = rng.standard_normal((1500, 128))

   tsne = SklTSNE(tsne_dim=2, perplexity=30.0, num_iter=1000, rng_seed=1234)
   y2d = tsne.fit(x)
   print(y2d.shape)  # (1500, 2)


Tips
----

* Fit transforms on a representative training set, then apply them to dev/eval data.
* Keep feature dimensions consistent across chained transforms.
* Use ``TransformList`` for deterministic, reusable inference pipelines.
