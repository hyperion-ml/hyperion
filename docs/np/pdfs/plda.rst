PLDA Tutorial (NumPy)
=====================

This tutorial shows practical usage of PLDA classes in
``hyperion.np.pdfs.plda``:

* :class:`hyperion.np.pdfs.plda.frplda.FRPLDA`
* :class:`hyperion.np.pdfs.plda.splda.SPLDA`
* :class:`hyperion.np.pdfs.plda.plda.PLDA`
* :class:`hyperion.np.pdfs.plda.factory.PLDAFactory`

See also:

* :doc:`../../numpy`
* :doc:`mixtures`

Quick Start
-----------

Train an SPLDA backend and compute 1-vs-1 scores:

.. code-block:: python

   import numpy as np
   from hyperion.np.pdfs.plda.splda import SPLDA

   rng = np.random.default_rng(1234)
   x = rng.standard_normal((2000, 256)).astype("float32")
   class_ids = np.repeat(np.arange(200), 10)

   plda = SPLDA(y_dim=128, fullcov_W=True, epochs=5)
   _ = plda.fit(x, class_ids=class_ids)

   scores = plda.llr_1vs1(x[:5], x[5:15])
   print(scores.shape)  # (5, 10)


Example 1: FRPLDA (Two-Covariance Model)
----------------------------------------

.. code-block:: python

   import numpy as np
   from hyperion.np.pdfs.plda.frplda import FRPLDA

   rng = np.random.default_rng(1)
   x = rng.standard_normal((1500, 192)).astype("float32")
   class_ids = np.repeat(np.arange(150), 10)

   model = FRPLDA(fullcov_W=True, epochs=3)
   _ = model.fit(x, class_ids=class_ids)

   llr = model.llr_1vs1(x[:20], x[20:40])
   print(llr.shape)  # (20, 20)


Example 2: N-vs-1 Scoring with Segment Groups
---------------------------------------------

Use side IDs when you have multiple enrollment segments per speaker side.

.. code-block:: python

   import numpy as np
   from hyperion.np.pdfs.plda.plda_base import PLDALLRNvsMMethod
   from hyperion.np.pdfs.plda.splda import SPLDA

   rng = np.random.default_rng(2)
   x = rng.standard_normal((1000, 128)).astype("float32")
   class_ids = np.repeat(np.arange(100), 10)

   model = SPLDA(y_dim=64, fullcov_W=True, epochs=3)
   _ = model.fit(x, class_ids=class_ids)

   x_enroll = x[:300]
   x_test = x[300:500]
   enroll_ids = np.repeat(np.arange(100), 3)  # 300 segments -> 100 sides

   scores = model.llr_Nvs1(
       x_enroll,
       x_test,
       ids1=enroll_ids,
       method=PLDALLRNvsMMethod.lnorm_vavg,
   )
   print(scores.shape)  # (100, 200)


Example 3: Full PLDA (Speaker + Channel Latents)
------------------------------------------------

.. code-block:: python

   import numpy as np
   from hyperion.np.pdfs.plda.plda import PLDA
   from hyperion.np.pdfs.plda.plda_base import PLDALLRNvsMMethod

   rng = np.random.default_rng(3)
   x = rng.standard_normal((1200, 128)).astype("float32")
   class_ids = np.repeat(np.arange(120), 10)

   model = PLDA(y_dim=48, z_dim=48, epochs=3)
   _ = model.fit(x, class_ids=class_ids)

   enroll = x[:400]
   test = x[400:800]
   scores = model.llr_NvsM(enroll, test, method=PLDALLRNvsMMethod.book)
   print(scores.shape)  # (400, 400)


Example 4: Create Models from PLDAFactory
-----------------------------------------

.. code-block:: python

   from hyperion.np.pdfs.plda.factory import PLDAFactory, PLDAType

   splda = PLDAFactory.create(
       plda_type=PLDAType.SPLDA,
       y_dim=150,
       fullcov_W=True,
       update_mu=True,
       update_V=True,
       update_W=True,
   )

   frplda = PLDAFactory.create(
       plda_type=PLDAType.FRPLDA,
       fullcov_W=True,
   )


Example 5: Adaptation with a Prior PLDA
---------------------------------------

Adapt a model and average parameters with a prior at each epoch.

.. code-block:: python

   import numpy as np
   from hyperion.np.pdfs.plda.splda import SPLDA

   rng = np.random.default_rng(4)
   x0 = rng.standard_normal((1500, 96)).astype("float32")
   ids0 = np.repeat(np.arange(150), 10)

   prior = SPLDA(y_dim=48, epochs=3)
   _ = prior.fit(x0, class_ids=ids0)

   x_adapt = rng.standard_normal((400, 96)).astype("float32")
   ids_adapt = np.repeat(np.arange(80), 5)

   model = SPLDA(y_dim=48, prior=prior, epochs=3)
   model.initialize(model.compute_stats_hard(x_adapt, ids_adapt))
   _ = model.fit_adapt_weighted_avg_model(
       x_adapt,
       class_ids=ids_adapt,
       plda0=prior,
       w_mu=0.8,
       w_B=0.5,
       w_W=0.5,
       epochs=2,
   )


Example 6: Sampling Synthetic Embeddings
----------------------------------------

.. code-block:: python

   import numpy as np
   from hyperion.np.pdfs.plda.splda import SPLDA

   rng = np.random.default_rng(5)
   x = rng.standard_normal((800, 64)).astype("float32")
   class_ids = np.repeat(np.arange(80), 10)

   model = SPLDA(y_dim=32, epochs=2)
   _ = model.fit(x, class_ids=class_ids)

   x_sampled = model.sample(num_classes=20, num_samples_per_class=5, seed=7)
   print(x_sampled.shape)  # (100, 64)


Tips
----

* Start with ``SPLDA`` as a strong default backend for x-vector style embeddings.
* Use ``FRPLDA`` when you specifically want a two-covariance parameterization.
* Use ``PLDA`` when channel-latent modeling is important and you have enough data.
* For ``savg`` scoring with IDs, remap IDs to contiguous integers ``0..K-1`` to avoid empty classes.
