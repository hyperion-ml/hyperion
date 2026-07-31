NumPy Backend API
=================

``hyperion.np`` provides the statistical backend used after embedding
extraction: preprocessing transforms, PLDA, calibration, score normalization,
and array-level metrics. These components operate on NumPy arrays and serialize
their state independently of PyTorch checkpoints.

Model serialization
-------------------

.. autoclass:: hyperion.np.HyperNPModel
   :no-index:
   :members: get_config, save, load, auto_load

``HyperNPModel`` subclasses register by class name and persist configuration
plus parameters in HDF5-style files. Use :meth:`auto_load` when callers should
restore the recorded concrete model class. See
:doc:`how-to/save-load-models-and-backends` for deployment guidance.

Transforms and preprocessing
----------------------------

.. autoclass:: hyperion.np.transforms.TransformList
   :no-index:
   :members: fit, predict, save, load

Use ``TransformList`` to preserve the ordered preprocessing chain used for a
backend. Fit transforms on development data only, then load and apply the same
chain to enrollment, test, and cohort embeddings. The principal transforms are
centering/whitening, length normalization, PCA, LDA, and CORAL.

* :doc:`np/transforms` provides worked transform examples.
* :doc:`how-to/extract-score-xvectors` shows transform use with PLDA.

PLDA and scoring backends
-------------------------

.. autoclass:: hyperion.np.pdfs.plda.factory.PLDAFactory
   :no-index:
   :members: create, add_class_args

PLDA backends consume matrices shaped ``(num_embeddings, embedding_dim)`` and
speaker/class ids aligned with rows during training. Scoring returns a matrix
whose rows correspond to enrollment models and columns to test segments.

* :doc:`np/pdfs/plda` covers SPLDA, FRPLDA, full PLDA, and N-vs-M scoring.
* Use ``hyperion-train-plda`` and ``hyperion-eval-plda-backend`` for the
  file/table workflow.

Calibration and score normalization
-----------------------------------

Calibration turns raw scores into application-specific calibrated scores using
development keys. Score normalization uses background cohorts. Both must be
fit without evaluation labels to avoid contaminating metrics.

.. autoclass:: hyperion.np.classifiers.binary_logistic_regression.BinaryLogisticRegression
   :no-index:
   :members: fit, predict, save, load

.. autoclass:: hyperion.np.score_norm.adapt_s_norm.AdaptSNorm
   :no-index:

See :doc:`how-to/extract-score-xvectors` for calibration and adaptive S-Norm
placement in a verification pipeline.

Other stable areas
------------------

The NumPy stack also includes speech augmentation, features, classifiers,
clustering, diarization, and metric utilities. Their public use should be
guided by the corresponding task documentation rather than by importing every
implementation module directly.

The contract-level reference for calibration, cohort score normalization,
clustering, diarization, and augmentation is :doc:`numpy-extension-points`.

.. toctree::
   :maxdepth: 1

   np/mfcc
   np/pdfs/mixtures
   np/pdfs/plda
   np/transforms
   np/speech_augmentation

See also
--------

* :doc:`metrics`
* :doc:`numpy-extension-points`
* :doc:`how-to/extract-score-xvectors`
* :doc:`how-to/save-load-models-and-backends`
