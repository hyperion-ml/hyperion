NumPy Backend Extension Points
==============================

This page covers the stable NumPy components that turn embeddings into scores,
clusters, and evaluation-ready decisions. These components use row-aligned
NumPy arrays and serialize independently from PyTorch model checkpoints. Read
:doc:`data-model` first when the inputs originate in tables or trial files.

Score backend pipeline
----------------------

The usual verification sequence is:

#. fit preprocessing and PLDA on development embeddings and speaker labels;
#. score enrollment versus test embeddings;
#. optionally apply cohort-based score normalization; and
#. fit calibration on separate labelled development trials.

Never use evaluation labels when fitting preprocessing, cohort statistics, or
calibration. Keep the fitted transforms, PLDA backend, score normalizer, and
calibrator as separate saved artifacts so they can be reproduced or replaced
independently.

PLDA and transforms
-------------------

.. autoclass:: hyperion.np.transforms.transform_list.TransformList
   :no-index:
   :members: fit, predict, save, load

.. autoclass:: hyperion.np.pdfs.plda.factory.PLDAFactory
   :no-index:
   :members: create, add_class_args

``TransformList`` preserves the order of preprocessing operations. PLDA
training expects one embedding row per class-id row. A score matrix returned by
a PLDA backend uses enrollment rows and test columns. Detailed model choices
are documented in :doc:`np/pdfs/plda`.

Calibration
-----------

.. autoclass:: hyperion.np.calibration.gauss_calibration.GaussCalibration
   :no-index:
   :members: fit, predict, save, load

``GaussCalibration`` learns an affine score mapping from one-dimensional
development scores and binary labels (target is ``1``, non-target is ``0``).
It requires both classes and a non-zero shared variance. For discriminative
calibration, use ``BinaryLogisticRegression`` from :doc:`numpy`.

Score normalization
-------------------

.. autoclass:: hyperion.np.score_norm.score_norm.ScoreNorm
   :no-index:
   :members: get_config, save, load

.. autoclass:: hyperion.np.score_norm.adapt_s_norm.AdaptSNorm
   :no-index:
   :members: predict

Score normalization is a cohort operation, not a classifier. Adaptive S-Norm
accepts three matrices: enrollment-versus-test scores, cohort-versus-test
scores, and enrollment-versus-cohort scores. Cohort dimensions must agree with
the corresponding rows or columns. Mask unavailable cohort trials instead of
silently changing matrix alignment.

Clustering and diarization
--------------------------

.. autoclass:: hyperion.np.clustering.ahc.AHC
   :no-index:
   :members: fit, get_flat_clusters

.. autoclass:: hyperion.np.clustering.spectral_clustering.SpectralClustering
   :no-index:
   :members: fit, predict_num_clusters

``AHC`` accepts a square pairwise score or distance matrix; set ``metric`` to
match the matrix semantics. Its threshold direction differs for LLR/probability
scores and distances, so save the configured metric with the backend.
``SpectralClustering`` accepts an affinity-style square matrix and can either
use a fixed cluster count or estimate it from eigengap statistics.

.. autoclass:: hyperion.np.diarization.diar_ahc_plda.DiarAHCPLDA
   :no-index:
   :members: __call__

``DiarAHCPLDA`` combines optional preprocessing, PLDA or cosine scoring,
optional calibration, and AHC. Its input rows are speech segments; optional
start/end arrays must remain aligned with those rows. The returned cluster ids
describe speakers for those segments, not global speaker identities.

Speech augmentation
-------------------

.. autoclass:: hyperion.np.augment.speech_augment.SpeechAugment
   :no-index:
   :members: create, reseed, forward

``SpeechAugment`` composes speed, reverberation, noise, and codec effects for a
one-dimensional waveform. Give it an explicit random seed or generator for
reproducible experiments, and record the returned augmentation metadata with
the experiment configuration. The complete configuration schema and CSV
manifest examples are in :doc:`np/speech_augmentation`.

See also
--------

* :doc:`numpy`
* :doc:`how-to/extract-score-xvectors`
* :doc:`how-to/save-load-models-and-backends`
