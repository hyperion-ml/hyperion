Statistical, Evaluation, and Preparation API Contracts
======================================================

This page records the public behavior of Hyperion's NumPy backends, evaluators,
data preparers, and text normalizers. It complements the signatures in
:doc:`numpy`, :doc:`metrics`, :doc:`data_prep`, and :doc:`text_norm`.

Metrics and evaluators
----------------------

Verification metric functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`hyperion.np.metrics.compute_eer` accepts one-dimensional target and
non-target score arrays and returns the equal-error rate and its decision
threshold. The arrays represent scores, not labels: higher scores must mean
greater target confidence. Empty target/non-target inputs or non-finite scores
are invalid evaluation inputs.

:func:`hyperion.np.metrics.compute_min_dcf` accepts the same score convention
plus an application policy (target prior and miss/false-alarm costs). It returns
the minimum normalized detection cost and threshold. DCF values are comparable
only when the policy is identical.

Neither function mutates the supplied arrays. Their inputs normally come from
``TrialScores.get_tar_non(TrialKey)`` after score/key alignment; see
:doc:`foundation-api-contracts`.

``VerificationEvaluator``
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`hyperion.metrics.VerificationEvaluator` is the file/table-level
verification contract. It consumes a trial key and scores, extracts target and
non-target values, and computes EER/DCF under its configured policy. Methods
such as ``compute_dcf_eer`` return scalar metrics and thresholds; plotting or
report-writing methods create output artifacts at caller-supplied paths.

The evaluator does not rescore embeddings or infer missing trials. Score/key
misalignment, missing score availability, or incompatible table axes must be
resolved before evaluation. Use a fixed evaluation policy and persist it beside
reports so metrics are reproducible.

Quality, anonymization, and VoxProfile evaluators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`hyperion.metrics.VerificationAnonymizationEvaluator` evaluates the
privacy/utility verification trade-off from aligned original/anonymized score
sets. :class:`hyperion.metrics.SpeechQualityEvaluator` reads recordings and
segments, then optionally writes per-segment and aggregate speech-quality/ASR
metrics. :class:`hyperion.metrics.VoxProfileEvaluator` enriches a segment table
with attribute predictions and aggregate statistics.

These evaluators may load TPM-backed models and may create CSV reports or
plots. They are not pure functions: input manifests are read, output paths are
mutated, and optional model packages/assets must be installed. Validate audio
sample rates, segment ids, reference mappings, and requested model availability
before long evaluation jobs. The anonymization workflow itself remains
experimental even though the evaluator interface is stable.

Data preparation
----------------

``DataPrep``
~~~~~~~~~~~~

:class:`hyperion.data_prep.DataPrep` is the registered corpus-preparer base
class. A subclass supplies a unique ``dataset_name()``, parser integration via
``add_class_args()``, and corpus-specific parsing. Registration lets
``hyperion-prepare-data`` select the preparer by name.

Preparers consume a corpus root and output directory and write standard CSV
recording/segment/class manifests. They may inspect audio headers to determine
duration or sample frequency, but do not train a model. Invalid corpus layout,
missing annotations, duplicate ids, and unreadable audio are preparation
errors; fail before writing a partially trusted manifest whenever possible.

The output contract is more important than a corpus-specific internal parser:
recording ids and paths must resolve; segment ids must be unique; segment timing
must match its recording; and labels must refer to known class ids. See
:doc:`how-to/prepare-data-and-vad` for a minimal invocation and validation.

NumPy model and backend contracts
---------------------------------

``HyperNPModel``
~~~~~~~~~~~~~~~~

:class:`hyperion.np.HyperNPModel` is the serializable NumPy model base. Its
subclasses register a ``class_name`` and provide ``get_config()``. ``save``
persists configuration and parameters; ``load`` restores a known concrete
class; ``auto_load`` reads the recorded class and restores the matching
registered implementation.

Serialization is a compatibility boundary. Configuration values must be
JSON/HDF5-friendly, and a checkpoint must be loaded with code that supports the
recorded class/configuration. Saving mutates the target file; loading returns a
new model instance and does not modify the source file.

Transforms, PLDA, calibration, and score normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TransformList`` holds an ordered list of fitted transforms. ``fit(x)``
estimates state from a float array shaped ``(num_samples, feature_dim)``;
``predict(x)`` applies each transform in order and returns an array whose row
order matches ``x``. Fitting mutates transform state. Apply development-fitted
transforms unchanged to enrollment, test, and cohort embeddings.

``PLDAFactory.create(...)`` selects a PLDA implementation from configuration.
PLDA fitting requires embeddings with shape ``(num_embeddings, embedding_dim)``
and row-aligned class/speaker ids. PLDA scoring returns a model-by-test score
matrix; construct the relevant ``TrialScores`` mask from trials rather than
assuming every pair is evaluated.

``BinaryLogisticRegression`` fits a binary score calibrator. Its input feature
matrix has one row per development trial and its labels identify target versus
non-target trials. ``fit`` mutates learned coefficients; ``predict`` returns
posterior/logit-style values according to ``eval_type``. The configured prior,
regularization, solver, bias behavior, and random state are part of the saved
model configuration and must be retained for reproducible calibration.

``AdaptSNorm`` applies adaptive symmetric score normalization. It combines a
model-by-test score matrix with cohort-versus-test and model-versus-cohort score
matrices. Their model, test, and cohort axes must be compatible. It returns a
normalized score matrix of the same model-by-test shape; optional returned
statistics describe selected cohorts. Cohort selection and standard-deviation
floor settings affect the result and should be stored with the evaluation
configuration.

Focused backend example
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hyperion.np.transforms import TransformList
   from hyperion.np.transforms import MVN

   transform = TransformList([MVN()])
   transform.fit(x_dev)                 # x_dev: (num_dev, embedding_dim)
   x_enroll = transform.predict(x_enroll)
   x_test = transform.predict(x_test)
   transform.save("exp/backend/transform.h5")

Text normalization
------------------

``BasicTextNormalizer``
~~~~~~~~~~~~~~~~~~~~~~~

:class:`hyperion.text_norm.BasicTextNormalizer` applies deterministic text
cleanup such as case, punctuation, or whitespace normalization according to its
configuration. It accepts and returns text strings; it does not infer language,
translate, or modify an input file unless the caller writes the returned text.

``EnglishTextNormalizer`` and number normalizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`hyperion.text_norm.EnglishTextNormalizer` adds supported English text
rules. :class:`hyperion.text_norm.EnglishNumberNormalizer` converts recognized
written/numeric number expressions to normalized Arabic-number forms;
:class:`hyperion.text_norm.EnglishReverseNumberNormalizer` performs the reverse
direction where supported. ``SpellingNormalizer`` applies configured spelling
normalizations.

These APIs are language-specific transformations, not universal linguistic
models. Preserve raw transcripts alongside normalized versions, especially for
ASR scoring and reproducibility. Unknown words are generally retained rather
than guessed; callers should test domain-specific abbreviations, names, and
punctuation before applying a normalizer to a corpus.

See also
--------

* :doc:`metrics`
* :doc:`numpy-extension-points`
* :doc:`how-to/save-load-models-and-backends`
* :doc:`how-to/prepare-data-and-vad`
