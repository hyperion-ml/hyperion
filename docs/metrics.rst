Metrics and Evaluation API
==========================

Hyperion separates metric primitives from evaluators. Use NumPy functions for
in-memory score arrays; use evaluators or
``hyperion-eval-verification-metrics`` when scores and keys are stored in
tables/files.

Verification metrics
--------------------

The primary verification contract is a ``TrialKey`` plus an aligned
``TrialScores`` object. The evaluator extracts target/non-target scores, then
computes EER and min/actual DCF for the requested target priors.

.. autofunction:: hyperion.np.metrics.compute_eer

.. autofunction:: hyperion.np.metrics.compute_min_dcf

.. autoclass:: hyperion.metrics.VerificationEvaluator
   :no-index:
   :members: compute_dcf_eer, compute_equalized_dcf_eer, get_tar_non

Use the command-line workflow in :doc:`how-to/extract-score-xvectors` for
CSV/HDF5 result files and DET/DCF plots. EER is threshold-independent; DCF
depends on the declared prior and costs, so only compare DCF values computed
under the same policy.

Other evaluator families
------------------------

.. autoclass:: hyperion.metrics.VerificationAdvAttackEvaluator
   :no-index:
   :members: compute_dcf_eer_vs_stats

.. autoclass:: hyperion.metrics.VerificationAnonymizationEvaluator
   :no-index:

.. autoclass:: hyperion.metrics.SpeechQualityEvaluator
   :no-index:

.. autoclass:: hyperion.metrics.VoxProfileEvaluator
   :no-index:

The anonymization/voice-conversion workflows that feed these evaluators are
experimental. TPM-backed quality and VoxProfile evaluators may require their
corresponding optional external packages.

Torch metrics
-------------

``hyperion.torch.metrics`` contains training-loop metrics such as categorical
accuracy. These are distinct from speaker-verification EER/DCF evaluation and
should not be used as a substitute for trial-based evaluation.

See also
--------

* :doc:`trials`
* :doc:`how-to/extract-score-xvectors`
* :doc:`data-model`
