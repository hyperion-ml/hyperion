Extract, Score, and Evaluate X-Vectors
=======================================

This guide continues from :doc:`train-waveform-xvector`. It extracts embeddings
from waveform audio, scores speaker-verification trials, and writes EER/minDCF
results. The same scoring steps apply to native Wav2XVector and pretrained
Wav2Vec2-family x-vector checkpoints.

Prepare evaluation metadata
---------------------------

You need four aligned inputs:

* An evaluation ``HyperDataset`` manifest, or recording and segment manifests
  for extraction.
* An embedding archive produced by the extractor. Every trial segment must
  have a corresponding embedding key.
* An ``EnrollmentMap`` that maps each enrollment model id to one or more
  enrollment segment ids.
* A ``TrialNdx`` or ``TrialKey`` whose rows are enrollment model ids and whose
  columns are test segment ids.

Use a ``TrialKey`` when you will compute metrics: it supplies target and
non-target labels. The score commands also accept a ``TrialNdx`` when only the
set of trials is known. See :doc:`../data-model` for the ordering contract.

Extract embeddings
------------------

For a native waveform x-vector checkpoint:

.. code-block:: bash

   hyperion-extract-wav2xvectors \
     --dataset-path data/eval/dataset.yaml \
     --model-path exp/wav2xvector-resnet1d/<checkpoint> \
     --xvector-path ark,csv:exp/eval/xvectors.ark,exp/eval/xvectors.csv \
     --use-gpu

``--dataset-path`` and ``--recordings-file`` are mutually exclusive. If you
use recordings directly, provide ``--segments-file`` as needed. The extractor
resamples waveforms to the checkpoint's expected sample frequency and stores a
``speech_duration`` metadata field with each vector.

For a Wav2Vec2/HuBERT/WavLM/Whisper x-vector checkpoint, use the matching
extractor:

.. code-block:: bash

   hyperion-extract-wav2vec2xvectors \
     --dataset-path data/eval/dataset.yaml \
     --model-path exp/hf-wav2vec2-resnet1d/<checkpoint> \
     --xvector-path ark,csv:exp/eval/xvectors.ark,exp/eval/xvectors.csv \
     --use-gpu

Long recordings can be processed in chunks with the extractor's chunk-length
options. Use VAD only when its frame alignment and speech-removal behavior are
appropriate for the model and evaluation protocol.

Score with cosine similarity
----------------------------

Cosine scoring is the simplest baseline. It averages enrollment vectors per
model when the enrollment map contains multiple segments, L2-normalizes the
vectors, and produces a score for every trial in the index/key.

.. code-block:: bash

   hyperion-eval-cosine-scoring-backend \
     --enroll-map-file data/eval/enrollment.csv \
     --ndx-file data/eval/trials.key \
     --feats-file exp/eval/xvectors.csv \
     --score-file exp/eval/cosine_scores.h5

Keep the embedding keys, enrollment-map segment ids, and trial-key segment ids
identical. Missing keys or mismatched ids are data errors, not scores to fill
with zeros.

Train and score a PLDA backend
------------------------------

PLDA normally needs separate labeled development embeddings. Train the
preprocessing transform and PLDA model from a segment table with a speaker
column and an aligned embedding archive:

.. code-block:: bash

   hyperion-train-plda \
     --segments-files data/dev/segments.csv \
     --feats-files exp/dev/xvectors.csv \
     --class-name speaker \
     --preproc-file exp/backend/preproc.h5 \
     --plda-file exp/backend/plda.h5

Score evaluation trials with the resulting backend:

.. code-block:: bash

   hyperion-eval-plda-backend \
     --enroll-map-file data/eval/enrollment.csv \
     --ndx-file data/eval/trials.key \
     --feats-file exp/eval/xvectors.csv \
     --preproc-file exp/backend/preproc.h5 \
     --plda-file exp/backend/plda.h5 \
     --score-file exp/eval/plda_scores.h5

The development data used for PLDA must be disjoint from evaluation labels and
trials. Apply the exact saved preprocessing transform during evaluation; do
not recompute normalization parameters on the evaluation set.

Evaluate EER and minDCF
-----------------------

Pass the trial key and one or more score files to the metrics command:

.. code-block:: bash

   hyperion-eval-verification-metrics \
     --key-files data/eval/trials.key \
     --key-names eval \
     --score-files exp/eval/cosine_scores.h5 exp/eval/plda_scores.h5 \
     --score-names cosine plda \
     --p-tar 0.01 0.001 \
     --output-file exp/eval/metrics.csv \
     --plot-det \
     --det-file exp/eval/det.pdf

The command prints a summary and writes the tabular metrics file. EER reflects
the equal false-accept/false-reject operating point. minDCF depends on the
requested target prior and costs; compare systems only under the same metric
configuration and trial key.

Calibrate scores only with development data
-------------------------------------------

For calibrated log-likelihood-ratio scores, train logistic-regression
calibration on development score/key pairs, then apply it to evaluation scores:

.. code-block:: bash

   hyperion-train-verification-calibration \
     --score-files exp/dev/plda_scores.h5 \
     --key-files data/dev/trials.key \
     --model-file exp/backend/calibration.h5 \
     --prior 0.01

   hyperion-eval-verification-calibration \
     --in-score-file exp/eval/plda_scores.h5 \
     --ndx-file data/eval/trials.key \
     --model-file exp/backend/calibration.h5 \
     --out-score-file exp/eval/plda_scores_calibrated.h5

Never train calibration, PLDA, or score-normalization parameters using the
evaluation labels. That contaminates the result.

Common problems
---------------

* **Score alignment error:** model or segment ids differ across embeddings,
  enrollment map, and trials. Verify the ids before scoring.
* **Non-finite embeddings:** locate and remove/repair invalid extracted vectors
  before fitting PLDA.
* **Inconsistent preprocessing:** cosine and PLDA experiments use different
  embedding preparation. Record every transform with the experiment.
* **Evaluation leakage:** backend or calibration development data overlaps the
  labeled evaluation trials. Split data by speaker and protocol as appropriate.

See also
--------

* :doc:`train-waveform-xvector`
* :doc:`../trials`
* :doc:`../metrics`
