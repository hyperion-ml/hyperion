Speaker Verification Quickstart
===============================

This quickstart shows the smallest complete speaker-verification evaluation
workflow in Hyperion. It creates deterministic synthetic embeddings, scores
every enrollment/test pair with cosine similarity, represents the labeled
trials with Hyperion tables, and computes equal error rate (EER).

It is intentionally model- and audio-free. In a real system, replace the
synthetic embeddings with vectors extracted by an embedding model; the trial,
score, and evaluation steps stay the same.

Requirements
------------

Install Hyperion and its regular Python dependencies as described in
:doc:`getting-started`. No GPU, dataset download, or ``egs/`` recipe is needed
for this example.

Create embeddings and trials
----------------------------

The code below creates four speakers. Each speaker has one enrollment
embedding and three test embeddings. Test embeddings are generated near their
speaker's enrollment representation, so target trials should score higher than
non-target trials.

.. code-block:: python

   import numpy as np

   from hyperion.np.metrics import compute_eer
   from hyperion.utils import TrialKey, TrialScores
   from hyperion.utils.math_funcs import cosine_scoring

   rng = np.random.default_rng(1234)
   num_speakers = 4
   embedding_dim = 192
   tests_per_speaker = 3

   model_ids = np.array([f"spk-{i}" for i in range(num_speakers)])
   speaker_centers = rng.normal(size=(num_speakers, embedding_dim))
   enroll_embeddings = speaker_centers + 0.10 * rng.normal(
       size=speaker_centers.shape
   )

   test_speaker_ids = np.repeat(model_ids, tests_per_speaker)
   test_embeddings = np.repeat(speaker_centers, tests_per_speaker, axis=0)
   test_embeddings += 0.10 * rng.normal(size=test_embeddings.shape)
   segment_ids = np.array([f"test-{i}" for i in range(len(test_embeddings))])

   # Rows are enrollment models; columns are test segments.
   is_target = model_ids[:, None] == test_speaker_ids[None, :]
   key = TrialKey(
       model_set=model_ids,
       seg_set=segment_ids,
       tar=is_target,
       non=~is_target,
   )

Score and evaluate
------------------

``cosine_scoring`` L2-normalizes each input vector internally and produces a
score matrix of shape ``(num_models, num_test_segments)``. ``TrialScores``
keeps that matrix aligned with the trial key, then ``get_tar_non`` extracts
the target and non-target score vectors needed by the metric.

.. code-block:: python

   score_matrix = cosine_scoring(enroll_embeddings, test_embeddings)
   scores = TrialScores(
       model_set=model_ids,
       seg_set=segment_ids,
       scores=score_matrix,
       score_mask=np.ones(score_matrix.shape, dtype=bool),
   )

   target_scores, non_target_scores = scores.get_tar_non(key)
   eer = compute_eer(target_scores, non_target_scores)

   print(f"Score matrix: {scores.scores.shape}")
   print(f"Target trials: {len(target_scores)}")
   print(f"Non-target trials: {len(non_target_scores)}")
   print(f"EER: {eer:.2%}")

The score matrix has four rows and twelve columns: one row per enrollment
speaker and one column per test segment. There are twelve target trials and
thirty-six non-target trials. With this deliberately well-separated synthetic
data, the EER should be close to zero.

What the tables mean
--------------------

* ``TrialKey`` identifies every valid trial and labels it as target or
  non-target. Its row order is ``model_set`` and its column order is
  ``seg_set``.
* ``TrialScores`` stores a score for the same ordered model/segment grid. Its
  ``score_mask`` says which entries are present.
* ``compute_eer`` compares the target-score distribution with the
  non-target-score distribution. Lower EER is better.

Next steps
----------

* Learn the full vocabulary and alignment rules in :doc:`data-model`.
* Read :doc:`trials` for dense/sparse trial tables and on-disk formats.
* Replace the synthetic vectors with real embeddings in the extraction and
  scoring guides added in the next documentation phase.
