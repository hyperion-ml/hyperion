Backend, Scoring, and Evaluation Commands
==========================================

This family consumes CSV manifests, embedding indexes, trial tables, and score
files to produce scores, calibration/fusion outputs, metrics, or reports.

Score backends
--------------

``hyperion-eval-cosine-scoring-backend`` and
``hyperion-eval-plda-backend`` produce score files for trial pairs. They need an
enrollment map, a trial index/key, and a CSV embedding index. PLDA additionally
needs saved preprocessing and backend files.

.. code-block:: bash

   hyperion-eval-cosine-scoring-backend \
     --enroll-map-file data/eval/enrollment.csv \
     --ndx-file data/eval/trials.key \
     --feats-file exp/eval/xvectors.csv \
     --score-file exp/eval/cosine_scores.h5

Calibration, QMF, and greedy fusion commands transform score artifacts; train
their parameters on development data only. Cluster and LGBE commands likewise
operate on existing embedding/score artifacts rather than raw audio.

Verification evaluation
-----------------------

``hyperion-eval-verification-metrics`` requires aligned key and score files:

.. code-block:: bash

   hyperion-eval-verification-metrics \
     --key-files data/eval/trials.key \
     --key-names eval \
     --score-files exp/eval/cosine_scores.h5 \
     --score-names cosine \
     --output-file exp/eval/metrics.csv

It writes tabular metrics and optional plots. Never fill missing scores with a
default value without documenting the protocol impact.

Speech quality, VoxProfile, and adversarial evaluation
------------------------------------------------------

Speech-quality and VoxProfile commands are stable but have conditional runtime
requirements: TPM packages/extras, model assets, writable caches, and sometimes
first-run network retrieval. Use local model paths and record the model revision,
asset checksum, package version, cache location, and device for every report.
DNSMOS can continue without its optional P.808 regressor, so record whether that
score was enabled. See :doc:`../optional-dependencies` for the complete policy.
Adversarial scoring/evaluation and attack-generation commands are stable PyTorch
workflows; preserve the model, attack, and perturbation configuration with every
report.

See also
--------

* :doc:`../how-to/extract-score-xvectors`
* :doc:`../metrics`
* :doc:`../statistical-api-contracts`
