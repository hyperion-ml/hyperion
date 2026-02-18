Command-Line Interface
======================

Entry point model
-----------------

Hyperion CLI commands are generated from scripts in ``hyperion/bin``.

Generation flow:

1. ``proto_pyproject.toml`` defines static package metadata template.
2. ``generate_pyproject.py`` scans ``hyperion/bin/*.py``.
3. Each script is mapped to ``hyperion-<script-name>`` in ``pyproject.toml``.
4. Dependencies are loaded from ``requirements.txt``.

Regenerate entry points with:

.. code-block:: bash

   python generate_pyproject.py

Naming convention
-----------------

A script file like ``hyperion/bin/train_qvector.py`` becomes command:

.. code-block:: bash

   hyperion-train-qvector

Current command groups
----------------------

Training / fine-tuning:

* ``hyperion-train-qvector``
* ``hyperion-train-dac``
* ``hyperion-train-freevc``
* ``hyperion-train-vi-anonymizer``
* ``hyperion-train-wav2xvector``
* ``hyperion-train-wav2vec2xvector``
* ``hyperion-train-xvector-from-wav``
* ``hyperion-finetune-dac``
* ``hyperion-finetune-vi-anonymizer``
* ``hyperion-finetune-wav2xvector``

Extraction / inference:

* ``hyperion-extract-wav2xvectors``
* ``hyperion-extract-wav2vec2xvectors``
* ``hyperion-extract-xvectors-from-wav``
* ``hyperion-infer-qvectors``

Evaluation / scoring:

* ``hyperion-eval-verification-metrics``
* ``hyperion-eval-verification-calibration``
* ``hyperion-eval-verification-greedy-fusion``
* ``hyperion-eval-speech-quality-metrics``
* ``hyperion-eval-voxprofile-metrics``
* ``hyperion-eval-plda-backend``

Data preparation / utilities:

* ``hyperion-prepare-data``
* ``hyperion-preprocess-audio-files``
* ``hyperion-compute-mfcc-feats``
* ``hyperion-compute-energy-vad``
* ``hyperion-convert-vad-format``
* ``hyperion-dataset``
* ``hyperion-tables``

Scope notes
-----------

* ``hyperion/bin_deprec`` and ``hyperion/bin_deprec2`` are deprecated and not
  part of the documented CLI surface.
* ``decode_wav2transducer`` and ``decode_wav2vec2rnn_transducer`` are currently
  excluded from this documentation scope.

Discover commands
-----------------

After installation:

.. code-block:: bash

   python -m pip show hyperion-ml
   hyperion-train-qvector --help
   hyperion-eval-verification-metrics --help
