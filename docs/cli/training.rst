Training Commands
=================

This family covers stable model and backend fitting commands. The full command
list is in :doc:`../cli`; use each command's ``--help`` for parser-specific
subcommands and defaults.

Prerequisites and inputs
------------------------

Waveform x-vector training needs CSV recording, segment, and class manifests.
Feature x-vector and backend training need CSV feature indexes plus segment
tables whose labels align with feature ids. PyTorch commands need a compatible
PyTorch installation; Hugging Face commands additionally need ``transformers``
and access to requested model assets. Pin model revisions and provide local
assets/caches for offline training; see :doc:`../optional-dependencies`.

Native waveform x-vector
------------------------

``hyperion-train-wav2xvector`` trains a frontend plus x-vector model. A minimal
configuration keeps paths and reproducibility settings in YAML:

.. code-block:: yaml

   data:
     train:
       dataset:
         recordings_file: data/train/recordings.csv
         segments_file: data/train/segments.csv
         class_names: [speaker]
         class_files: [data/classes/speaker.csv]
   trainer:
     exp_path: exp/wav2xvector

.. code-block:: bash

   hyperion-train-wav2xvector resnet1d --cfg configs/wav2xvector.yaml

Artifacts include the experiment configuration, checkpoints, logs, and any
configured validation reports. Do not reuse an experiment directory with a
different class inventory, frontend, or architecture.

Pretrained waveform x-vector
----------------------------

``hyperion-train-wav2vec2xvector`` adds a Hugging Face frontend. Its model
configuration must identify the pretrained source and a compatible sample rate:

.. code-block:: bash

   hyperion-train-wav2vec2xvector hf_wav2vec2resnet1d \
     --cfg configs/wav2vec2xvector.yaml

Cache model assets before offline runs. The output checkpoint must be used with
the matching Wav2Vec2-family extractor.

Backend fitting
---------------

``hyperion-train-plda``, ``hyperion-train-verification-calibration``,
``hyperion-train-verification-greedy-fusion``, ``hyperion-train-qmf``, and
``hyperion-train-lgbe`` fit scoring-side components. For example, PLDA consumes
development embeddings and speaker labels:

.. code-block:: bash

   hyperion-train-plda \
     --segments-files data/dev/segments.csv \
     --feats-files exp/dev/xvectors.csv \
     --class-name speaker \
     --preproc-file exp/backend/preproc.h5 \
     --plda-file exp/backend/plda.h5

These artifacts are serialized backends, not neural checkpoints. Fit them only
on development data that is disjoint from labeled evaluation trials.

Other stable training commands
------------------------------

``hyperion-train-xvector-from-feats``, ``hyperion-train-xvector-from-wav``,
``hyperion-train-dino-wav2xvector``, and ``hyperion-make-wav2xvector`` use the
same configuration principles: CSV-aligned inputs, an explicit experiment path,
and a parser-selected model/trainer configuration. Inspect their exact parser
surface before creating a new configuration:

.. code-block:: bash

   hyperion-train-xvector-from-feats --help
   hyperion-train-dino-wav2xvector --help

See also
--------

* :doc:`../how-to/train-waveform-xvector`
* :doc:`../how-to/save-load-models-and-backends`
* :doc:`../torch-api-contracts`
