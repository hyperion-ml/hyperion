Hyperion Documentation
======================

Hyperion is a speech processing toolkit with NumPy and PyTorch stacks for
speaker recognition, anonymization/voice conversion, neural codecs, and
evaluation tooling.

This documentation reflects the current implementation in this repository,
including:

* ``hyperion.np``: NumPy models and utilities.
* ``hyperion.torch``: layers, neural architectures, models, trainers, and TPM wrappers.
* ``hyperion.io``: audio/feature readers and writers.
* ``hyperion.utils``: dataset manifests, trial tables, and Kaldi-style helpers.
* ``hyperion.data_prep``: dataset preparation classes.
* ``hyperion.text_norm``: text normalization utilities.
* ``hyperion.metrics``: high-level evaluators.
* ``hyperion.bin``: CLI entry points generated from scripts.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting-started
   quickstart
   optional-dependencies

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   architecture
   data-model
   glossary

.. toctree::
   :maxdepth: 2
   :caption: How-to guides

   how-to/train-waveform-xvector
   how-to/extract-score-xvectors
   how-to/prepare-data-and-vad
   how-to/run-resumable-distributed-training
   how-to/train-pretrained-wav2vec2-xvector
   how-to/use-configuration-files
   how-to/save-load-models-and-backends

.. toctree::
   :maxdepth: 2
   :caption: Documentation and maintenance

   building-documentation
   documentation-policy
   contributor-extension-guide
   model-extension-contracts
   torch-extension-workflows
   data-preparation-and-cli-extensions
   contributor-validation
   deprecation-and-compatibility
   public-surface
   api-contract-coverage

.. toctree::
   :maxdepth: 2
   :caption: Package reference

   torch
   torch-api
   torch-api-contracts
   torch-extension-points
   torch-layers-and-architectures
   torch-training-support
   torch-integrations-and-robustness
   experimental-components
   numpy
   numpy-extension-points
   np/speech_augmentation
   metrics
   io
   utils
   foundation-api-contracts
   statistical-api-contracts
   info_tables
   hyper_dataset
   trials
   data_prep
   text_norm
   cli

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
