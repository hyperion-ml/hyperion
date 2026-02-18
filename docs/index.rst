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
   :caption: Contents

   getting-started
   architecture
   torch
   numpy
   metrics
   io
   utils
   data_prep
   text_norm
   cli

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
