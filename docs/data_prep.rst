Data Preparation
================

Overview
--------

``hyperion.data_prep`` contains dataset preparation classes used to convert
corpora into Hyperion-compatible metadata layouts.

The package includes:

* a shared base class for data preparation flows
* dataset-specific preparation classes (ASVspoof, LibriSpeech-family, VoxCeleb,
  SRE, and others)

Package exports
---------------

.. automodule:: hyperion.data_prep
   :members:

Base API
--------

.. autoclass:: hyperion.data_prep.data_prep.DataPrep
   :members:

Notes
-----

Each dataset prep class usually defines dataset-specific parsing, metadata
construction, and output writing logic, while following the common interface
from ``DataPrep``.
