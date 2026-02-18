IO Utilities
============

Overview
--------

``hyperion.io`` provides unified readers/writers for features, audio, VAD, and
packed data formats used across scripts and recipes.

Factories and specifiers
------------------------

Reader/writer creation is centralized through factories and Kaldi-style
specifier parsing.

.. automodule:: hyperion.io.data_rw_factory
   :members:

.. automodule:: hyperion.io.rw_specifiers
   :members:

Feature readers and writers
---------------------------

ARK and HDF5 readers/writers:

.. automodule:: hyperion.io.ark_data_reader
   :members:

.. automodule:: hyperion.io.ark_data_writer
   :members:

.. automodule:: hyperion.io.h5_data_reader
   :members:

.. automodule:: hyperion.io.h5_data_writer
   :members:

Audio IO
--------

.. automodule:: hyperion.io.audio_reader
   :members:

.. automodule:: hyperion.io.audio_writer
   :members:

Packed audio IO
---------------

.. automodule:: hyperion.io.packed_audio_reader
   :members:

.. automodule:: hyperion.io.packed_audio_writer
   :members:

VAD IO
------

.. automodule:: hyperion.io.vad_rw_factory
   :members:

.. automodule:: hyperion.io.bin_vad_reader
   :members:

.. automodule:: hyperion.io.segment_vad_reader
   :members:

.. automodule:: hyperion.io.table_vad_reader
   :members:

Legacy compatibility classes
----------------------------

Some legacy compatibility classes remain available:

.. autoclass:: hyperion.io.HypDataReader

.. autoclass:: hyperion.io.HypDataWriter
