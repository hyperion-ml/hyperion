Utils
=====

The ``hyperion.utils`` module contains several utility classes and functions.

Trial Management Classes
------------------------

These are a series of utils to handle Trial Indices, Keys and Scores. These are based on the MATLAB implementations in the BOSARIS Toolkit.

.. autoclass:: hyperion.utils.trial_key.TrialKey

.. autoclass:: hyperion.utils.trial_ndx.TrialNdx

.. autoclass:: hyperion.utils.trial_scores.TrialScores

.. autoclass:: hyperion.utils.trial_stats.TrialStats

.. autoclass:: hyperion.utils.sparse_trial_key.SparseTrialKey

.. autoclass:: hyperion.utils.sparse_trial_scores.SparseTrialScores

Kaldi Data Directory Manipulaton Classes
----------------------------------------

Thise are classes to manipulate Kaldi data directory files like ``wav.scp``, ``utt2spk``, ``segments``, ``rttm``.

.. autoclass:: hyperion.utils.scp_list.SCPList

.. autoclass:: hyperion.utils.utt2info.Utt2Info

.. autoclass:: hyperion.utils.segment_list.SegmentList

.. autoclass:: hyperion.utils.rttm.RTTM
	       

Kaldi Matrix Read/Write Classes
-------------------------------

These are classes to read/write text and binary matrices from ARK files.  They support the compression methods in Kaldi ARK files.

.. autoclass:: hyperion.utils.kaldi_matrix.KaldiMatrix
	       
.. autoclass:: hyperion.utils.kaldi_matrix.KaldiCompressedMatrix

Kaldi I/O Functions
-------------------------------

Utils to read/write binary ARK files

.. automodule:: hyperion.utils.kaldi_io_funcs

VAD Utils
---------------

Functions to manipulate VAD output, convert from binary to timestamps, intersect VADs, etc.

.. automodule:: hyperion.utils.vad_utils

Math Functions
--------------

.. automodule:: hyperion.utils.math

Miscellaneous Functions
-----------------------

.. automodule:: hyperion.utils.misc



		
	       
