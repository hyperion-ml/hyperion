Utility Layer
=============

Overview
--------

``hyperion.utils`` contains table abstractions, dataset manifests, trial data
structures, and Kaldi-style helper classes.

Dataset and table abstractions
------------------------------

.. autoclass:: hyperion.utils.InfoTable

.. autoclass:: hyperion.utils.HyperDataset

Domain-specific table sets built on top of ``InfoTable``:

.. autoclass:: hyperion.utils.SegmentSet

.. autoclass:: hyperion.utils.RecordingSet

.. autoclass:: hyperion.utils.FeatureSet

.. autoclass:: hyperion.utils.VADSet

.. autoclass:: hyperion.utils.ImageSet

.. autoclass:: hyperion.utils.VideoSet

.. autoclass:: hyperion.utils.DiarizationSet

Trial/key/score structures
--------------------------

.. autoclass:: hyperion.utils.TrialNdx

.. autoclass:: hyperion.utils.TrialKey

.. autoclass:: hyperion.utils.TrialScores

.. autoclass:: hyperion.utils.SparseTrialKey

.. autoclass:: hyperion.utils.SparseTrialScores

Enrollment and class metadata
-----------------------------

.. autoclass:: hyperion.utils.EnrollmentMap

.. autoclass:: hyperion.utils.ClassInfo

Kaldi-style helper structures
-----------------------------

.. autoclass:: hyperion.utils.SCPList

.. autoclass:: hyperion.utils.Utt2Info

.. autoclass:: hyperion.utils.SegmentList

.. autoclass:: hyperion.utils.RTTM

.. autoclass:: hyperion.utils.KaldiMatrix

.. autoclass:: hyperion.utils.KaldiCompressedMatrix

Miscellaneous utilities
-----------------------

.. automodule:: hyperion.utils.misc
   :members:
