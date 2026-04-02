Utility Layer
=============

Overview
--------

``hyperion.utils`` contains table abstractions, dataset manifests, trial data
structures, and Kaldi-style helper classes.

Dataset and table abstractions
------------------------------

See :doc:`info_tables` for a practical tutorial on manifest usage and how the
different ``InfoTable`` child classes relate to each other. See
:doc:`hyper_dataset` for a dataset-level tutorial covering how those manifests
are bundled and manipulated together.

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

See :doc:`trials` for a practical guide to ``TrialNdx``, ``TrialKey``,
``TrialScores``, and their sparse variants.

.. autoclass:: hyperion.utils.TrialNdx

.. autoclass:: hyperion.utils.TrialKey

.. autoclass:: hyperion.utils.TrialScores

.. autoclass:: hyperion.utils.SparseTrialNdx

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
