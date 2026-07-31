Utility Layer
=============

Overview
--------

``hyperion.utils`` defines the package's identifier and table contracts:
dataset manifests, enrollment mappings, trial keys, and score containers.
These objects keep rows and columns aligned across preparation, extraction,
scoring, and evaluation.

CSV/TSV tables are the maintained interchange format for manifests and indexes.
Kaldi-style list helpers remain available for compatibility, but new workflows
should use the table classes and CSV files.

Alignment rules
---------------

``InfoTable``-derived manifests use an ``id`` column as their stable key.
Verification tables use enrollment model ids for rows and test segment ids for
columns. ``TrialScores`` must use the same ordered model/segment sets as the
``TrialKey`` or ``TrialNdx`` it is evaluated against.

See :doc:`data-model` for the complete relationship between manifests,
enrollment maps, trials, and scores.

Dataset and table abstractions
------------------------------

See :doc:`info_tables` for a practical tutorial on manifest usage and how the
different ``InfoTable`` child classes relate to each other. See
:doc:`hyper_dataset` for a dataset-level tutorial covering how those manifests
are bundled and manipulated together.

.. autoclass:: hyperion.utils.InfoTable
   :no-index:
   :members: load, save, filter, split, cat

.. autoclass:: hyperion.utils.HyperDataset
   :no-index:
   :members: load, save, clean, describe

Domain-specific table sets built on top of ``InfoTable``:

.. autoclass:: hyperion.utils.SegmentSet
   :no-index:

.. autoclass:: hyperion.utils.RecordingSet
   :no-index:

.. autoclass:: hyperion.utils.FeatureSet
   :no-index:

.. autoclass:: hyperion.utils.VADSet
   :no-index:

.. autoclass:: hyperion.utils.ImageSet
   :no-index:

.. autoclass:: hyperion.utils.VideoSet
   :no-index:

.. autoclass:: hyperion.utils.DiarizationSet
   :no-index:

Trial/key/score structures
--------------------------

See :doc:`trials` for a practical guide to ``TrialNdx``, ``TrialKey``,
``TrialScores``, and their sparse variants.

.. autoclass:: hyperion.utils.TrialNdx
   :no-index:
   :members: load, save, filter, split

.. autoclass:: hyperion.utils.TrialKey
   :no-index:
   :members: load, save, to_ndx, filter, split

.. autoclass:: hyperion.utils.TrialScores
   :no-index:
   :members: load, save, align_with_ndx, get_tar_non, filter, split

.. autoclass:: hyperion.utils.SparseTrialNdx
   :no-index:

.. autoclass:: hyperion.utils.SparseTrialKey
   :no-index:

.. autoclass:: hyperion.utils.SparseTrialScores
   :no-index:

Enrollment and class metadata
-----------------------------

.. autoclass:: hyperion.utils.EnrollmentMap
   :no-index:

.. autoclass:: hyperion.utils.ClassInfo
   :no-index:

Compatibility helper structures
-------------------------------

These types support legacy Kaldi/BOSARIS-style data interchange. Prefer
``RecordingSet``, ``FeatureSet``, ``VADSet``, and CSV-indexed archives in new
package code.

.. autoclass:: hyperion.utils.SCPList
   :no-index:

.. autoclass:: hyperion.utils.Utt2Info
   :no-index:

.. autoclass:: hyperion.utils.SegmentList
   :no-index:

.. autoclass:: hyperion.utils.RTTM
   :no-index:

.. autoclass:: hyperion.utils.KaldiMatrix
   :no-index:

.. autoclass:: hyperion.utils.KaldiCompressedMatrix
   :no-index:

Miscellaneous utilities
-----------------------

``hyperion.utils.misc`` contains low-level convenience helpers used internally
by package subsystems. It is not a supported extension namespace; import a
named utility only when another documented public API requires it. The
supported table, trial, dataset, and Kaldi-style contracts are documented on
:doc:`foundation-api-contracts`.
