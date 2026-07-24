Hyperion Data Model
===================

Hyperion uses a small set of aligned tables to represent speech data and
speaker-verification evaluation. Understanding these tables is more important
than memorizing a particular command: the same model works for NumPy APIs,
PyTorch workflows, and command-line tools.

The three levels of data
------------------------

Media and segment manifests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RecordingSet`` describes source media. ``SegmentSet`` describes the speech
segments, utterances, or clips used by a workflow. Other manifests attach data
to those segment identifiers, including ``FeatureSet``, ``VADSet``, and
``DiarizationSet``.

Dataset container
~~~~~~~~~~~~~~~~~

``HyperDataset`` groups related manifests, class metadata, enrollment maps,
and trial information. It is useful when a workflow needs to load, validate,
filter, or save a complete dataset consistently.

Verification evaluation
~~~~~~~~~~~~~~~~~~~~~~~

Verification uses a matrix convention:

* rows are enrollment model identifiers;
* columns are test segment identifiers;
* each cell is one ``(model, segment)`` trial.

``TrialNdx`` marks trials to score. ``TrialKey`` adds the target/non-target
ground truth. ``TrialScores`` stores system scores on the same grid. This
alignment is essential: score matrices must use the same model and segment
order as their key or index.

Typical flow
------------

.. code-block:: text

   recordings -> segments -> features / VAD / labels
                              |
                              v
                       embedding extraction
                              |
   enrollment map -> trial index/key -> scores -> metrics

An embedding system may supply the vectors from precomputed features or raw
audio. The evaluation objects are independent of how those vectors were
created.

Detailed guides
---------------

* :doc:`info_tables` explains the manifest classes and their common operations.
* :doc:`hyper_dataset` explains dataset-level composition and consistency.
* :doc:`trials` explains trial keys, score tables, sparse variants, and file
  formats.
* :doc:`glossary` defines the terminology used across these pages.
