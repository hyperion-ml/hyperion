Foundation API Contracts
========================

This page defines the behavioral contracts for Hyperion's foundational I/O and
table APIs. It complements the generated signatures in :doc:`io` and
:doc:`utils`: use those pages to discover members, and use this page before
writing an integration or extension.

I/O contracts
-------------

Feature archive factories
~~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`hyperion.io.DataWriterFactory`,
:class:`hyperion.io.SequentialDataReaderFactory`, and
:class:`hyperion.io.RandomAccessDataReaderFactory` rather than choosing a
concrete Ark/HDF5 implementation in application code.

``create(specifier, ...)``
  Parses an Ark/HDF5 read or write specifier and returns the corresponding
  concrete reader or writer. Write specifiers may pair an archive with a CSV
  index, for example ``ark,csv:exp/xvectors.ark,exp/xvectors.csv``. An invalid
  archive type or specifier raises :class:`ValueError`.

``filter_args(...)``
  Returns only the keyword arguments accepted by the selected factory. It does
  not create files and is intended for ``jsonargparse`` configuration plumbing.

The output index is CSV for new workflows. Its ``id`` column is the stable key
used by manifests, enrollment maps, trials, and random-access lookup. Ark/HDF5
archives preserve numeric arrays; the CSV stores locations and optional
metadata, not the array payload itself.

``DataWriter`` and concrete writers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`hyperion.io.DataWriter` is the base writer contract, implemented by
:class:`hyperion.io.ArkDataWriter` and :class:`hyperion.io.H5DataWriter`.

``write(keys, data, metadata=None)``
  Writes one array per key. ``keys`` may be a string or a sequence of strings.
  A single NumPy array is accepted for a single key; batched arrays or a list
  are required for multiple keys. The writer rejects a key/data length mismatch
  and unsupported array dimensionality with :class:`ValueError`.

``metadata``
  Is optional row-aligned metadata. When a CSV index is requested, declared
  ``metadata_columns`` are written alongside each key. Metadata never changes
  the stored feature values.

``compression_method``
  Affects Ark/HDF5 representation only when ``compress=True``. It must be a
  supported Kaldi compression method; callers should not assume compressed
  arrays can be memory-mapped as ordinary NumPy arrays.

Writers create parent output directories, own the open archive/index handles,
and must be closed. A ``with`` block closes them even when an exception is
raised. Writing mutates files immediately; ``flush`` makes buffered output
visible without closing the writer.

``DataReader`` and access modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sequential readers preserve archive order and return keys paired with arrays.
They are appropriate for conversion and bulk processing. Random-access readers
select values by key and are appropriate when a ``TrialNdx`` or ``TrialKey``
determines the required embeddings. A missing requested key is an input
alignment error and should be handled before scoring.

Reader shape helpers report archive dimensions without changing data. Returned
features are NumPy vectors or matrices; a matrix conventionally has shape
``(num_frames, feature_dim)`` and an embedding has shape ``(embedding_dim,)``.
Reader/writer methods do not normalize, length-normalize, or otherwise alter
features unless their concrete format requires decoding compressed storage.

Audio and VAD
~~~~~~~~~~~~~

:class:`hyperion.io.SequentialAudioReader` streams manifest entries in order;
:class:`hyperion.io.RandomAccessAudioReader` selects recordings or segments by
id. Audio arrays use channel-first layout ``(num_channels, num_samples)`` when
multichannel data is retained. Segment requests crop the recording according
to manifest timing; callers are responsible for valid storage paths and timing.

:class:`hyperion.io.VADReaderFactory` creates either binary-frame or
table-based VAD readers. Binary VAD is meaningful only with its associated
``frame_length``, ``frame_shift``, and ``snip_edges`` convention. Table VAD
returns time marks. Converting between them is lossy at frame boundaries; use
the same convention at extraction and evaluation time.

Table and trial contracts
-------------------------

``InfoTable`` and manifest subclasses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`hyperion.utils.InfoTable` owns a pandas DataFrame with a unique ``id``
column. ``SegmentSet``, ``RecordingSet``, ``FeatureSet``, ``VADSet``, and
related subclasses add schema-specific validation and helpers, but retain the
same table operations.

``load(path, sep=None)`` and ``save(path, sep=None)``
  Load or write CSV/TSV-like tables based on the path suffix or explicit
  separator. Save is a file mutation and can overwrite an existing target.

``filter(items, ...)`` and ``split(idx, num_parts, group_by=None)``
  Return a table of the same concrete type. Filtering and splitting do not
  mutate the original table unless an API explicitly states otherwise. Splits
  are one-based at the public interface and validate the requested part.

Subclasses require their identifying columns: a recording table describes
storage, a segment table references recordings and timing, and feature/VAD
tables bind ids to stored artifacts. CSV ids are strings even when they look
numeric; do not coerce them before joins.

``HyperDataset``
~~~~~~~~~~~~~~~~

:class:`hyperion.utils.HyperDataset` is the multi-table dataset contract. It
holds recordings, segments, class vocabularies, enrollment maps, and related
tables as one coherent unit.

``load(...)`` loads declared dataset tables and checks relationships where
metadata is available. ``filter_by_segments(...)`` and
``filter_by_classes(...)`` produce relationship-consistent subsets and may
remove dependent rows. ``save(...)``, ``save_changed(...)``, and ``save_all``
write dataset metadata; they mutate the destination directory, not in-memory
audio or features.

Dense trial tables
~~~~~~~~~~~~~~~~~~

The dense trial classes share two ordered axes: models and test segments.

``TrialNdx``
  Defines pairs to score through a boolean ``trial_mask`` of shape
  ``(num_models, num_segments)``.

``TrialKey``
  Adds target/non-target labels for the same grid. Its masks must have the same
  shape and key order as the model and segment id arrays.

``TrialScores``
  Stores a float score matrix and a ``score_mask`` of the same shape. A false
  mask entry means no score is available; it is not equivalent to a zero score.

``validate()`` checks axis lengths, mask shapes, and mutually compatible labels
and raises :class:`ValueError` for inconsistent state. ``align_with_ndx(ndx)``
reorders or restricts a trial object to the supplied index; use it before
metrics rather than relying on coincidental row/column order. ``filter(...)``
and ``split(...)`` return new restricted/sharded objects. ``set_missing_to_value``
mutates ``TrialScores`` and can materially change EER/DCF.

Dense classes load/save CSV-like or HDF5 representations. HDF5 is appropriate
for dense matrices; text/table formats are useful for inspection.

Sparse trial tables
~~~~~~~~~~~~~~~~~~~

``SparseTrialNdx``, ``SparseTrialKey``, and ``SparseTrialScores`` represent
only observed pairs rather than allocating a full model-by-segment matrix.
Use them when the trial graph is sparse. Their ids and pair rows must still be
unique and aligned; sparse representation does not relax label or score
availability requirements.

Focused example
---------------

.. code-block:: python

   from hyperion.utils import TrialKey, TrialScores
   from hyperion.np.metrics import compute_eer

   key = TrialKey.load("data/dev/key.csv")
   scores = TrialScores.load("exp/dev/scores.h5")
   scores = scores.align_with_ndx(key)
   tar, non = scores.get_tar_non(key)
   eer, threshold = compute_eer(tar, non)

See :doc:`data-model`, :doc:`how-to/extract-score-xvectors`, and :doc:`trials`
for file layouts and an end-to-end scoring workflow.
