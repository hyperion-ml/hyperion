Working With HyperDataset
=========================

.. currentmodule:: hyperion.utils

Overview
--------

``HyperDataset`` is the dataset-level container that ties together the manifest
classes described in :doc:`info_tables` with the evaluation structures
described in :doc:`trials`.

At the center of every ``HyperDataset`` is a :class:`SegmentSet`. Everything
else is optional and is attached to those segments by one of a few standard
alignment rules:

* :class:`RecordingSet`, :class:`ImageSet`, and :class:`VideoSet` describe the
  physical source media for segments.
* :class:`ClassInfo` tables describe label vocabularies such as speaker,
  gender, language, or accent.
* :class:`FeatureSet`, :class:`VADSet`, and :class:`DiarizationSet` attach
  per-segment artifacts by segment id.
* :class:`EnrollmentMap` and :class:`TrialKey` / :class:`TrialNdx` attach
  speaker-recognition evaluation data to the dataset.

Use ``HyperDataset`` when you want to manage those pieces as one coherent
dataset instead of passing many independent tables around.

This tutorial focuses on:

* how the different tables fit together
* how to build datasets from existing manifests or from in-memory objects
* how lazy loading and saving work
* how to filter, clean, split, and transform datasets safely
* how to attach enrollment and trial tables for evaluation

What ``HyperDataset`` Adds On Top Of ``InfoTable``
--------------------------------------------------

``InfoTable`` subclasses are individual manifests. ``HyperDataset`` is the
orchestration layer above them.

``HyperDataset`` provides:

* one required anchor table: ``segments``
* optional registration of recordings, images, videos, features, VADs,
  diarizations, classes, enrollments, and trials
* lazy loading from file paths or from a dataset YAML manifest
* consistency cleanup across related tables with :meth:`HyperDataset.clean`
* higher-level operations such as train/validation splitting, fold creation,
  trial/cohort generation, subsegment sampling, and segment concatenation

Conceptually, the flow looks like this:

.. code-block:: text

   SegmentSet
      |
      +-- recording/image/video references -> RecordingSet / ImageSet / VideoSet
      +-- label columns such as speaker/gender/language -> ClassInfo tables
      +-- segment ids -> FeatureSet / VADSet / DiarizationSet
      +-- segment ids used by EnrollmentMap / TrialKey / TrialNdx

Core Alignment Rules
--------------------

These are the conventions that make the container work predictably.

Segments
~~~~~~~~

The :class:`SegmentSet` is always required. It defines the dataset rows that
most other tables depend on.

Typical segment columns include:

* ``id``
* ``recording``, ``start``, ``duration``
* labels such as ``speaker``, ``gender``, ``language``
* transcript or provenance fields

Recordings, images, and videos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are media-level manifests:

* :class:`RecordingSet` is aligned using the segment ``recording`` column
* :class:`ImageSet` is aligned using the segment ``image`` column
* :class:`VideoSet` is aligned using the segment ``video`` column

If a segment table does not contain ``recording`` / ``image`` / ``video``,
the corresponding helper methods in :class:`SegmentSet` fall back to the
segment ``id``.

Features, VADs, and diarizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are keyed directly by segment id. In practice that means:

* the ``id`` column of the feature/VAD/diarization table should match segment ids
* if you change segment ids, those tables usually need to be rebuilt or removed

Classes
~~~~~~~

Each :class:`ClassInfo` table is usually associated with one segment column of
the same name.

Examples:

* segment column ``speaker`` -> class table ``speaker``
* segment column ``gender`` -> class table ``gender``
* segment column ``language`` -> class table ``language``

The segment table stores the class id used by each row. The corresponding
``ClassInfo`` table stores the class inventory and optional metadata such as
``class_idx`` or ``weights``.

Enrollments and trials
~~~~~~~~~~~~~~~~~~~~~~

Evaluation metadata is attached separately:

* :class:`EnrollmentMap` maps enrollment model ids to segment ids
* :class:`TrialNdx` defines which ``(model, segment)`` trials exist
* :class:`TrialKey` adds the trial ground truth

See :doc:`trials` for the full semantics of those classes. The important point
here is that ``HyperDataset`` can store them together with the manifests that
define the actual segments being evaluated.

Minimal End-To-End Example
--------------------------

The following example builds a small in-memory dataset with segments,
recordings, a class table, one feature table, one VAD table, and a trial setup.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from hyperion.utils import (
       ClassInfo,
       EnrollmentMap,
       FeatureSet,
       HyperDataset,
       RecordingSet,
       SegmentSet,
       TrialKey,
       VADSet,
   )

   segments = SegmentSet(
       pd.DataFrame(
           {
               "id": ["utt1", "utt2", "utt3"],
               "recording": ["rec1", "rec1", "rec2"],
               "speaker": ["spk1", "spk1", "spk2"],
               "gender": ["f", "f", "m"],
               "duration": [1.8, 2.1, 1.3],
           }
       )
   )

   recordings = RecordingSet(
       pd.DataFrame(
           {
               "id": ["rec1", "rec2"],
               "storage_path": ["audio/rec1.wav", "audio/rec2.wav"],
               "duration": [3.9, 1.3],
               "sample_freq": [16000, 16000],
           }
       )
   )

   speaker_info = ClassInfo(pd.DataFrame({"id": ["spk1", "spk2"]}))
   speaker_info.add_class_idx()

   features = FeatureSet(
       pd.DataFrame(
           {
               "id": ["utt1", "utt2", "utt3"],
               "storage_path": ["feats/utt1.ark:10", "feats/utt2.ark:20", "feats/utt3.ark:30"],
           }
       )
   )

   vads = VADSet(
       pd.DataFrame(
           {
               "id": ["utt1", "utt2", "utt3"],
               "storage_path": ["vad/utt1.ark:5", "vad/utt2.ark:7", "vad/utt3.ark:9"],
           }
       )
   )

   enrollments = EnrollmentMap(
       pd.DataFrame(
           {
               "id": ["spk1", "spk2"],
               "segmentid": ["utt1", "utt3"],
           }
       )
   )

   trials = TrialKey(
       model_set=["spk1", "spk2"],
       seg_set=["utt1", "utt2", "utt3"],
       tar=np.array(
           [
               [True, True, False],
               [False, False, True],
           ],
           dtype=bool,
       ),
       non=np.array(
           [
               [False, False, True],
               [True, True, False],
           ],
           dtype=bool,
       ),
   )

   dataset = HyperDataset(
       segments=segments,
       recordings=recordings,
       classes={"speaker": speaker_info},
       features={"mfcc": features},
       vads={"speech": vads},
       enrollments={"eval": enrollments},
       trials={"eval": trials},
   )

   summary = dataset.describe()
   print(summary["msg"])
   print(sorted(dataset.classes_keys()))
   print(sorted(dataset.features_keys()))

This pattern is typical: start from ``SegmentSet`` and then attach auxiliary
manifests that describe the same corpus from different angles.

Building A Dataset
------------------

From existing manifest paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can construct a dataset directly from file paths. This is often the
cleanest way to represent a prepared corpus on disk.

.. code-block:: python

   from hyperion.utils import HyperDataset

   dataset = HyperDataset(
       segments="data/train/segments.csv",
       recordings="data/train/recordings.csv",
       classes={
           "speaker": "data/train/speaker.csv",
           "gender": "data/train/gender.csv",
       },
       features={
           "mfcc": "data/train/mfcc.scp",
           "fbank": "data/train/fbank.scp",
       },
       vads={"speech": "data/train/vad.scp"},
       diarizations={"oracle": "data/train/diarization.csv"},
   )

At construction time, ``HyperDataset`` stores those paths and loads the actual
tables only when you access them.

From already loaded manifest objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the tables are already in memory, pass the objects directly.

.. code-block:: python

   from hyperion.utils import HyperDataset

   dataset = HyperDataset(
       segments=segments,
       recordings=recordings,
       classes={"speaker": speaker_info},
   )

From a ``SegmentSet`` with automatic class creation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :meth:`HyperDataset.from_segments` when you already have a segment table and
want a dataset quickly.

.. code-block:: python

   from hyperion.utils import HyperDataset

   dataset = HyperDataset.from_segments(
       segments="data/train/segments.csv",
       recordings="data/train/recordings.csv",
       class_names=["speaker", "gender", "language"],
   )

This does two useful things:

* if ``recordings`` is provided and the segments table has no ``duration``,
  durations are copied from the recordings table
* for each name in ``class_names``, a ``ClassInfo`` table is created from the
  unique non-missing values in the corresponding segment column

From a ``RecordingSet`` when no segmentation exists
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When your corpus is still recording-level, use
:meth:`HyperDataset.from_recordings`.

.. code-block:: python

   dataset = HyperDataset.from_recordings("data/raw/recordings.csv")

This creates a segment table whose rows mirror the recordings table.

From Lhotse or Kaldi-style inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``HyperDataset`` also provides import helpers for other ecosystems:

* :meth:`HyperDataset.from_lhotse`
* :meth:`HyperDataset.from_kaldi`

Typical usage:

.. code-block:: python

   dataset = HyperDataset.from_lhotse(cuts="cuts.jsonl.gz")

.. code-block:: python

   dataset = HyperDataset.from_kaldi("data/kaldi_train")

Those helpers are useful when your recipe already produces Lhotse manifests or
Kaldi-style ``wav.scp``, ``segments``, ``utt2spk``, ``vad.scp``, or feature SCPs.

Lazy Loading And Access Patterns
--------------------------------

Lazy loading is one of the main reasons to use ``HyperDataset`` for large
corpora.

Single-table access
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   segments = dataset.segments()
   recordings = dataset.recordings()
   speaker_info = dataset.classes_value("speaker")
   mfcc = dataset.features_value("mfcc")
   speech_vad = dataset.vads_value("speech")
   eval_trials = dataset.trials_value("eval")

The first call loads the table if necessary. Subsequent calls reuse the cached
object unless you ask otherwise.

``keep_loaded=False``
~~~~~~~~~~~~~~~~~~~~~

If you only want a temporary object and do not want to cache it on the
dataset, use ``keep_loaded=False``.

.. code-block:: python

   speaker_info = dataset.classes_value("speaker", keep_loaded=False)

This is useful when the dataset is only a manifest registry and you want to
avoid keeping many large tables in memory at once.

Iterating over keyed manifest families
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For keyed collections such as features, VADs, classes, enrollments, and
trials, iterate by name:

.. code-block:: python

   for name, feats in dataset.features():
       print(name, len(feats))

   for name, class_info in dataset.classes():
       print(name, len(class_info))

   for name, trial_data in dataset.trials():
       print(name, type(trial_data).__name__)

The corresponding ``*_keys()`` helpers return only the registered names.

.. code-block:: python

   print(sorted(dataset.features_keys()))
   print(sorted(dataset.vads_keys()))
   print(sorted(dataset.classes_keys()))

Presence checks
~~~~~~~~~~~~~~~

For the single-table media manifests, use:

.. code-block:: python

   if dataset.has_recordings:
       print("recordings available")

   if dataset.has_images:
       print("images available")

   if dataset.has_videos:
       print("videos available")

Saving And Loading Dataset Bundles
----------------------------------

Dataset YAML layout
~~~~~~~~~~~~~~~~~~~

When you save a dataset, Hyperion writes the individual manifests and a
``dataset.yaml`` file that points to them.

A typical saved YAML bundle looks like this:

.. code-block:: yaml

   segments: segments.csv
   recordings: recordings.csv
   classes:
     speaker: speaker.csv
     gender: gender.csv
   features:
     mfcc: mfcc.csv
   vads:
     speech: speech.csv
   enrollments:
     eval: enrollment.csv
   trials:
     eval: trials.csv

Saving
~~~~~~

.. code-block:: python

   dataset.save("exp/my_dataset", force_save_all=True)

By default, :meth:`HyperDataset.save` delegates to
:meth:`HyperDataset.save_changed`, which saves only the manifests that are
loaded, modified, or missing from the target location. If you want a complete
bundle regardless of change tracking, pass ``force_save_all=True`` or call
``save_all`` directly.

You can also control the table separator:

.. code-block:: python

   dataset.save("exp/my_dataset_tsv", table_sep="\t", force_save_all=True)

Trial manifests can use a different separator through ``trials_sep`` when
needed.

Loading
~~~~~~~

.. code-block:: python

   from hyperion.utils import HyperDataset

   ds1 = HyperDataset.load("exp/my_dataset", lazy=True)
   ds2 = HyperDataset.load("exp/my_dataset/dataset.yaml", lazy=True)

Both forms are supported: pass either the dataset directory or the YAML file.

If you want trial files to be loaded through :class:`SparseTrialKey` when
possible, use:

.. code-block:: python

   ds_sparse = HyperDataset.load("exp/my_dataset", sparse_trials=True)
   trials = ds_sparse.trials_value("eval")

Registering, Replacing, And Removing Tables
-------------------------------------------

``HyperDataset`` lets you attach or replace manifests after construction.

Adding or replacing tables
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   dataset.set_recordings(recordings)
   dataset.add_features("mfcc", features)
   dataset.add_vads("speech", vads)
   dataset.add_diarizations("oracle", "data/train/diarization.csv")
   dataset.add_classes("speaker", speaker_info)
   dataset.add_enrollments("eval", enrollments)
   dataset.add_trials("eval", trials)

``set_recordings``, ``set_images``, and ``set_videos`` manage the single
media-level manifests. The ``add_*`` methods manage keyed collections.

Removing tables
~~~~~~~~~~~~~~~

The remove methods work in two modes:

* pass a name to remove a single keyed table
* call them without a name to remove all keyed tables of that type

.. code-block:: python

   dataset.remove_features("mfcc")
   dataset.remove_vads()
   dataset.remove_diarizations()
   dataset.remove_classes("gender")
   dataset.remove_enrollments()
   dataset.remove_trials()

For recordings, images, and videos there is only one table of each kind:

.. code-block:: python

   dataset.remove_recordings()
   dataset.remove_images()
   dataset.remove_videos()

Working With Classes
--------------------

Creating ``ClassInfo`` from segment columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is one of the most common workflows:

.. code-block:: python

   dataset.add_classes_from_segments(["speaker", "gender", "language"])

For each listed column, ``HyperDataset``:

* reads the unique non-missing values from the segments table
* creates a :class:`ClassInfo` table with those values as ``id``
* registers that table under the same name as the segment column

If you want integer indices after building the table, rebuild them explicitly:

.. code-block:: python

   dataset.rebuild_class_idx("speaker")

Joining additional columns into the segments table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :meth:`HyperDataset.add_cols_to_segments` to enrich the segment table from
another registered table.

For example, copy ``sample_freq`` from the recordings table into the segments
table using the segment ``recording`` column:

.. code-block:: python

   dataset.add_cols_to_segments(
       right_table="recordings",
       column_names=["sample_freq"],
       on="recording",
       right_on="id",
   )

You can also create class info immediately for the newly added columns:

.. code-block:: python

   dataset.add_cols_to_segments(
       right_table="recordings",
       column_names=["source_type"],
       on="recording",
       right_on="id",
       create_class_info=True,
   )

That is useful when a recording-level attribute becomes a classification target
at segment level.

Keeping The Dataset Consistent
------------------------------

The role of ``clean()``
~~~~~~~~~~~~~~~~~~~~~~~

After manual edits to the segment table, auxiliary tables may contain orphaned
rows. :meth:`HyperDataset.clean` prunes them.

Examples of what ``clean()`` does:

* removes recordings/images/videos no longer referenced by segments
* removes feature/VAD/diarization rows whose segment ids disappeared
* trims ``ClassInfo`` tables to the classes still present in segments
* trims enrollments and trials to the surviving segment and model ids

Typical pattern:

.. code-block:: python

   dataset.set_segments(dataset.segments().filter(predicate="duration >= 2.0"))
   dataset.clean(rebuild_class_idx=True)

Many high-level filtering helpers already call ``clean()`` internally. You
mainly need it when you directly replace or mutate tables yourself.

Common Filtering And Curation Operations
----------------------------------------

By segment ids or predicates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   dataset.filter_by_segments(["utt1", "utt3"])
   dataset.filter_by_segments_predicate("duration >= 2.0")

By classes
~~~~~~~~~~

.. code-block:: python

   dataset.filter_by_classes(
       class_name="speaker",
       classes=["spk1", "spk2", "spk5"],
       remove_na=True,
       rebuild_idx=True,
   )

This keeps only the segments whose ``speaker`` value is one of those ids and
then cleans dependent tables.

Filtering classes and enrollments together
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For evaluation subsets, use
:meth:`HyperDataset.filter_by_classes_and_enrollments` so that class filtering,
enrollment filtering, and trial filtering stay synchronized.

.. code-block:: python

   eval_enroll = dataset.enrollments_value("eval")

   dataset.filter_by_classes_and_enrollments(
       class_name="speaker",
       classes=["spk1", "spk2"],
       enrollment_name="eval",
       enrollments=eval_enroll,
       remove_na=True,
       rebuild_idx=True,
   )

Removing short segments or underrepresented classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   dataset.remove_short_segments(min_length=1.5)

   dataset.remove_classes_few_segments(
       class_name="speaker",
       min_segs=5,
       rebuild_idx=True,
   )

   dataset.remove_classes_few_toomany_segments(
       class_name="speaker",
       min_segs=5,
       max_segs=500,
       rebuild_idx=True,
   )

Splitting Datasets
------------------

Train/validation split
~~~~~~~~~~~~~~~~~~~~~~

Basic random split:

.. code-block:: python

   train_ds, val_ds = dataset.split_train_val(val_prob=0.1, seed=1234)

Keep each joint label combination in both splits:

.. code-block:: python

   train_ds, val_ds = dataset.split_train_val(
       val_prob=0.1,
       joint_classes=["speaker", "gender"],
       min_train_samples=1,
       seed=1234,
   )

Force specific classes to be disjoint across train and validation:

.. code-block:: python

   train_ds, val_ds = dataset.split_train_val(
       val_prob=0.1,
       disjoint_classes=["speaker"],
       seed=1234,
   )

Current limitation: passing both ``joint_classes`` and ``disjoint_classes`` to
``split_train_val`` is not implemented in the current code.

Cross-validation folds
~~~~~~~~~~~~~~~~~~~~~~

Use :meth:`HyperDataset.split_folds` to generate parallel train/test fold
datasets.

.. code-block:: python

   train_folds, test_folds = dataset.split_folds(
       num_folds=5,
       joint_classes=["speaker"],
       seed=1234,
   )

Each returned element is itself a ``HyperDataset`` whose dependent manifests
have already been cleaned.

Working With Enrollments And Trials
-----------------------------------

Manual registration
~~~~~~~~~~~~~~~~~~~

If you already have enrollment and trial manifests, attach them directly.

.. code-block:: python

   dataset.add_enrollments("eval", "data/eval/enrollment.csv")
   dataset.add_trials("eval", "data/eval/trials.csv")

Later:

.. code-block:: python

   enroll = dataset.enrollments_value("eval")
   trial_data = dataset.trials_value("eval")

``trial_data`` may be:

* :class:`TrialKey`
* :class:`TrialNdx`
* :class:`SparseTrialKey`

depending on what you stored and how you loaded the dataset.

Generating a trial/cohort split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``HyperDataset`` includes a convenience method for creating a trial subset and
a cohort subset for QMF-style workflows:

.. code-block:: python

   dataset_trials, dataset_cohort = dataset.split_into_trials_and_cohort(
       num_1k_tar_trials=10,
       num_trial_speakers=200,
       intra_gender=True,
       trials_name="qmf_trials",
       seed=1234,
   )

The first returned dataset contains:

* a restricted segment set
* an ``EnrollmentMap`` stored under ``"enrollments"``
* a trials entry stored under the name passed in ``trials_name``

The second returned dataset contains the cohort segments.

Transforming A Dataset
----------------------

Sampling random subsegments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :meth:`HyperDataset.sample_random_subsegments` when you want to create new
segment rows by sampling shorter windows from existing ones.

.. code-block:: python

   sub_ds = dataset.sample_random_subsegments(
       subsegments_per_segment=2,
       min_duration=1.0,
       max_duration=2.0,
       seg_suffix="sub",
       inplace=False,
   )

Important side effect: because the segment ids no longer correspond to the
original feature/VAD/diarization manifests, those tables are removed from the
new dataset. If ids change, enrollments and trials are removed as well.

Concatenating segments
~~~~~~~~~~~~~~~~~~~~~~

Use :meth:`HyperDataset.cat_segments` to concatenate adjacent or grouped
segments into longer examples.

.. code-block:: python

   cat_ds = dataset.cat_segments(
       group_by=["speaker"],
       max_duration=30.0,
       inplace=False,
   )

This operation has stricter requirements:

* the dataset must have a recordings table
* recording ``storage_path`` entries must not already be pipe commands
* recording ``sample_freq`` must be available
* if the segments table contains ``start``, all starts must be ``0``

Because concatenation creates new segment ids and new synthetic recordings, the
operation removes features, VADs, diarizations, enrollments, and trials from
the transformed dataset.

Merging datasets
~~~~~~~~~~~~~~~~

Use :meth:`HyperDataset.merge` to concatenate several datasets that belong to
the same general schema.

.. code-block:: python

   merged = HyperDataset.merge([dataset_a, dataset_b, dataset_c])

Current behavior:

* segments, classes, recordings, images, videos, features, VADs, and
  diarizations are merged when present
* enrollments and trials are not merged by the current implementation

Best Practices
--------------

1. Treat ``segments`` as the authoritative table.
   Most consistency logic in ``HyperDataset`` flows outward from the segment
   table.

2. Prefer deriving class tables from segment columns when possible.
   ``add_classes_from_segments`` avoids drift between the label inventory and
   the labels actually used in the dataset.

3. Run ``clean()`` after direct table mutations.
   If you manually replace ``segments`` or modify ids, call ``clean()`` so
   dependent tables are pruned.

4. Use lazy loading for large corpora.
   Constructing a dataset from paths is cheap, and tables are only loaded when
   accessed.

5. Save datasets as YAML bundles for reproducibility.
   A saved ``dataset.yaml`` plus the referenced manifests is much easier to
   version and reuse than ad hoc file lists in recipe code.

6. Be explicit after id-changing transforms.
   Operations such as subsegment sampling and concatenation intentionally drop
   manifests that can no longer be trusted to align with the new segments.

Related Tutorials
-----------------

For the lower-level manifest classes, see :doc:`info_tables`.

For the dense and sparse trial structures, see :doc:`trials`.
