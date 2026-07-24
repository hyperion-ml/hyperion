Working With Info Tables
========================

.. currentmodule:: hyperion.utils

Overview
--------

``InfoTable`` and its child classes are the main manifest layer used throughout
Hyperion. They wrap a pandas ``DataFrame`` but impose a few conventions that
make dataset metadata easier to manipulate consistently:

* every table has an ``id`` column
* ``id`` is also used as the pandas index
* common operations such as ``load``, ``save``, ``filter``, ``split``, and
  ``cat`` are available across the table types
* domain-specific child classes add validation and helper methods for their
  own manifest schema

This guide focuses on the table types that appear most often in package-level
data preparation and evaluation workflows:

* :class:`SegmentSet`
* :class:`RecordingSet`
* :class:`EnrollmentMap`
* :class:`ClassInfo`

It also shows how the same patterns extend to :class:`FeatureSet`,
:class:`VADSet`, :class:`ImageSet`, :class:`VideoSet`, and
:class:`DiarizationSet`.

Representative Example Tables
-----------------------------

The schemas vary by dataset and task. That is normal. ``InfoTable`` does not
force every dataset to have exactly the same columns. It standardizes table
mechanics and the key columns required by each subclass.

The following illustrative segment manifests show the main pattern:

.. code-block:: text

   # segments.csv: utterances with speaker labels
   id,video_id,speaker,gender,nationality,language_est,language_est_conf,duration,corpusid,dataset,source_type
   id10001-1zcIwhmdeo4-00001,1zcIwhmdeo4,id10001,m,Ireland,english,0.99,8.1200625,voxceleb,voxceleb1,afv

.. code-block:: text

   # segments.csv: time-marked excerpts from recordings
   id,recording,start,duration,transcript,transcript_no_punc,speaker,book,language,...,voxprofile_narrow_accent,...
   libriheavy-100-sea_fairies_..._0,libriheavy-100-sea_fairies_...,243.919,7.359999999999985,"The little girl was thoughtful...",...,eng,...

.. code-block:: text

   # segments.csv: clips with transcript and demographic labels
   id,speaker,transcript,duration,language,age_decade,gender,gender_extended,sentence_domain,up_votes,down_votes,...,voxprofile_narrow_accent,...
   cv-common_voice_en_100038,cv-04960d53...,Why does Melissandre look like she wants to consume Jon Snow...,7.704,eng,fourties,m,male_masculine,,2,0,...

The same idea holds for recordings, enrollment maps, and class metadata:

.. code-block:: text

   # recordings.csv
   id,storage_path,duration,sample_freq
   id10001-1zcIwhmdeo4-00001,/export/corpora5/VoxCeleb1_v2/wav/id10001/1zcIwhmdeo4/00001.wav,8.1200625,16000

.. code-block:: text

   # enrollment.csv
   modelid,segmentid
   id10001-1zcIwhmdeo4-00001,id10001-1zcIwhmdeo4-00001

.. code-block:: text

   # speaker.csv
   id,vgg_id,gender,nationality,class_idx,weights
   id10001,A.J._Buckley,m,Ireland,0,0.005

.. code-block:: text

   # gender.csv
   id,class_idx,weights
   m,0,0.5

Core ``InfoTable`` Behavior
---------------------------

All child classes inherit the same core mechanics.

Loading and inspection
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hyperion.utils import SegmentSet

   segments = SegmentSet.load("data/voxceleb1_test/segments.csv")
   print(type(segments))
   print(len(segments))
   print(list(segments.columns))
   print(segments.df.head(2))

Important details:

* ``segments.df`` is the underlying pandas ``DataFrame``.
* ``segments.index`` is the ``id`` index.
* ``segments["speaker"]`` returns a pandas ``Series``.
* ``segments[["id", "speaker", "duration"]]`` returns another ``SegmentSet``
  because the result still contains ``id``.

Indexing
~~~~~~~~

``InfoTable`` exposes pandas-style accessors and preserves the subclass when
the result is still a valid manifest.

.. code-block:: python

   first_two = segments.iloc[:2]
   single_duration = segments.at["id10001-1zcIwhmdeo4-00001", "duration"]
   spk_column = segments["speaker"]

   assert isinstance(first_two, SegmentSet)
   assert isinstance(spk_column, type(segments.df["speaker"]))

Filtering
~~~~~~~~~

There are three common ways to filter rows:

.. code-block:: python

   # By id
   subset = segments.filter(
       items=["id10001-1zcIwhmdeo4-00001", "id10001-1zcIwhmdeo4-00002"]
   )

   # By another column
   male_segments = segments.filter(items=["m"], by="gender")

   # By boolean expression
   long_segments = segments.filter(predicate="duration > 8")

   # By callable
   english_segments = segments.filter(
       predicate=lambda df: df["language_est"] == "english"
   )

Use ``keep=False`` to drop the matching rows instead of keeping them.

Splitting and concatenation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``split`` is useful for sharding large manifests for parallel processing.

.. code-block:: python

   shard_1 = segments.split(1, 4)
   shard_2 = segments.split(2, 4)
   merged = SegmentSet.cat([shard_1, shard_2])

If you want all examples from the same group to stay together, pass
``group_by``:

.. code-block:: python

   # Keep all rows from the same speaker in the same shard.
   shard = segments.split(1, 8, group_by="speaker")

Saving
~~~~~~

.. code-block:: python

   subset.save("tmp/segments_subset.csv")
   subset.save("tmp/segments_subset.tsv")

CSV vs TSV is inferred from the extension unless you pass ``sep=...``.

``SegmentSet``
--------------

Use :class:`SegmentSet` for per-segment metadata. Each row corresponds to one
speech segment, utterance, or clip.

What is required
~~~~~~~~~~~~~~~~

Only ``id`` is required by the class itself. Other columns depend on the
dataset.

Typical optional columns include:

* ``recording``, ``start``, ``duration`` for segmentation over a larger source
  recording
* labels such as ``speaker``, ``gender``, ``language``, ``age_decade``
* transcripts and quality annotations
* dataset provenance fields such as ``corpusid`` and ``dataset``

Two important schema patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pattern 1: one row is a slice of a larger recording

This is what the time-marked excerpt example shows. The manifest contains
``recording``, ``start``, and ``duration``.

.. code-block:: python

   from hyperion.utils import SegmentSet

   segments = SegmentSet.load("data/libriheavy_50k_annotated/segments.csv")
   print(segments.has_time_marks)      # True
   print(segments.has_recording_ids)   # True

   seg_ids = segments.df["id"].head(2).tolist()
   print(segments.recording(seg_ids))
   print(segments.recording_time_marks(seg_ids))

Pattern 2: one row is already a standalone clip

This is what the VoxCeleb and Common Voice examples look like. They may not
have ``recording`` or ``start`` columns because the segment id already
identifies a standalone audio file or clip.

.. code-block:: python

   segments = SegmentSet.load("data/voxceleb1_test/segments.csv")
   print(segments.has_time_marks)      # False
   print(segments.recording(["id10001-1zcIwhmdeo4-00001"]))
   # Falls back to the segment id when there is no "recording" column.

Note that recipe-specific columns stay recipe-specific. For example the
VoxCeleb segment table uses ``video_id`` rather than ``video``. ``SegmentSet``
does not rename that column automatically.

Sampling subsegments
~~~~~~~~~~~~~~~~~~~~

If a ``SegmentSet`` already has durations, you can sample smaller random
subsegments from each row:

.. code-block:: python

   subsegments = segments.sample_random_subsegments(
       subsegments_per_segment=2,
       min_duration=2.0,
       max_duration=4.0,
       seg_suffix="sub",
   )

This is useful when training from fixed-duration chunks derived from a larger
segment inventory.

Practical label filtering
~~~~~~~~~~~~~~~~~~~~~~~~~

Because labels are just columns, many operations are simple:

.. code-block:: python

   segments = SegmentSet.load("data/cv_eng_train_annotated/segments.csv")

   english = segments.filter(items=["eng"], by="language")
   male = segments.filter(items=["m"], by="gender")
   long_english = segments.filter(
       predicate=lambda df: (df["language"] == "eng") & (df["duration"] >= 5.0)
   )

   # Keep only the columns needed by a downstream stage.
   compact = segments.filter(columns=["speaker", "language", "duration"])

``RecordingSet``
----------------

Use :class:`RecordingSet` for per-recording storage metadata. Each row points
to an audio file.

Recording schema
~~~~~~~~~~~~~~~~

All three examples you provided use the same core columns:

* ``id``
* ``storage_path``
* ``duration``
* ``sample_freq``

.. code-block:: python

   from hyperion.utils import RecordingSet

   recordings = RecordingSet.load("data/voxceleb1_test/recordings.csv")
   print(recordings.df.head(2))

The value in ``id`` usually matches the segment id when each segment is stored
as a separate file. When segments come from a longer recording, the
``SegmentSet.recording`` column points into this table.

Core operations
~~~~~~~~~~~~~~~

.. code-block:: python

   recordings = RecordingSet.load("data/libriheavy_50k_annotated/recordings.csv")

   subset = recordings.filter(items=recordings.df["id"].head(10).tolist())
   shard = recordings.split(1, 8)
   combined = RecordingSet.cat([recordings.split(1, 2), recordings.split(2, 2)])

   subset.save("tmp/recordings_subset.csv")

Estimating duration and sample rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``duration`` and ``sample_freq`` are missing or need recomputation:

.. code-block:: python

   recordings = RecordingSet.load("data/my_dataset/recordings.csv")
   recordings.get_durations(num_threads=8)

This method inspects the audio files and writes the resulting ``duration`` and
``sample_freq`` columns back into the table in memory.

Saving and loading CSV manifests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use CSV for recording manifests:

.. code-block:: python

   recordings = RecordingSet.load("data/my_dataset/recordings.csv")
   recordings.save("tmp/recordings.csv")

The table stores the stable ``id`` and ``storage_path`` columns, with optional
duration and sample-rate metadata.

``EnrollmentMap``
-----------------

Use :class:`EnrollmentMap` when a model is enrolled from one or more segments.
Each row says that model ``id`` is associated with one ``segmentid``.

Enrollment-map schema
~~~~~~~~~~~~~~~~~~~~~

On disk, the CSV commonly looks like this:

.. code-block:: text

   modelid,segmentid
   id10001-1zcIwhmdeo4-00001,id10001-1zcIwhmdeo4-00001

When you load it, the class normalizes ``modelid`` to the in-memory ``id``
column used by ``InfoTable``.

.. code-block:: python

   from hyperion.utils import EnrollmentMap

   enroll = EnrollmentMap.load("data/voxceleb1_test/enrollment.csv")
   print(list(enroll.columns))
   # ['id', 'segmentid']

Grouping by model
~~~~~~~~~~~~~~~~~

This table often has repeated model ids because one model can have multiple
enrollment segments.

.. code-block:: python

   unique_modelids, segment_to_model_idx = enroll.model_idx()
   print(unique_modelids[:5])
   print(segment_to_model_idx[:5])

If you already have a reference model order, pass it explicitly:

.. code-block:: python

   desired_order = ["model_a", "model_b", "model_c"]
   idx = enroll.model_idx(desired_order)
   # rows whose model id is not present in desired_order get -1

Useful helpers
~~~~~~~~~~~~~~

.. code-block:: python

   shard = enroll.split(1, 4)
   merged = EnrollmentMap.cat([enroll.split(1, 2), enroll.split(2, 2)])
   unique_df = enroll.get_unique_modelid_df()

Saving and NIST compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default ``save`` writes ``modelid`` on disk for compatibility with common
SRE-style files:

.. code-block:: python

   enroll.save("tmp/enrollment.csv")

If you want the internal ``id`` column name on disk instead:

.. code-block:: python

   enroll.save("tmp/enrollment_internal.csv", nist_compatible=False)

``ClassInfo``
-------------

Use :class:`ClassInfo` for a label vocabulary or classification target space.
Each row is one class, not one segment.

This is the most important conceptual point:

* a ``SegmentSet`` row is an example
* a ``ClassInfo`` row is a class label that examples can point to

For example:

* ``speaker.csv`` defines the speaker classes
* ``gender.csv`` defines the gender classes
* ``language.csv`` defines the language classes
* ``age_decade.csv`` defines the age-decade classes
* ``voxprofile_narrow_accent.csv`` defines the accent classes

The corresponding ``SegmentSet`` then contains columns with the same semantic
meaning:

* ``segments.df["speaker"]`` contains ids from ``speaker.csv``
* ``segments.df["gender"]`` contains ids from ``gender.csv``
* ``segments.df["language"]`` contains ids from ``language.csv``
* ``segments.df["age_decade"]`` contains ids from ``age_decade.csv``
* ``segments.df["voxprofile_narrow_accent"]`` contains ids from
  ``voxprofile_narrow_accent.csv``

Representative schemas
~~~~~~~~~~~~~~~~~~~~~~

Speaker labels can carry extra metadata:

.. code-block:: text

   id,vgg_id,gender,nationality,class_idx,weights
   id10001,A.J._Buckley,m,Ireland,0,0.005

Other class vocabularies are smaller and often only need:

.. code-block:: text

   id,class_idx,weights
   m,0,0.5
   f,1,0.5

Class-info loading and inspection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hyperion.utils import ClassInfo

   speaker_info = ClassInfo.load("data/voxceleb1_test/speaker.csv")
   gender_info = ClassInfo.load("data/cv_eng_train_annotated/gender.csv")
   accent_info = ClassInfo.load("data/cv_eng_train_annotated/voxprofile_narrow_accent.csv")

   print(speaker_info.num_classes)
   print(gender_info.df)
   print(accent_info.weights(["east-asia", "english"]))

Weights and class rebalancing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ClassInfo`` maintains a normalized ``weights`` column.

.. code-block:: python

   import numpy as np

   speaker_info = ClassInfo.load("data/voxceleb1_test/speaker.csv")

   # Uniform weighting
   speaker_info.set_uniform_weights()

   # Use custom weights and re-normalize
   speaker_info.set_weights(np.ones(len(speaker_info.df)))

   # Down-weight or remove classes from training
   speaker_info.set_zero_weight(["id10001"])

   # Apply temperature-like sharpening or flattening
   speaker_info.exp_weights(0.5)

Rebuilding class indices after filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After filtering classes, use ``rebuild_idx=True`` when the downstream code
expects contiguous class indices from ``0`` to ``num_classes - 1``.

.. code-block:: python

   speaker_subset = speaker_info.filter(
       items=["id10001", "id10002", "id10003"],
       rebuild_idx=True,
   )

Mapping a ``SegmentSet`` to ``ClassInfo``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the most common workflow.

.. code-block:: python

   from hyperion.utils import SegmentSet, ClassInfo

   segments = SegmentSet.load("data/cv_eng_train_annotated/segments.csv")
   gender_info = ClassInfo.load("data/cv_eng_train_annotated/gender.csv")

   valid = segments.df["gender"].notna()
   gender_ids = segments.df.loc[valid, "gender"]
   gender_class_idx = gender_info.df.loc[gender_ids, "class_idx"].to_numpy()

   segments.df.loc[valid, "gender_class_idx"] = gender_class_idx

The same pattern applies to speaker, language, age, accent, or any other
classification target stored as a segment column plus a matching ``ClassInfo``
table.

Putting The Tables Together
---------------------------

A typical pipeline uses several table types at once.

Example 1: start from segments and recover recording paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hyperion.utils import SegmentSet, RecordingSet

   segments = SegmentSet.load("data/libriheavy_50k_annotated/segments.csv")
   recordings = RecordingSet.load("data/libriheavy_50k_annotated/recordings.csv")

   marks = segments.recording_time_marks(segments.df["id"].head(3).tolist())
   storage_paths = recordings.df.loc[marks["recording"], "storage_path"]

Example 2: build a speaker-class training subset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hyperion.utils import SegmentSet, ClassInfo

   segments = SegmentSet.load("data/voxceleb1_test/segments.csv")
   speaker_info = ClassInfo.load("data/voxceleb1_test/speaker.csv")

   # Keep only speakers that appear at least 5 times.
   counts = segments.df["speaker"].value_counts()
   keep_speakers = counts[counts >= 5].index.tolist()

   filtered_segments = segments.filter(items=keep_speakers, by="speaker")
   filtered_speaker_info = speaker_info.filter(
       items=keep_speakers,
       rebuild_idx=True,
   )

Example 3: align enrollment rows with segment metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hyperion.utils import EnrollmentMap, SegmentSet

   enroll = EnrollmentMap.load("data/voxceleb1_test/enrollment.csv")
   segments = SegmentSet.load("data/voxceleb1_test/segments.csv")

   enrollment_segments = segments.df.loc[enroll.df["segmentid"]]
   enrollment_speakers = enrollment_segments["speaker"].to_numpy()

Other ``InfoTable`` Children
----------------------------

The same ideas extend to the other manifest classes.

``FeatureSet``
~~~~~~~~~~~~~~

Use :class:`FeatureSet` for CSV-indexed feature archives with ``storage_path``
and optional frame offsets.

.. code-block:: python

   from hyperion.utils import FeatureSet

   feats = FeatureSet.load("data/my_dataset/features.csv")
   feats.add_prefix_to_storage_path("/mnt/features")
   feats.save("tmp/features.csv")

``VADSet``
~~~~~~~~~~

``VADSet`` is a ``FeatureSet`` specialized for VAD storage.

.. code-block:: python

   from hyperion.utils import VADSet

   vad = VADSet.load("data/my_dataset/vad.csv")
   vad = vad.filter(items=vad.df["id"].head(100).tolist())

``ImageSet``
~~~~~~~~~~~~

Use ``ImageSet`` when each row points to an image file.

.. code-block:: python

   from hyperion.utils import ImageSet

   images = ImageSet.from_dict(
       {"id": ["img1", "img2"], "storage_path": ["a.jpg", "b.jpg"]}
   )
   first = images.filter(items=["img1"])

``VideoSet``
~~~~~~~~~~~~

Use ``VideoSet`` for audiovisual files. It can inspect media metadata and add
``duration``, ``video_duration``, ``sample_freq``, and ``fps``.

.. code-block:: python

   from hyperion.utils import VideoSet

   videos = VideoSet.from_dict(
       {"id": ["vid1"], "storage_path": ["/path/to/video.mp4"]}
   )
   videos.get_metadata(num_threads=4)

``DiarizationSet``
~~~~~~~~~~~~~~~~~~

Use ``DiarizationSet`` for diarization-related artifacts such as RTTM files.

.. code-block:: python

   from hyperion.utils import DiarizationSet

   diar = DiarizationSet.from_dict(
       {"id": ["utt1"], "storage_path": ["rttm/utt1.rttm"]}
   )
   diar.add_prefix_to_storage_path("/mnt/exp/")

Practical Guidelines
--------------------

1. Use the most specific subclass available. That gives you schema validation
   and helper methods.
2. Keep ``id`` stable. Other tables usually refer to it directly.
3. Treat ``SegmentSet`` as example-level metadata and ``ClassInfo`` as
   label-space metadata.
4. When you filter a ``ClassInfo`` table that will feed a classifier head, use
   ``rebuild_idx=True`` if the output needs contiguous class ids.
5. When you split large tables for parallel work, use ``group_by`` whenever a
   whole group must stay together, such as all segments from the same speaker.
6. Use the ``.df`` attribute whenever you need raw pandas functionality beyond
   the helper methods exposed by the class.

See Also
--------

* :doc:`utils`
* :class:`InfoTable`
* :class:`SegmentSet`
* :class:`RecordingSet`
* :class:`EnrollmentMap`
* :class:`ClassInfo`
