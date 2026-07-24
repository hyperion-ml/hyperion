Segment Datasets
================

The segment-oriented datasets in ``hyperion.torch.data`` load audio or feature
sequences using segment metadata. A segment table normally contains an ``id``
column and, for audio segments, recording/time-mark columns such as
``recording``, ``start``, and ``duration``.

Dataset selection
-----------------

``AudioDataset`` is the current dataset for audio stored in a
``HyperDataset``. ``LegacyAudioDataset`` reads separate recording and segment
manifest files and remains available for older training recipes.
``DINOAudioDataset`` extends the legacy audio dataset with teacher and student
chunks. ``FeatSeqDataset`` reads feature sequences and uses ``num_frames`` for
segment lengths.

All datasets implement the PyTorch ``Dataset`` interface. Dataset items are
dictionaries whose input keys depend on the dataset: ``AudioDataset`` returns
``audio``, feature and legacy audio datasets use ``x`` or augmented ``x_*``
keys, and metadata is returned under ``id`` or ``seg_id``. Requested class
labels and other segment attributes are added to the same dictionary.

Current audio dataset
~~~~~~~~~~~~~~~~~~~~~

``AudioDataset`` loads a ``HyperDataset`` directory or file. Tokenizers,
augmentations, resampling, and selected segment attributes are configured in
the constructor:

.. code-block:: python

   from hyperion.torch.data.audio_dataset import AudioDataset

   dataset = AudioDataset(
       dataset_path="data/train_hyper_dataset",
       extra_attrs=["speaker"],
       aug_cfgs=["conf/train_aug.yaml"],
       num_augs=2,
       target_sample_freq=16000,
   )

   item = dataset["segment-0001"]
   waveform = item["audio"]

The dataset can be passed directly to a ``DataLoader``. Use a sequential
sampler to provide segment IDs in the desired batch order:

.. code-block:: python

   from torch.utils.data import DataLoader
   from hyperion.torch.data import SegSamplerFactory

   sampler = SegSamplerFactory.create(
       dataset,
       sampler_type="seg_sampler",
       min_batch_size=16,
       max_batch_length=240.0,
       shuffle=True,
   )
   loader = DataLoader(
       dataset,
       batch_sampler=sampler,
       collate_fn=dataset.collate,
   )

Legacy audio dataset
~~~~~~~~~~~~~~~~~~~~

Use ``LegacyAudioDataset`` when recordings and segments are stored in separate
manifest files:

.. code-block:: python

   from hyperion.torch.data.legacy_audio_dataset import LegacyAudioDataset

   dataset = LegacyAudioDataset(
       recordings_file="data/recordings.csv",
       segments_file="data/segments.csv",
       class_names=["speaker"],
       class_files=["data/speaker.csv"],
       return_segment_info=["speaker"],
       target_sample_freq=16000,
   )

   item = dataset["segment-0001"]
   speaker_index = item["speaker"]

``DINOAudioDataset`` uses the same manifest inputs and returns teacher/student
chunks for self-supervised training:

.. code-block:: python

   from hyperion.torch.data.dino_audio_dataset import DINOAudioDataset

   dataset = DINOAudioDataset(
       recordings_file="data/recordings.csv",
       segments_file="data/segments.csv",
       teacher_aug_cfg="conf/teacher_aug.yaml",
       student_aug_cfg="conf/student_aug.yaml",
       teacher_chunk_length=4.0,
       student_chunk_length=2.0,
   )

Feature sequence dataset
~~~~~~~~~~~~~~~~~~~~~~~~

``FeatSeqDataset`` reads feature matrices from a random-access reader and
associates each sequence with a row in a ``SegmentSet``. The segment table must
contain ``num_frames`` unless a separate duration file is supplied:

.. code-block:: python

   from hyperion.torch.data.feat_seq_dataset import FeatSeqDataset

   dataset = FeatSeqDataset(
       feat_file="csv:data/feats.csv",
       segments_file="data/feats_segments.csv",
       class_names=["speaker"],
       class_files=["data/speaker.csv"],
       return_segment_info=["speaker"],
       transpose_input=True,
   )

   item = dataset["segment-0001"]
   features = item["x"]
   num_frames = item["x_lengths"]

Dataset APIs
------------

.. autoclass:: hyperion.torch.data.audio_dataset.AudioDataset
   :members:
   :show-inheritance:

.. autoclass:: hyperion.torch.data.legacy_audio_dataset.LegacyAudioDataset
   :members:
   :show-inheritance:

.. autoclass:: hyperion.torch.data.dino_audio_dataset.DINOAudioDataset
   :members:
   :show-inheritance:

.. autoclass:: hyperion.torch.data.feat_seq_dataset.FeatSeqDataset
   :members:
   :show-inheritance:
