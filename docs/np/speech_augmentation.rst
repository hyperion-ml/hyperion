Speech Augmentation Tutorial
============================

Overview
--------

``SpeechAugment`` is the high-level NumPy augmentation pipeline in
``hyperion.np.augment``. It orchestrates the individual augmenters in this
order:

1. :class:`hyperion.np.augment.SpeedAugment`
2. :class:`hyperion.np.augment.ReverbAugment`
3. :class:`hyperion.np.augment.NoiseAugment`
4. :class:`hyperion.np.augment.CodecAugment`
5. optional second :class:`hyperion.np.augment.CodecAugment` (``transcodec_aug``)

Why this order matters:

* Speed perturbation runs first so all later effects operate on the final time scale.
* Reverb/noise run before codec, so codec artifacts affect already-degraded audio.
* ``transcodec_aug`` runs only if ``codec_aug`` actually applied a codec in that call.


Input/Output Contract
---------------------

Input
~~~~~

* ``x``: 1-D NumPy waveform (``shape == (num_samples,)``).
* ``sample_freq``: sample rate in Hz.
  This is required whenever codec augmentation is enabled and sampled.

Output
~~~~~~

``SpeechAugment.forward`` (or calling the object directly) returns:

* ``y``: augmented waveform
* ``info``: dictionary with augmentation metadata

``info`` always contains ``reverb``, ``noise``, ``codec``, ``transcodec``, and
``sdr``. ``speed`` is present when ``speed_aug`` is configured.


Configuration Structure
-----------------------

Top-level ``SpeechAugment`` config may contain:

* ``speed_aug``
* ``reverb_aug``
* ``noise_aug``
* ``codec_aug``
* ``transcodec_aug``

Each section is optional; include only the stages you want.


Stage Reference
---------------

SpeedAugment (``speed_aug``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Required keys:

* ``speed_prob``: probability in ``[0, 1]``
* ``speed_ratios``: list/sequence of candidate speed factors

Optional keys:

* ``keep_length`` (default: ``False``): crop/pad output to original length


ReverbAugment (``reverb_aug``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Required keys:

* ``reverb_prob``: probability in ``[0, 1]``
* ``rir_types``: dictionary of RIR categories

Optional keys:

* ``max_reverb_context`` (default: ``0``)

Each RIR category entry in ``rir_types`` requires:

* ``weight``
* ``rir_path`` (path to RIR manifest/index, for example ``data/rirs/small_room.csv``)

Optional per-RIR-type keys:

* ``rir_norm``: ``"none"``, ``"max"``, or ``"energy"``
* ``comp_delay`` (default: ``True``)
* ``preload_rirs`` (default: ``True``)


NoiseAugment (``noise_aug``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Required keys:

* ``noise_prob``: probability in ``[0, 1]``
* ``noise_types``: dictionary of noise categories

Each noise category entry in ``noise_types`` requires:

* ``weight``
* ``noise_path`` (path to noise recordings manifest/index, for example ``data/noise/music.csv``)
* ``min_snr``
* ``max_snr``


CodecAugment (``codec_aug`` and ``transcodec_aug``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both sections use the same schema.

Required keys:

* ``codec_prob``: probability in ``[0, 1]``

Optional keys include:

* ``codec_types`` (supported: ``alaw``, ``mulaw``, ``g723_1``, ``g726``, ``g722``,
  ``ac3``, ``mp3``, ``vorbis``, ``opus``)
* ``codec_choice_prob`` (``"uniform"`` or list of probabilities)
* ``mp3_vbr_prob``, ``mp3_cbr``, ``mp3_qscale``, ``mp3_compression``
* ``vorbis_compression``, ``opus_compression``


Quick Start (Dictionary Config)
-------------------------------

.. code-block:: python

   import numpy as np
   from hyperion.np.augment import SpeechAugment

   cfg = {
       "speed_aug": {
           "speed_prob": 0.3,
           "speed_ratios": [0.9, 1.1],
           "keep_length": True,
       },
       "reverb_aug": {
           "reverb_prob": 0.4,
           "max_reverb_context": 3200,
           "rir_types": {
               "small_room": {
                   "weight": 1.0,
                   "rir_path": "data/rirs/small_room.csv",
                   "rir_norm": "energy",
                   "comp_delay": True,
                   "preload_rirs": True,
               },
           },
       },
       "noise_aug": {
           "noise_prob": 0.5,
           "noise_types": {
               "music": {
                   "weight": 0.6,
                   "noise_path": "data/noise_music/recordings.csv",
                   "min_snr": 5,
                   "max_snr": 20,
               },
               "babble": {
                   "weight": 0.4,
                   "noise_path": "data/noise_babble/recordings.csv",
                   "min_snr": 0,
                   "max_snr": 15,
               },
           },
       },
       "codec_aug": {
           "codec_prob": 0.35,
           "codec_types": ["mulaw", "g722", "mp3", "opus"],
           "codec_choice_prob": "uniform",
       },
       "transcodec_aug": {
           "codec_prob": 0.2,
           "codec_types": ["mp3", "vorbis"],
       },
   }

   aug = SpeechAugment.create(cfg, random_seed=1234)

   fs = 16000
   x = np.random.randn(fs * 3).astype("float32")

   y, info = aug(
       x,
       sample_freq=fs,
       enable_tel_codecs=True,
       enable_media_codecs=True,
       enable_transcodec=True,
   )

   print("output shape:", y.shape)
   print("sdr:", info["sdr"])
   print("reverb:", info["reverb"])
   print("noise:", info["noise"])
   print("codec:", info["codec"])
   print("transcodec:", info["transcodec"])


YAML Config Example
-------------------

You can store the same config in YAML and load it with
``SpeechAugment.create("speech_aug.yaml")``:

.. code-block:: yaml

   speed_aug:
     speed_prob: 0.3
     speed_ratios: [0.9, 1.1]
     keep_length: true

   reverb_aug:
     reverb_prob: 0.4
     max_reverb_context: 3200
     rir_types:
       small_room:
         weight: 1.0
         rir_path: "data/rirs/small_room.csv"
         rir_norm: "energy"
         comp_delay: true
         preload_rirs: true

   noise_aug:
     noise_prob: 0.5
     noise_types:
       music:
         weight: 0.6
         noise_path: "data/noise_music/recordings.csv"
         min_snr: 5
         max_snr: 20
       babble:
         weight: 0.4
         noise_path: "data/noise_babble/recordings.csv"
         min_snr: 0
         max_snr: 15

   codec_aug:
     codec_prob: 0.35
     codec_types: ["mulaw", "g722", "mp3", "opus"]
     codec_choice_prob: "uniform"

   transcodec_aug:
     codec_prob: 0.2
     codec_types: ["mp3", "vorbis"]

.. code-block:: python

   from hyperion.np.augment import SpeechAugment
   aug = SpeechAugment.create("speech_aug.yaml", random_seed=1234)


Runtime Control Flags
---------------------

At call time you can gate codec families:

.. code-block:: python

   # only media codecs
   y, info = aug(
       x,
       sample_freq=16000,
       enable_tel_codecs=False,
       enable_media_codecs=True,
       enable_transcodec=True,
   )

   # disable transcodec pass
   y, info = aug(
       x,
       sample_freq=16000,
       enable_tel_codecs=True,
       enable_media_codecs=True,
       enable_transcodec=False,
   )


Interpreting ``info`` Safely
----------------------------

.. code-block:: python

   speed_ratio = info.get("speed", {}).get("speed_ratio", 1.0)
   rir_type = info["reverb"]["rir_type"]          # None if not applied
   noise_type = info["noise"]["noise_type"]       # None if not applied
   codec_type = info["codec"]["codec_type"]       # None if not applied
   transcodec_type = info["transcodec"]["codec_type"]


Reproducibility and Reseeding
-----------------------------

Create-time reproducibility:

.. code-block:: python

   aug = SpeechAugment.create(cfg, random_seed=1234)

Manual reseeding:

.. code-block:: python

   aug.reseed(987654)

If you use custom PyTorch DataLoader code, reseed each worker to avoid
identical streams across workers:

.. code-block:: python

   def worker_init_fn(worker_id):
       import torch
       seed = int(torch.initial_seed() % (2**32))
       dataset.augmenter.reseed(seed)


Common Pitfalls
---------------

* **Wrong input shape**: use 1-D waveforms only.
* **Missing sample rate with codec enabled**: provide ``sample_freq``.
* **Invalid probability/weight ranges**: probabilities must be in ``[0, 1]``;
  per-type weights must be non-negative and sum to a positive value.
* **Noise/RIR source mismatch**: ensure paths use formats expected by the
  underlying readers (for example CSV manifests if that is what your setup uses).


Context Requirement for Reverb
------------------------------

If your model needs to know left context introduced by reverberation, use:

.. code-block:: python

   left_context = aug.max_reverb_context

This is useful when aligning segment extraction with augmentation context.


See Also
--------

* :doc:`/numpy`
* :doc:`/getting-started`
